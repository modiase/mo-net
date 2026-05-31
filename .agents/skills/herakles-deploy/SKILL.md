---
name: herakles-deploy
description: How to package the mo-net training image and launch jobs on herakles via just package + sbatch. Use whenever the user asks about deploying, packaging, pushing images, running spikes, or submitting sweeps on herakles.
allowed-tools: [Bash, Read, Edit, Write]
---

# Herakles deploy: just package + sbatch

The deploy loop has two halves: **build + push image** (`just package`), and **run a job** (`sbatch` invoked over ssh).

## Prerequisites

Two env vars must be set in `.envrc.local` (already gitignored via `**/*.local*`):

```bash
export REGISTRY_HOST=localhost:5000       # the herakles registry:2 endpoint as seen from herakles
export BUILD_HOST=herakles                # ssh target for the streamLayeredImage execution step
```

Plus on **pallas/hestia** (whatever the dev host is), the `heraklesBuildServer` dotfile module must be attached so the nix-daemon has `nix.buildMachines` configured for ssh-ng → herakles. Without that, `nix build` will fail to find a builder and fall back to local (which can't produce x86_64-linux from darwin). See `~/Dotfiles/systems/hestia/configuration.nix` line ~40 for the import pattern; pallas needs the same wiring.

`heraklesBuildServer` also handles the harder part: nix-daemon (running as root) needs SSH access to herakles. The module deploys an SSH identity at `/root/.ssh/...` via sops and adds a system-level ssh_config entry so root's ssh uses it. CLI `--builders` from a non-trusted user is NOT a substitute — the daemon silently drops it.

## Build + push: `just package`

```
just package                            # → $REGISTRY_HOST/mo-net-cuda:<ts>-<sha>[-dirty]
just package --no-cuda                  # cpu variant: mo-net:<ts>-<sha>[-dirty]
just package my-registry.io             # → my-registry.io/mo-net-cuda:<ts>-<sha>[-dirty]
just package my-registry.io/mo:v1.2     # exact ref, no auto-tag
```

Tag format: `<UTC YYYYMMDD-HHMMSS>-<short-sha>[-dirty]`. Sortable chronologically, pinned to a commit, `-dirty` suffix flags uncommitted changes.

What the recipe does:

1. `nix build .#packages.x86_64-linux.mo-net-cuda-image` — routes through the daemon-configured `buildMachines`, runs the build on herakles, no source rsync (nix-daemon protocol ships inputs over ssh-ng).
2. `ssh $BUILD_HOST "<store-path> | skopeo copy docker-archive:/dev/stdin docker://<ref>"` — runs the streamLayeredImage script on herakles (it's a linux-only script that can't execute on darwin), pipes its tar straight into skopeo, which pushes to the registry. No `docker load` round-trip, no tarball on disk.

The recipe ends with `Done: pushed <ref>`. Copy that ref into the sbatch command.

**Before pressing go on new files**: `git add -A` (don't commit, just stage). flit_core's sdist generation only includes git-tracked files, so any _new_ file (e.g. a freshly-created `mo_net/scripts/foo.py`) gets silently dropped from the built wheel — manifests at runtime as `ModuleNotFoundError`. Modified files of already-tracked paths are fine without staging.

## Run a one-off spike

```bash
ssh herakles "sbatch \
  --job-name=w2v-spike \
  --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=60:00 \
  --output=/data/mo-net/sweeps/w2v/logs/%j.out \
  --container-image=registry.herakles.local/mo-net-cuda:20260528-235451-ec771cc \
  --container-mounts=/data/mo-net/cache:/var/lib/mo-net/cache:rw,/data/mo-net/sweeps/w2v/spike-out:/var/lib/mo-net/data \
  --container-workdir=/workspace \
  --container-env=MO_NET_LOKI_URL,SLURM_JOB_ID \
  --wrap='mo-net-train -m mo_net.samples.word2vec train \
            --model-type cbow --softmax-strategy full \
            --embedding-dim 256 --learning-rate 1e-4 --num-epochs 8 \
            --vocab-size 3000 --batch-size 4096 --subsample-t 1e-5 \
            --logging-backend-connection-string postgresql://mo_net@127.0.0.1:5432/mo_net'"
```

Key points:

- **`--container-image`** uses `docker://<registry>#<repo>:<tag>` — the `docker://` scheme is required and the `#` separator between registry and repo is mandatory when the registry has a port (`localhost:5000`). Without `#`, enroot's parser splits on the wrong colon and fails with `Invalid image reference`. skopeo wants the same `docker://` prefix; only pyxis args for registries without ports tolerate the shorthand.
- **`--wrap`** turns the inline shell command into the job body. Avoids needing a script file for one-offs.
- **Image refs encode the commit + timestamp**, so the sbatch line is itself a reproducibility record.

## Run an array sweep

Inline isn't ergonomic for array jobs (grid expansion is bash logic that doesn't fit `--wrap='…'` cleanly). Use a script on herakles in `/tmp/mo-net-stage/sweeps/w2v/`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=w2v-sweepN
#SBATCH --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=180:00
#SBATCH --array=0-N%2
#SBATCH --output=/data/mo-net/sweeps/w2v/logs/%A_%a.out

set -euo pipefail
# grid decoder from $SLURM_ARRAY_TASK_ID → (model, softmax, embed, lr, vocab)
# … usual pattern …
srun \
  --container-image=registry.herakles.local/mo-net-cuda:<pinned-tag> \
  --container-mounts=…:rw,$JOB_DIR:/var/lib/mo-net/data \
  --container-workdir=/workspace \
  --container-env=MO_NET_LOKI_URL,MO_NET_LOKI_LABELS,SLURM_JOB_ID,SLURM_ARRAY_JOB_ID,SLURM_ARRAY_TASK_ID \
  mo-net-train -m mo_net.samples.word2vec train …
```

Pin the image tag in the script. Re-deploying images mid-sweep would mean different tasks ran on different code — `:dirty` aside, that's exactly the thing tag-pinning fixes.

## Hugging Face datasets + cache pre-warming

`HF_HOME` is baked into the image at `/var/lib/mo-net/cache/huggingface` (see `flake.nix`), so anything `datasets.load_dataset` materialises lands under the persistent cache mount automatically — no `--container-env=HF_HOME` needed.

**One-time host prep** (cache directory must exist and be user-writable):

```bash
ssh herakles 'sudo mkdir -p /data/mo-net/cache && sudo chown $USER /data/mo-net/cache'
```

`/data/mo-net/cache` is on `/data` (persistent across reboots), unlike the legacy `/tmp/mo-net-stage/mo-net-cache` which tmpfs wipes — multi-GB HF dataset downloads would have to repeat every reboot otherwise.

**Pre-warm before launching a sweep** (avoid burning a GPU node on downloads):

```bash
# FineWeb sample-10BT — ~28 GB on disk, 10B tokens.
ssh herakles 'nix develop -c mo-net-prewarm \
  "hf://HuggingFaceFW/fineweb?config=sample-10BT&split=train&text_field=text"'

# Cheap sanity-check it landed:
ssh herakles 'du -sh /data/mo-net/cache/huggingface'
```

**Pass an HF corpus to a training job** — `--corpus-url` accepts any scheme `mo_net.resources` knows:

```bash
mo-net-train -m mo_net.samples.word2vec train \
  --corpus-url 'hf://HuggingFaceFW/fineweb?config=sample-10BT&split=train&text_field=text' \
  --model-type cbow --softmax-strategy negative-sampling \
  --vocab-size 50000 --batch-size 4096 …
```

## Inspecting the registry

From herakles:

```
curl -s http://localhost:5000/v2/_catalog
curl -s http://localhost:5000/v2/mo-net-cuda/tags/list
nix shell nixpkgs#skopeo --command skopeo list-tags docker://localhost:5000/mo-net-cuda
```

## Common pitfalls

- **`Failed to find a machine for remote build!`** — nix-daemon either doesn't have `buildMachines` configured (attach `heraklesBuildServer` to the host's dotfile config) or can't ssh to herakles as root (sops-deployed SSH key missing). The error message about feature mismatch is misleading; the real cause is usually SSH-from-daemon failing silently.
- **`unknown transport "registry.herakles.local/..."`** — skopeo needs an explicit `docker://` prefix on its args. `just package` already does this; if you're invoking skopeo directly, remember the prefix.
- **`no such host: registry.herakles.local`** — resolution failure on the host running skopeo or pyxis. Either add to `/etc/hosts` on herakles (`127.0.0.1 registry.herakles.local`) or set `REGISTRY_HOST=localhost:5000` and accept the less-self-documenting tag prefix.
- **pyxis `Invalid image reference: docker://localhost:5000/...`** — missing `#` separator between registry and repo. With a port in the registry the parser can't tell the port colon from the tag colon. Write `docker://localhost:5000#mo-net-cuda:tag`, not `docker://localhost:5000/mo-net-cuda:tag`.
- **pyxis `TLS connect error: wrong version number`** — enroot is forcing HTTPS and registry:2 is plain HTTP. Set `ENROOT_ALLOW_HTTP yes` in the herakles enroot config (NixOS module level — single line). Until that ships, a one-off pre-import works: `ssh herakles 'ENROOT_ALLOW_HTTP=yes enroot import -o /tmp/.../image.sqsh docker://localhost:5000#mo-net-cuda:tag'` and then `--container-image=/tmp/.../image.sqsh`.
- **Sweep image suddenly changes mid-run** — happens if the sbatch script references a mutable tag (e.g. `:latest`). Always pin to a specific timestamp-sha tag.

## Quick checklist before submitting a sweep

1. `git status` is clean (`-dirty` in your image tag flags this in the registry too, but a clean working tree is the only way the tag really means a specific commit).
2. `just package` succeeded and printed a ref.
3. The ref is in the sbatch script (or `--container-image=...` arg) — not a stale tag.
4. Postgres (`psql -h 127.0.0.1 -U mo_net -d mo_net -c "\\dt"`) and Loki (`curl http://localhost:3100/loki/api/v1/labels`) are responsive on herakles, or expect silent logging gaps.
