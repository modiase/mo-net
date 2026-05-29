# Hand-off: enroot config change needed on herakles

**Context for the dotfiles Claude session.** This is one piece of plumbing the
`mo-net` workflow needs from herakles' system config but can't change itself.
Once landed, the local-registry → pyxis → enroot pipeline works directly from
sbatch with no manual sqsh staging.

## What needs to change

On the **herakles** NixOS module, add `ENROOT_ALLOW_HTTP yes` to enroot's
config so pyxis can pull from the local plain-HTTP `registry:2` running on
`localhost:5000`.

The likely location is wherever the existing enroot/pyxis config is wired
(probably `systems/herakles/configuration.nix` or a sibling module under
`systems/herakles/`). Add a single line to the existing enroot config block;
the file is at `/etc/enroot/enroot.conf` and uses **whitespace-separated
key value pairs**, **not** equals signs:

```
# /etc/enroot/enroot.conf
ENROOT_ALLOW_HTTP        yes
```

In a NixOS-style module:

```nix
environment.etc."enroot/enroot.conf".text = ''
  # ...existing settings (ENROOT_RUNTIME_PATH, ENROOT_CACHE_PATH, etc.)
  ENROOT_ALLOW_HTTP        yes
'';
```

If the module already declares `environment.etc."enroot/enroot.conf"`, append
that one line to the existing `text`. Don't replace the whole block — there
are probably tuned settings for `ENROOT_RUNTIME_PATH`, `ENROOT_DATA_PATH`,
`ENROOT_SQUASH_OPTIONS`, etc. that pyxis depends on.

## Why

`registry:2` (Docker's open-source distribution) is running on herakles itself
at `localhost:5000` over plain HTTP. No TLS, no auth — it's a single-host
artefact store on a private network. enroot's default behaviour is to insist
on HTTPS and rejects HTTP registries with:

```
curl: (35) TLS connect error: error:0A00010B:SSL routines::wrong version number
```

`ENROOT_ALLOW_HTTP yes` overrides that. It's documented in
`/nix/store/.../enroot/share/doc/enroot/configuration.md` (or the upstream
`doc/configuration.md` in the NVIDIA/enroot repo) and is the standard knob
for self-hosted plain-HTTP registries.

## Why this is safe in our setup

- The registry is bound to `localhost:5000`, not exposed beyond the host.
- The traffic stays on loopback — no plaintext on the wire that anyone
  outside the box can read.
- The only consumers are pyxis jobs running on the same node.

If the registry ever gets exposed externally (e.g. behind a reverse proxy on
a real domain), revert this setting and serve over HTTPS — at that point the
default HTTPS-only behaviour is the right one.

## What the mo-net side already does

`mo-net`'s `just package` recipe pushes images to
`docker://localhost:5000#mo-net-cuda:<auto-tag>` via skopeo (which is fine
because skopeo has its own `--dest-tls-verify=false` flag and uses it). The
*pull* side (enroot import, triggered by pyxis on the compute nodes) is the
piece that needs the system-level allow.

## Verification

After the rebuild, this should work from any login session on herakles:

```bash
enroot import -o /tmp/test.sqsh docker://localhost:5000#mo-net-cuda:<tag>
```

…without needing `ENROOT_ALLOW_HTTP=yes` set inline. If the import succeeds,
the equivalent pyxis sbatch invocation will too:

```bash
sbatch --container-image=docker://localhost:5000#mo-net-cuda:<tag> --wrap=...
```

## Not in scope for this change

- DNS resolution for `registry.herakles.local`. Not needed since everything
  resolves via `localhost:5000`. If a friendlier name is desired later, add
  a `127.0.0.1 registry.herakles.local` line to `networking.hosts` on
  herakles only.
- Authentication / authorisation on the registry. None today; if ever needed,
  add basic auth at the registry config level and credentials in enroot's
  `.credentials` file rather than removing `ENROOT_ALLOW_HTTP`.
- registry:2 itself. Already running; this just lets enroot reach it.
