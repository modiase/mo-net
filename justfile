set shell := ["nix", "develop", "--no-warn-dirty", "--command", "bash", "-c"]

default:
    @just --list

format:
    taplo format pyproject.toml
    files=$(git diff --name-only HEAD -- '*.py'); [[ -z "$files" ]] && echo "0 files to format" || printf '%s\n' "$files" | xargs -P0 ruff format

test:
    pytest -n auto mo_net/tests

typecheck:
    python mo_net/scripts/typecheck.py

test-smoke:
    pytest -n auto -m smoke mo_net/tests

test-collect:
    pytest --collect-only mo_net/tests

# Open a marimo notebook: `just nb corpus_analysis` -> notebooks/corpus_analysis.py
# Binds 0.0.0.0 so you can reach it from another device on the LAN; pass
# `host=127.0.0.1` to restrict to loopback.
nb name host="0.0.0.0" port="2718":
    #!/usr/bin/env bash
    set -euo pipefail
    lan_ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo "<your-ip>")
    echo "LAN URL: http://${lan_ip}:{{port}}"
    nix develop -c marimo edit --watch --host {{host}} --port {{port}} notebooks/{{name}}.py

# Build the OCI training image and push it straight into a registry via skopeo.
# Default registry host comes from $REGISTRY_HOST (set in .envrc.local), with
# localhost:5000 as the fallback. Any missing parts of the ref are filled in:
#   just package                                -> $REGISTRY_HOST/mo-net-cuda:<auto>
#   just package registry.herakles.local        -> registry.herakles.local/mo-net-cuda:<auto>
#   just package registry.herakles.local/mo:v1  -> registry.herakles.local/mo:v1
#   just package --no-cuda                      -> $REGISTRY_HOST/mo-net:<auto>
# <auto> is `<UTC YYYYMMDD-HHMMSS>-<short-sha>[-dirty]`.
[doc("Build + push OCI image to a registry (defaults to $REGISTRY_HOST)")]
package ref="" *flags:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${REGISTRY_HOST:?set REGISTRY_HOST (e.g. in .envrc.local)}"
    : "${BUILD_HOST:?set BUILD_HOST (e.g. in .envrc.local)}"
    target=".#packages.x86_64-linux.mo-net-cuda-image"
    name="mo-net-cuda"
    for f in {{flags}}; do
        if [[ "$f" == "--no-cuda" ]]; then
            target=".#packages.x86_64-linux.mo-net-image"
            name="mo-net"
        fi
    done
    sha=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
    ts=$(date -u +%Y%m%d-%H%M%S)
    dirty=""
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then dirty="-dirty"; fi
    auto_tag="${ts}-${sha}${dirty}"
    ref="{{ref}}"
    [[ -z "$ref" ]] && ref="${REGISTRY_HOST}"
    if [[ "$ref" != */* ]]; then
        ref="${ref}/${name}"
    fi
    last_segment="${ref##*/}"
    if [[ "$last_segment" != *:* ]]; then
        ref="${ref}:${auto_tag}"
    fi
    echo "Building $target ..."
    result_path=$(nix build --no-warn-dirty --no-link --print-out-paths "$target")
    echo "Pushing image to ${ref} ..."
    ssh "$BUILD_HOST" "
        set -euo pipefail
        ${result_path} | nix shell nixpkgs#skopeo --command \
            skopeo copy --insecure-policy --dest-tls-verify=false \
            docker-archive:/dev/stdin 'docker://${ref}'
    "
    echo "Done: pushed ${ref}"

[doc("CI: run tests via uv")]
ci-test:
    #!/usr/bin/env bash
    uv run pytest -n auto mo_net/tests

[doc("CI: run typecheck via uv")]
ci-typecheck *args:
    #!/usr/bin/env bash
    uv run python mo_net/scripts/typecheck.py {{args}}

# Sync to herakles and run tests remotely
[no-cd]
[unix]
test-remote *flags:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Syncing to herakles..."
    rsync -az --delete \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='.direnv' \
        --exclude='.venv' \
        --exclude='result' \
        --exclude='*.egg-info' \
        . herakles:~/mo-net/
    # flit_core's sdist generation respects git's tracked-file set, so any
    # rsync'd file that the remote git doesn't know about gets silently dropped
    # from the Nix-built mo-net package (manifests as ModuleNotFoundError at
    # runtime for newly added files). Stage everything to make flit see it.
    ssh herakles "cd ~/mo-net && git init -q && git add -A"
    flake="path:."
    for flag in {{flags}}; do
        [[ "$flag" == "--cuda" ]] && flake="path:.#cuda"
    done
    ssh -t herakles "cd ~/mo-net && nix develop $flake --no-warn-dirty --command pytest mo_net/tests"
