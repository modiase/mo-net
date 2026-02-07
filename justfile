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
    flake="path:."
    for flag in {{flags}}; do
        [[ "$flag" == "--cuda" ]] && flake="path:.#cuda"
    done
    ssh -t herakles "cd ~/mo-net && nix develop $flake --no-warn-dirty --command pytest mo_net/tests"
