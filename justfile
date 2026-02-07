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
