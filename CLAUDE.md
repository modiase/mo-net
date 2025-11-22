# Claude Code Instructions

## Environment

Always run commands inside the Nix development shell:

```bash
nix develop -c <command>
```

Examples:
- `nix develop -c python -m pytest mo_net/tests`
- `nix develop -c python script.py`
- `nix develop -c uv lock`

The flake.nix manages all Python dependencies via uv2nix. Do not use pip or create virtualenvs.
