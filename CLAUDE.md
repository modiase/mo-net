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

## Nix + CUDA Gotchas

### Driver Version Mismatch on Non-NixOS Linux

On non-NixOS Linux (e.g., Ubuntu with NVIDIA drivers via apt), do **not** include `linuxPackages.nvidia_x11` in `cudaLibs`. Nix bundles a specific driver version that will conflict with the system driver, causing `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`.

Let applications use the system's NVIDIA driver instead - it's already available via `/run/opengl-driver/lib`.

### uv2nix Dependency Groups

uv2nix doesn't directly expose dependency-groups from `pyproject.toml`. The `workspace.deps.groups.<project>` returns group *names* (e.g., `["cuda", "dev"]`), not actual dependencies.

To include CUDA packages, explicitly merge them into the deps:

```nix
cudaDeps = workspace.deps.default // {
  jax-cuda12-plugin = [];
  jax-cuda12-pjrt = [];
};
```

Use `[]` (no extras) - the `["with-cuda"]` extra pulls in `nvidia-nvshmem-cu12` which requires MPI/infiniband libraries.

### macOS ARM64 Wheel Selection

Set `darwinSdkVersion` to enable wheel selection for scipy/numpy (avoids source builds requiring Fortran):

```nix
stdenv = pkgs.stdenv.override {
  targetPlatform = pkgs.stdenv.targetPlatform // {
    darwinSdkVersion = "14.0";
  };
};
```
