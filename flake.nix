{
  description = "Mo-net: Deep Learning Library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        inherit (nixpkgs) lib;

        mkPkgs = cudaSupport: import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            inherit cudaSupport;
          };
        };

        pkgs = mkPkgs false;
        pkgsCuda = if pkgs.stdenv.isLinux then mkPkgs true else null;

        cudaLibs = lib.optionals (pkgsCuda != null) (with pkgsCuda; [
          cudatoolkit
          cudaPackages.cudnn
          cudaPackages.nccl
          linuxPackages.nvidia_x11
        ]);

        systemLibs = with pkgs; [
          brotli
          freetype
          gfortran.cc.lib
          harfbuzz
          hdf5
          lapack
          lcms2
          libaec
          libjpeg
          libpng
          libsodium
          libtiff
          libwebp
          openblas
          openjpeg
          openssl
          pkg-config
          szip
          xorg.libX11
          xorg.libXau
          xorg.libxcb
          xz
          zeromq
          zlib
        ] ++ lib.optionals pkgs.stdenv.isLinux [
          glibc
          gcc-unwrapped.lib
          libgcc.lib
        ];

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        buildSystemOverrides = final: prev:
          let
            inherit (final) resolveBuildSystem;
            addBuildSystem = name: spec:
              if prev ? ${name} then {
                ${name} = prev.${name}.overrideAttrs (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or []) ++ resolveBuildSystem spec;
                });
              } else {};
          in
          addBuildSystem "contourpy" { meson-python = []; }
          // addBuildSystem "flameprof" { setuptools = []; }
          // addBuildSystem "h5py" { setuptools = []; pkgconfig = []; cython = []; }
          // addBuildSystem "matplotlib" { meson-python = []; setuptools = []; }
          // addBuildSystem "numpy" { meson-python = []; cython = []; }
          // addBuildSystem "pandas" { meson-python = []; cython = []; }
          // addBuildSystem "pillow" { setuptools = []; }
          // addBuildSystem "plotille" { setuptools = []; };

        systemDepsOverrides = final: prev:
          let
            addSystemDeps = name: overrideFn:
              if prev ? ${name} then {
                ${name} = prev.${name}.overrideAttrs overrideFn;
              } else {};
          in
          {}
          // addSystemDeps "h5py" (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ pkgs.pkg-config ];
            buildInputs = (old.buildInputs or []) ++ [ pkgs.hdf5 ];
          })
          // addSystemDeps "pillow" (old: {
            buildInputs = (old.buildInputs or []) ++ [
              pkgs.lcms2 pkgs.libwebp pkgs.openjpeg pkgs.zlib
              pkgs.libjpeg pkgs.libpng pkgs.libtiff pkgs.freetype
            ];
          })
          // addSystemDeps "contourpy" (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ pkgs.meson pkgs.ninja ];
          })
          // addSystemDeps "matplotlib" (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ pkgs.meson pkgs.ninja pkgs.pkg-config ];
            buildInputs = (old.buildInputs or []) ++ [ pkgs.freetype pkgs.libpng pkgs.qhull ];
          })
          // addSystemDeps "pandas" (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ pkgs.meson pkgs.ninja ];
          })
          // addSystemDeps "numpy" (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ pkgs.meson pkgs.ninja pkgs.pkg-config ];
            buildInputs = (old.buildInputs or []) ++ [ pkgs.openblas ];
          });

        pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
          python = pkgs.python312;
          stdenv = pkgs.stdenv.override {
            targetPlatform = pkgs.stdenv.targetPlatform // {
              # Required for scipy wheel selection on macOS arm64
              darwinSdkVersion = "14.0";
            };
          };
        }).overrideScope (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
            buildSystemOverrides
            systemDepsOverrides
          ]
        );

        venv = pythonSet.mkVirtualEnv "mo-net-env" workspace.deps.default;

        editableVenv = (pythonSet.overrideScope (
          workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; }
        )).mkVirtualEnv "mo-net-dev-env" { mo-net = [ "dev" ]; };

        mkShell = useCuda: pkgs.mkShell {
          packages = [ editableVenv pkgs.uv ] ++ systemLibs
            ++ (if useCuda then cudaLibs else []);

          shellHook = ''
            export NIX_CFLAGS_COMPILE="-I${pkgs.lib.makeSearchPathOutput "dev" "include" (systemLibs ++ (if useCuda then cudaLibs else []))}"
            export NIX_LDFLAGS="-L${pkgs.lib.makeLibraryPath (systemLibs ++ (if useCuda then cudaLibs else []))}"
            export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" (systemLibs ++ (if useCuda then cudaLibs else []))}:$PKG_CONFIG_PATH"
            export REPO_ROOT="$PWD"
            # Prevent uv from managing Python - Nix handles this
            export UV_NO_SYNC=1
            export UV_PYTHON="${editableVenv}/bin/python"
            export UV_PYTHON_DOWNLOADS=never

            ${if useCuda && pkgsCuda != null then ''
              export CUDA_PATH="${pkgsCuda.cudatoolkit}"
              export CUDA_ROOT="${pkgsCuda.cudatoolkit}"
              export CUDNN_PATH="${pkgsCuda.cudaPackages.cudnn}"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (systemLibs ++ cudaLibs)}:''${LD_LIBRARY_PATH:-}"
              export PATH="${pkgsCuda.cudatoolkit}/bin:$PATH"
              export XLA_FLAGS="--xla_gpu_cuda_data_dir=${pkgsCuda.cudatoolkit}/lib"
              if [[ "''${DEBUG:-0}" != "0" ]]; then
                echo "mode: CUDA enabled"
                echo "  CUDA_PATH: $CUDA_PATH"
                echo "  CUDNN_PATH: $CUDNN_PATH"
              fi
            '' else ''
              if [[ "''${DEBUG:-0}" != "0" ]]; then
                echo "mode: CPU-only ${if pkgs.stdenv.isLinux then " (use 'nix develop .#cuda' for CUDA)" else ""}"
              fi
            ''}
            if [[ "''${DEBUG:-0}" != "0" ]]; then
              echo "Python: $(python --version)"
            fi
          '';
        };

      in
      {
        devShells = {
          default = mkShell false;
        } // lib.optionalAttrs pkgs.stdenv.isLinux {
          cuda = mkShell true;
        };

        packages = {
          default = venv;
          inherit venv editableVenv;
        };
      }
    );
}
