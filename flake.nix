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

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        inherit (nixpkgs) lib;

        mkPkgs =
          cudaSupport:
          import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              inherit cudaSupport;
            };
          };

        pkgs = mkPkgs false;
        pkgsCuda = if pkgs.stdenv.isLinux then mkPkgs true else null;

        # Note: Don't include nvidia_x11 - use system driver to avoid version mismatch
        cudaLibs = lib.optionals (pkgsCuda != null) (
          with pkgsCuda;
          [
            cudatoolkit
            cudaPackages.cudnn
            cudaPackages.nccl
          ]
        );

        systemLibs =
          with pkgs;
          [
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
          ]
          ++ lib.optionals pkgs.stdenv.isLinux [
            glibc
            gcc-unwrapped.lib
            libgcc.lib
          ];

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        buildSystemOverrides =
          final: prev:
          let
            inherit (final) resolveBuildSystem;
            addBuildSystem =
              name: spec:
              if prev ? ${name} then
                {
                  ${name} = prev.${name}.overrideAttrs (old: {
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ resolveBuildSystem spec;
                  });
                }
              else
                { };
          in
          addBuildSystem "contourpy" { meson-python = [ ]; }
          // addBuildSystem "flameprof" { setuptools = [ ]; }
          // addBuildSystem "h5py" {
            setuptools = [ ];
            pkgconfig = [ ];
            cython = [ ];
          }
          // addBuildSystem "matplotlib" {
            meson-python = [ ];
            setuptools = [ ];
          }
          // addBuildSystem "numpy" {
            meson-python = [ ];
            cython = [ ];
          }
          // addBuildSystem "pandas" {
            meson-python = [ ];
            cython = [ ];
          }
          // addBuildSystem "pillow" { setuptools = [ ]; }
          // addBuildSystem "plotille" { setuptools = [ ]; };

        systemDepsOverrides =
          final: prev:
          let
            addSystemDeps =
              name: overrideFn:
              if prev ? ${name} then
                {
                  ${name} = prev.${name}.overrideAttrs overrideFn;
                }
              else
                { };
          in
          { }
          // addSystemDeps "h5py" (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.pkg-config ];
            buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.hdf5 ];
          })
          // addSystemDeps "pillow" (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [
              pkgs.lcms2
              pkgs.libwebp
              pkgs.openjpeg
              pkgs.zlib
              pkgs.libjpeg
              pkgs.libpng
              pkgs.libtiff
              pkgs.freetype
            ];
          })
          // addSystemDeps "contourpy" (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              pkgs.meson
              pkgs.ninja
            ];
          })
          // addSystemDeps "matplotlib" (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              pkgs.meson
              pkgs.ninja
              pkgs.pkg-config
            ];
            buildInputs = (old.buildInputs or [ ]) ++ [
              pkgs.freetype
              pkgs.libpng
              pkgs.qhull
            ];
          })
          // addSystemDeps "pandas" (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              pkgs.meson
              pkgs.ninja
            ];
          })
          // addSystemDeps "numpy" (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              pkgs.meson
              pkgs.ninja
              pkgs.pkg-config
            ];
            buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.openblas ];
          })
          // addSystemDeps "nvidia-nvshmem-cu12" (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [
              pkgs.libfabric
              pkgs.openmpi
              pkgs.pmix
              pkgs.rdma-core
              pkgs.ucx
            ];
          })
          // addSystemDeps "nvidia-cusolver-cu12" (old: {
            buildInputs =
              (old.buildInputs or [ ])
              ++ lib.optionals (pkgsCuda != null) [
                pkgsCuda.cudaPackages.libcublas
                pkgsCuda.cudaPackages.libcusparse
                pkgsCuda.cudaPackages.libnvjitlink
              ];
          })
          // addSystemDeps "nvidia-cusparse-cu12" (old: {
            buildInputs =
              (old.buildInputs or [ ])
              ++ lib.optionals (pkgsCuda != null) [
                pkgsCuda.cudaPackages.libnvjitlink
              ];
          });

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            python = pkgs.python312;
            stdenv = pkgs.stdenv.override {
              targetPlatform = pkgs.stdenv.targetPlatform // {
                # Required for scipy wheel selection on macOS arm64
                darwinSdkVersion = "14.0";
              };
            };
          }).overrideScope
            (
              lib.composeManyExtensions [
                pyproject-build-systems.overlays.wheel
                (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
                buildSystemOverrides
                systemDepsOverrides
              ]
            );

        venv = pythonSet.mkVirtualEnv "mo-net-env" workspace.deps.default;

        # CUDA variant of `venv` — adds the `cuda` extras (jax-cuda12-* plus
        # the nvidia-*-cu12 wheels that bring cuDNN 9.10, cuBLAS 12.9 etc.).
        # Linux-only: the cuda extras are platform-specific wheels.
        cudaVenv = pythonSet.mkVirtualEnv "mo-net-cuda-env" {
          mo-net = [
            "cuda"
            "postgres"
          ];
        };

        # Wheel-bundled NVIDIA libs land under `<venv>/lib/python3.12/site-
        # packages/nvidia/<libname>/lib`. JAX dlopens via LD_LIBRARY_PATH, and
        # the wheel cuDNN 9.10 / cuBLAS 12.9 are what jax-cuda12-plugin 0.6.2
        # actually wants — so put them first. /run/opengl-driver/lib is
        # mounted into the container by the CDI runtime and contains libcuda
        # from the host driver.
        nvidiaWheelLibPath = lib.concatStringsSep ":" (
          map (sub: "${cudaVenv}/lib/python3.12/site-packages/nvidia/${sub}/lib") [
            "cublas"
            "cuda_cupti"
            "cuda_nvrtc"
            "cuda_runtime"
            "cudnn"
            "cufft"
            "cusolver"
            "cusparse"
            "nccl"
            "nvjitlink"
            "nvshmem"
          ]
        );

        # Entrypoint wrapper: when MO_NET_LOKI_URL is set, tees stdout/stderr
        # to per-stream files that fluent-bit tails and ships to Loki. Otherwise
        # exec's python directly so the wrapper is a no-op for ad-hoc runs.
        trainEntrypoint = pkgs.writeShellApplication {
          name = "mo-net-train";
          runtimeInputs = [
            pkgs.coreutils
            pkgs.fluent-bit
            pkgs.gawk
          ];
          text = ''
            set -uo pipefail
            if [ -z "''${MO_NET_LOKI_URL:-}" ]; then
              exec python "$@"
            fi

            log_dir="''${MO_NET_LOG_DIR:-/tmp/mo-net-logs}"
            mkdir -p "$log_dir"
            stdout_log="$log_dir/stdout.log"
            stderr_log="$log_dir/stderr.log"
            : > "$stdout_log"
            : > "$stderr_log"

            # Split URL like http://host:3100 into host + port.
            url=''${MO_NET_LOKI_URL#http://}
            url=''${url#https://}
            loki_host=''${url%%:*}
            loki_port=''${url##*:}
            [ "$loki_port" = "$url" ] && loki_port=3100

            extra_labels=''${MO_NET_LOKI_LABELS:-}
            base_labels="job=mo-net,slurm_job_id=''${SLURM_JOB_ID:-na},slurm_array_job_id=''${SLURM_ARRAY_JOB_ID:-na},slurm_array_task_id=''${SLURM_ARRAY_TASK_ID:-na}"
            [ -n "$extra_labels" ] && base_labels="$base_labels,$extra_labels"

            fb_config=$(mktemp /tmp/fluent-bit-XXXXXX.yaml)
            cat > "$fb_config" <<YAML
            service:
              flush: 2
              log_level: warn
            pipeline:
              inputs:
                - name: tail
                  path: $stdout_log
                  tag: mo-net.stdout
                  read_from_head: true
                  refresh_interval: 1
                - name: tail
                  path: $stderr_log
                  tag: mo-net.stderr
                  read_from_head: true
                  refresh_interval: 1
              outputs:
                - name: loki
                  match: mo-net.*
                  host: $loki_host
                  port: $loki_port
                  labels: $base_labels
                  label_keys: \$tag
                  line_format: json
            YAML

            fluent-bit --config="$fb_config" > /tmp/fluent-bit.log 2>&1 &
            fb_pid=$!

            # Mirror file growth to the container's own stdout/stderr so sbatch
            # --output still captures everything for sacct correlation.
            tail -F -q "$stdout_log" &
            tail_out=$!
            tail -F -q "$stderr_log" >&2 &
            tail_err=$!

            python "$@" >> "$stdout_log" 2>> "$stderr_log"
            exit_code=$?

            # Let fluent-bit run one more flush cycle before we kill it.
            sleep 3
            kill "$fb_pid" "$tail_out" "$tail_err" 2>/dev/null || true
            wait "$fb_pid" 2>/dev/null || true
            exit "$exit_code"
          '';
        };

        mkTrainingImage =
          {
            name,
            pyVenv,
            extraEnv ? [ ],
          }:
          pkgs.dockerTools.streamLayeredImage {
            inherit name;
            tag = self.shortRev or "dirty";
            contents = [
              pyVenv
              pkgs.bashInteractive
              pkgs.coreutils
              pkgs.cacert
              pkgs.fluent-bit
              trainEntrypoint
            ];
            config = {
              Env = [
                "PYTHONUNBUFFERED=1"
                "PATH=${pyVenv}/bin:/bin"
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                # mo_net.settings reads these; mount a host dir at /var/lib/mo-net
                # to persist train.db / run / output artefacts across container runs.
                "MO_NET_DATA_DIR=/var/lib/mo-net/data"
                "MO_NET_RESOURCE_CACHE=/var/lib/mo-net/cache"
              ]
              ++ extraEnv;
              Entrypoint = [ "${trainEntrypoint}/bin/mo-net-train" ];
              WorkingDir = "/workspace";
            };
          };

        editablePythonSet = pythonSet.overrideScope (
          workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; }
        );

        editableVenv = editablePythonSet.mkVirtualEnv "mo-net-dev-env" {
          mo-net = [
            "dev"
            "postgres"
          ];
        };
        editableCudaVenv = editablePythonSet.mkVirtualEnv "mo-net-cuda-env" {
          mo-net = [
            "cuda"
            "dev"
            "postgres"
          ];
        };

        mkShell =
          useCuda:
          let
            activeVenv = if useCuda then editableCudaVenv else editableVenv;
          in
          pkgs.mkShell {
            packages = [
              activeVenv
              pkgs.uv
            ]
            ++ systemLibs
            ++ (if useCuda then cudaLibs else [ ]);

            shellHook = ''
              export NIX_CFLAGS_COMPILE="-I${
                pkgs.lib.makeSearchPathOutput "dev" "include" (systemLibs ++ (if useCuda then cudaLibs else [ ]))
              }"
              export NIX_LDFLAGS="-L${
                pkgs.lib.makeLibraryPath (systemLibs ++ (if useCuda then cudaLibs else [ ]))
              }"
              export PKG_CONFIG_PATH="${
                pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" (
                  systemLibs ++ (if useCuda then cudaLibs else [ ])
                )
              }:$PKG_CONFIG_PATH"
              export REPO_ROOT="$PWD"
              # Prevent uv from managing Python - Nix handles this
              export UV_NO_SYNC=1
              export UV_PYTHON="${activeVenv}/bin/python"
              export UV_PYTHON_DOWNLOADS=never

              ${
                if useCuda && pkgsCuda != null then
                  ''
                    export CUDA_PATH="${pkgsCuda.cudatoolkit}"
                    export CUDA_ROOT="${pkgsCuda.cudatoolkit}"
                    export CUDNN_PATH="${pkgsCuda.cudaPackages.cudnn}"
                    # JAX dlopens libcuda*, libcudnn*, libcublas*, etc. via
                    # LD_LIBRARY_PATH. The wheel-bundled nvidia/* libs (cuDNN 9.10,
                    # cuBLAS 12.9) match jax-cuda12-plugin's required versions; the
                    # nix `cudaPackages.cudnn` (9.8) and `cudatoolkit` (12.8) do not
                    # — putting them first triggers the "cuDNN < 9.10.0" version
                    # check and, if bypassed, a real ABI crash on matmul. So:
                    # wheel libs first, then driver, then system libs.
                    WHEEL_NVIDIA_LIBS=$(ls -d ${activeVenv}/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
                    export LD_LIBRARY_PATH="''${WHEEL_NVIDIA_LIBS}/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath systemLibs}:''${LD_LIBRARY_PATH:-}"
                    export PATH="${pkgsCuda.cudatoolkit}/bin:$PATH"
                    # ptxas etc. live under cudatoolkit; nvrtc is in the wheel.
                    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${pkgsCuda.cudatoolkit}"
                    if [[ "''${DEBUG:-0}" != "0" ]]; then
                      echo "mode: CUDA enabled"
                    fi
                  ''
                else
                  ''
                    if [[ "''${DEBUG:-0}" != "0" ]]; then
                      echo "mode: CPU-only ${
                        if pkgs.stdenv.isLinux then " (use 'nix develop .#cuda' for CUDA)" else ""
                      }"
                    fi
                  ''
              }
              if [[ "''${DEBUG:-0}" != "0" ]]; then
                echo "Python: $(python --version)"
              fi
            '';
          };

      in
      {
        devShells = {
          default = mkShell false;
        }
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          cuda = mkShell true;
        };

        packages = {
          default = venv;
          inherit venv editableVenv;
          mo-net-image = mkTrainingImage {
            name = "mo-net";
            pyVenv = venv;
          };
        }
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          mo-net-cuda-image = mkTrainingImage {
            name = "mo-net-cuda";
            pyVenv = cudaVenv;
            extraEnv = [
              # /usr/lib64 is where libnvidia-container (enroot's nvidia hook,
              # pyxis path) mounts the host driver — libcuda.so.1 lives there.
              # /run/opengl-driver/lib is where NixOS docker+CDI typically
              # mounts the same — kept for the docker path.
              "LD_LIBRARY_PATH=${nvidiaWheelLibPath}:/usr/lib64:/run/opengl-driver/lib"
            ];
          };
        };
      }
    );
}
