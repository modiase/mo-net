{
  description = "Mo-Net: A neural network framework";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        mkPkgs = cudaSupport: import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            inherit cudaSupport;
          };
        };

        pkgs = mkPkgs false;
        pkgsCuda = mkPkgs true;

        python = pkgs.python313;

        cudaLibs = with pkgsCuda; [
          cudatoolkit
          cudaPackages.cudnn
          cudaPackages.nccl
        ] ++ pkgs.lib.optional pkgsCuda.stdenv.isLinux pkgsCuda.linuxPackages.nvidia_x11;

        systemLibs = with pkgs; [
          openblas
          lapack
          gfortran.cc.lib
          hdf5
          szip
          libaec
          libjpeg
          libpng
          libtiff
          freetype
          harfbuzz
          lcms2
          openjpeg
          libwebp
          brotli
          xz
          xorg.libX11
          xorg.libXau
          xorg.libxcb
          zeromq
          libsodium
          zlib
          openssl
          pkg-config
        ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
          glibc
          gcc-unwrapped.lib
          libgcc.lib
        ];

        pythonEnv = python.withPackages (ps: with ps; [
          pip
          setuptools
          wheel
        ]);

        mkShell = useCuda: pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.uv
          ] ++ systemLibs ++ (if useCuda then cudaLibs else []);

          shellHook = ''
            ${if useCuda then ''
              export CUDA_PATH="${pkgsCuda.cudatoolkit}"
              export CUDA_ROOT="${pkgsCuda.cudatoolkit}"
              export CUDNN_PATH="${pkgsCuda.cudaPackages.cudnn}"
              export PATH="${pkgsCuda.cudatoolkit}/bin:$PATH"
              export XLA_FLAGS="--xla_gpu_cuda_data_dir=${pkgsCuda.cudatoolkit}/lib"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (systemLibs ++ cudaLibs)}:''${LD_LIBRARY_PATH:-}"
            '' else ""}

            export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" (systemLibs ++ (if useCuda then cudaLibs else []))}:$PKG_CONFIG_PATH"
            export NIX_CFLAGS_COMPILE="-I${pkgs.lib.makeSearchPathOutput "dev" "include" (systemLibs ++ (if useCuda then cudaLibs else []))}"
            export NIX_LDFLAGS="-L${pkgs.lib.makeLibraryPath (systemLibs ++ (if useCuda then cudaLibs else []))}"
            export UV_PYTHON="${pythonEnv}/bin/python"
            export PYTHONPATH="$PWD"

            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              uv venv
            fi

            source .venv/bin/activate
            echo "Syncing dependencies..."
            uv sync
            ${if useCuda then ''
              echo ""
              echo "CUDA enabled"
              echo "  CUDA_PATH: $CUDA_PATH"
              echo "  CUDNN_PATH: $CUDNN_PATH"
            '' else ''
              echo ""
              echo "CPU-only mode"
            ''}
          '';
        };

      in
      {
        devShells = {
          default = mkShell false;
          cuda = mkShell true;
        };
        
        packages.default = pkgs.python313Packages.buildPythonApplication {
          pname = "mo-net";
          version = "0.1.0";
          src = ./.;
          buildInputs = systemLibs;
          propagatedBuildInputs = with pkgs.python313Packages; [
            click
            fastapi
            h5py
            loguru
            jax
            jaxlib
            matplotlib
            msgpack
            mypy
            networkx
            numpy
            pandas
            pydantic
            scipy
            sqlalchemy
            tabulate
            tqdm
            uvicorn
          ];
          doCheck = false;
        };
      }
    );
}
