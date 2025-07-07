{
  description = "Mo-Net: A neural network framework";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        
        python = pkgs.python313;
        
        # CUDA libraries - use the correct package paths
        cudaLibs = with pkgs; [
          cudatoolkit
          cudaPackages.cudnn
          cudaPackages.nccl
        ];
        
        # Non-CUDA system libraries
        systemLibs = with pkgs; [
          # Core libraries
          glibc
          gcc-unwrapped.lib
          libgcc.lib
          
          # Scientific computing libraries
          openblas
          lapack
          gfortran.cc.lib
          
          # HDF5 support
          hdf5
          szip
          libaec
          
          # Image processing (PIL/Pillow)
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
          
          # X11 libraries (for some image operations)
          xorg.libX11
          xorg.libXau
          xorg.libxcb
          
          # ZeroMQ support
          zeromq
          libsodium
          
          # Other common libraries
          zlib
          openssl
          
          # Tools
          pkg-config
        ];
        
        pythonEnv = python.withPackages (ps: with ps; [
          pip
          setuptools
          wheel
        ]);
        
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.uv
            
          ] ++ systemLibs ++ cudaLibs;
          
          shellHook = ''
            # Set up library paths including CUDA libraries for JAX
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (systemLibs ++ cudaLibs)}:''${LD_LIBRARY_PATH:-}"
            
            # CUDA environment variables for JAX
            export CUDA_PATH="${pkgs.cudatoolkit}"
            export CUDA_ROOT="${pkgs.cudatoolkit}"
            export CUDNN_PATH="${pkgs.cudaPackages.cudnn}"
            
            # Add CUDA binaries to PATH (needed for nvcc, etc.)
            export PATH="${pkgs.cudatoolkit}/bin:$PATH"
            
            # XLA flags for GPU data directory
            export XLA_FLAGS="--xla_gpu_cuda_data_dir=${pkgs.cudatoolkit}/lib"
            
            # Set PKG_CONFIG_PATH for building packages
            export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" (systemLibs ++ cudaLibs)}:$PKG_CONFIG_PATH"
            
            # Set compile flags
            export NIX_CFLAGS_COMPILE="-I${pkgs.lib.makeSearchPathOutput "dev" "include" (systemLibs ++ cudaLibs)}"
            export NIX_LDFLAGS="-L${pkgs.lib.makeLibraryPath (systemLibs ++ cudaLibs)}"
            
            export UV_PYTHON="${pythonEnv}/bin/python"
            
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              uv venv
            fi
            
            source .venv/bin/activate
            
            echo "Syncing dependencies..."
            uv sync
            
            echo ""
            echo "CUDA Environment:"
            echo "  CUDA_PATH: $CUDA_PATH"
            echo "  CUDNN_PATH: $CUDNN_PATH"
            echo "  XLA_FLAGS: $XLA_FLAGS"
            echo "  LD_LIBRARY_PATH includes CUDA libs for JAX"
          '';
          
          PYTHONPATH = "$PWD";
        };
        
        packages.default = pkgs.python313Packages.buildPythonApplication {
          pname = "mo-net";
          version = "0.1.0";
          src = ./.;
          
          buildInputs = systemLibs ++ cudaLibs;
          
          propagatedBuildInputs = with pkgs.python313Packages; [
            click
            fastapi
            h5py
            loguru
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
