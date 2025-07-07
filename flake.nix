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
          };
        };
        
        # Python with required packages
        python = pkgs.python313;
        
        # System libraries needed for scientific Python packages
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
        
        # Python environment with uv
        pythonEnv = python.withPackages (ps: with ps; [
          # Core Python packages that are better from nixpkgs
          pip
          setuptools
          wheel
        ]);
        
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Python and uv
            pythonEnv
            pkgs.uv
            
            # System libraries
          ] ++ systemLibs;
          
          shellHook = ''
            # Set up library paths for Python packages
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath systemLibs}:$LD_LIBRARY_PATH"
            
            # Set up PKG_CONFIG_PATH
            export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" systemLibs}:$PKG_CONFIG_PATH"
            
            # Ensure uv uses the system Python
            export UV_PYTHON="${pythonEnv}/bin/python"
            
            # Create .venv if it doesn't exist and sync dependencies
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              uv venv
            fi
            
            # Activate the virtual environment
            source .venv/bin/activate
            
            # Sync dependencies
            echo "Syncing dependencies..."
            uv sync
            
            echo "Mo-Net development environment ready!"
            echo "You can now run: uv run train -i 100 -i 100 -n 1 --dataset-url s3://mo-net-resources/mnist_test.csv"
          '';
          
          # Environment variables for building native extensions
          NIX_CFLAGS_COMPILE = "-I${pkgs.lib.makeSearchPathOutput "dev" "include" systemLibs}";
          NIX_LDFLAGS = "-L${pkgs.lib.makeLibraryPath systemLibs}";
          
          # Additional environment variables
          PYTHONPATH = "$PWD";
        };
        
        # Default package
        packages.default = pkgs.python313Packages.buildPythonApplication {
          pname = "mo-net";
          version = "0.1.0";
          src = ./.;
          
          buildInputs = systemLibs;
          
          propagatedBuildInputs = with pkgs.python313Packages; [
            numpy
            pandas
            matplotlib
            tqdm
            click
            loguru
            mypy
            networkx
            pydantic
            h5py
            scipy
            fastapi
            uvicorn
            sqlalchemy
            tabulate
            msgpack
          ];
          
          doCheck = false; # Skip tests for now
        };
      }
    );
}