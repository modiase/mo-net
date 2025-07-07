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
        
        python = pkgs.python313;
        
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
            
          ] ++ systemLibs;
          
          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath systemLibs}:$LD_LIBRARY_PATH"
            
            export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" systemLibs}:$PKG_CONFIG_PATH"
            
            export UV_PYTHON="${pythonEnv}/bin/python"
            
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              uv venv
            fi
            
            source .venv/bin/activate
            
            echo "Syncing dependencies..."
            uv sync
          '';
          
          NIX_CFLAGS_COMPILE = "-I${pkgs.lib.makeSearchPathOutput "dev" "include" systemLibs}";
          NIX_LDFLAGS = "-L${pkgs.lib.makeLibraryPath systemLibs}";
          
          PYTHONPATH = "$PWD";
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
