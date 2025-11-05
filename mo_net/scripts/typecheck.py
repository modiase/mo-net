#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def run_typechecker(file: str, *, checker: str, verbose: bool) -> int:
    result = subprocess.run(
        ["uv", "run", checker, file], capture_output=True, text=True
    )

    if result.returncode != 0 or verbose:
        if result.stdout:
            print(f"=== {file} ({checker}) ===")
            print(result.stdout)
        if result.stderr:
            print(f"=== {file} ({checker} stderr) ===")
            print(result.stderr)

    return result.returncode


def check_file(file: str, *, verbose: bool) -> int:
    mypy_code = run_typechecker(file, checker="mypy", verbose=verbose)
    pyright_code = run_typechecker(file, checker="pyright", verbose=verbose)
    return max(mypy_code, pyright_code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print stdout and stderr"
    )
    parser.add_argument("files", nargs="*", help="Files to typecheck")
    args = parser.parse_args()

    ncpu: int = os.cpu_count() or 1

    if args.files:
        files = args.files
    else:
        files = [
            f
            for f in subprocess.run(["fd", "-e", "py"], capture_output=True, text=True)
            .stdout.strip()
            .split("\n")
            if f
        ]

    if files:
        check_file_with_verbose = partial(check_file, verbose=args.verbose)
        with ProcessPoolExecutor(max_workers=ncpu) as executor:
            exit_codes: list[int] = list(executor.map(check_file_with_verbose, files))

        if any(code != 0 for code in exit_codes):
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
