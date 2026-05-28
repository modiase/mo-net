#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Marimo notebooks reuse ``_`` as the cell function name across every cell
# which both type checkers reject; skip the notebooks dir regardless of
# whether files were passed explicitly (pre-commit) or auto-discovered.
EXCLUDED_PREFIXES = ("notebooks/", "./notebooks/")


def _excluded(file: str) -> bool:
    return any(file.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


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
        files = [f for f in args.files if not _excluded(f)]
    else:
        files = [
            f
            for f in subprocess.run(
                ["fd", "-e", "py", "--exclude", "notebooks"],
                capture_output=True,
                text=True,
            )
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
