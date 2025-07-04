#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def run_mypy(file: str, *, verbose: bool) -> int:
    result = subprocess.run(["uv", "run", "mypy", file], capture_output=True, text=True)

    if result.returncode != 0 or verbose:
        if result.stdout:
            print(f"=== {file} ===")
            print(result.stdout)
        if result.stderr:
            print(f"=== {file} (stderr) ===")
            print(result.stderr)

    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print stdout and stderr"
    )
    args = parser.parse_args()

    ncpu: int = os.cpu_count() or 1

    if files := [
        f
        for f in subprocess.run(["fd", "-e", "py"], capture_output=True, text=True)
        .stdout.strip()
        .split("\n")
        if f
    ]:
        run_mypy_with_verbose = partial(run_mypy, verbose=args.verbose)
        with ProcessPoolExecutor(max_workers=ncpu) as executor:
            exit_codes: list[int] = list(executor.map(run_mypy_with_verbose, files))

        if any(code != 0 for code in exit_codes):
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
