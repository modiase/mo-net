#!/usr/bin/env python3

import argparse
import asyncio
import os
import subprocess
import sys

MAX_CONCURRENT = os.cpu_count() or 4


async def run_typechecker(
    file: str, *, checker: str, verbose: bool, sem: asyncio.Semaphore
) -> int:
    async with sem:
        proc = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            checker,
            file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0 or verbose:
            if stdout:
                print(f"=== {file} ({checker}) ===")
                print(stdout.decode())
            if stderr:
                print(f"=== {file} ({checker} stderr) ===")
                print(stderr.decode())

        return proc.returncode or 0


async def check_file(file: str, *, verbose: bool, sem: asyncio.Semaphore) -> int:
    mypy_code, pyright_code = await asyncio.gather(
        run_typechecker(file, checker="mypy", verbose=verbose, sem=sem),
        run_typechecker(file, checker="pyright", verbose=verbose, sem=sem),
    )
    return max(mypy_code, pyright_code)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print stdout and stderr"
    )
    parser.add_argument("files", nargs="*", help="Files to typecheck")
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        result = subprocess.run(["fd", "-e", "py"], capture_output=True, text=True)
        files = [f for f in result.stdout.strip().split("\n") if f]

    if files:
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        exit_codes = await asyncio.gather(
            *(check_file(f, verbose=args.verbose, sem=sem) for f in files)
        )
        if any(code != 0 for code in exit_codes):
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
