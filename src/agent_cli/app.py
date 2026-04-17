"""CLIApp — top-level lifecycle and entrypoint."""
from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Agent-Harness interactive CLI",
    )
    parser.add_argument(
        "--version", action="store_true", help="show version and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from agent_cli import __version__
        print(f"harness {__version__}")
        return 0

    from agent_cli import _check_deps
    _check_deps()
    sys.stderr.write("REPL not yet wired; arrives in Step 2.\n")
    return 1
