import argparse
from pathlib import Path
from typing import Optional, Sequence

from rich import print

from . import engine


def valid_filepath(s: str) -> Path:
    try:
        path = Path(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid filepath {s}")

    if not path.exists():
        raise argparse.ArgumentTypeError(f"file {path} does not exist")

    return path


def cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("filename", type=valid_filepath, help="path to file")

    args = parser.parse_args(argv)

    print(engine.reader(args.filename))

    return 0
