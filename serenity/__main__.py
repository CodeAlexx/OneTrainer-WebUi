"""Entry point for `python -m serenity`."""

import sys

from serenity.cli.commands import train_command


def main() -> None:
    raise SystemExit(train_command(sys.argv[1:]))


if __name__ == "__main__":
    main()
