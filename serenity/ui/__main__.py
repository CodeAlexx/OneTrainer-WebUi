"""Entry point: python -m serenity.ui.main (or python -m serenity.ui)."""

from __future__ import annotations

from serenity.ui.app import SerenityApp


def main() -> None:
    app = SerenityApp()
    app.run()


if __name__ == "__main__":
    main()
