import argparse
from pathlib import Path
from typing import Optional, Sequence

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static

from . import engine


class DataPane(Static):
    """Pane for displaying tabular data"""

    filename: Path

    def __init__(self, filename: Path, *args, **kwargs) -> None:
        self.filename = filename
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        """Create child widgets of a stopwatch."""
        data = engine.reader(self.filename)
        yield Static(f"{data}")


class Pique(App):
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]
    filename: Path
    dark: bool

    def __init__(self, *args, **kwargs) -> None:
        parsed_args = parse_args()
        self.filename = parsed_args.filename
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield DataPane(filename=self.filename)
        yield Footer()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark


def valid_filepath(s: str) -> Path:
    try:
        path = Path(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid filepath {s}")

    if not path.exists():
        raise argparse.ArgumentTypeError(f"file {path} does not exist")

    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("filename", type=valid_filepath, help="path to file")

    return parser.parse_args(argv)


def cli() -> None:
    app = Pique()
    app.run()
