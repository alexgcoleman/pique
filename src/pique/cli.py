import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Type

import polars as pl
from polars import datatypes
from rich.text import Text
from textual import containers, events
from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Static

from . import engine


@dataclass
class DTypeFormat:
    colour: str = "white"
    justify: str = "center"
    overflow: str | None = None


DTYPE_STYLE: dict[Type, DTypeFormat] = {
    datatypes.String: DTypeFormat(colour="green", justify="left", overflow="ellipsis"),
    datatypes.Float64: DTypeFormat(colour="magenta", justify="right"),
    datatypes.Int64: DTypeFormat(colour="blue", justify="center"),
}


class Lol(Static):
    """lol"""


class Msg(Static):
    """Msg"""


class DataViewport(containers.Container):
    DEFAULT_CSS = """
        DataTable {
            scrollbar-gutter: stable;
        }
    """
    filename: Path
    frame: pl.LazyFrame
    rows = reactive(1)
    start_row = reactive(0)

    _ROW_OFFSET = -2
    """Subtract 2 from the container size to get number of rows to display,
    accounts for the table header + horizontal scrollbar"""

    def watch_rows(self, old_rows: int, new_rows: int) -> None:
        self.render_table()

    def watch_start_row(self, old_start_row: int, new_start_row: int) -> None:
        self.render_table()

    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.frame = engine.reader(self.filename)
        super().__init__()

    @property
    def table(self) -> DataTable:
        table = self.query_one(DataTable)
        return table

    def compose(self) -> ComposeResult:
        yield DataTable(zebra_stripes=True, cell_padding=1)

    def render_table(self) -> None:
        self.table.clear(columns=True)

        cols = self.frame.columns

        df = self.frame.slice(self.start_row, self.rows).collect()

        nulls = {col: df.select(pl.col(col).is_null()) for col in cols}
        styling = {col: DTYPE_STYLE.get(df[col].dtype, DTypeFormat()) for col in cols}

        rows = [
            [
                format_cell(
                    df[col][row], fmt=styling[col], is_na=nulls[col].row(row)[0]
                )
                for col in cols
            ]
            for row in range(len(df))
        ]

        self.table.add_columns(*cols)
        self.table.add_rows(rows)

    def set_rows_from_height(self, height: int) -> None:
        self.rows = max(height + self._ROW_OFFSET, 1)

    def on_mount(self) -> None:
        self.set_rows_from_height(self.content_size.height)
        self.render_table()

    def on_resize(self, event: events.Resize) -> None:
        self.set_rows_from_height(event.size.height)


class DataPane(containers.Vertical):
    """Pane for displaying tabular data"""

    filename: Path
    content_size_msg = reactive("PENDING")

    def __init__(self, filename: Path) -> None:
        self.filename = filename
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Msg("MESSAGE")
        yield DataViewport(filename=self.filename)
        yield Lol("lololol")

    def render_msg(self) -> None:
        self.content_size_msg = f"size = {self.content_size}"
        self.display_content_size()

    def on_mount(self) -> None:
        self.render_msg()

    def display_content_size(self) -> None:
        msg = self.query_one(Msg)
        msg.update(self.content_size_msg)

    def on_resize(self, event: events.Resize) -> None:
        self.content_size_msg = f"size = {event.size.height}"
        self.display_content_size()


class Pique(App):
    filename: Path
    dark: bool

    def __init__(self, *args, **kwargs) -> None:
        parsed_args = parse_args()
        self.filename = parsed_args.filename
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
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


def format_cell(cell, fmt: DTypeFormat, is_na: bool = False) -> Text:
    if isinstance(cell, str):
        cell_repr = repr(cell)
    else:
        cell_repr = str(cell)

    colour = fmt.colour if not is_na else "gray"
    return Text(cell_repr, style=colour, overflow=fmt.overflow, justify=fmt.justify)
