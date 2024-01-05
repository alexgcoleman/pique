import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Type

import polars as pl
from polars import datatypes
from rich.text import Text
from textual import containers, events
from textual.app import App, ComposeResult
from textual.binding import Binding
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


class Msg(Static):
    """Msg"""


class PiqueTable(DataTable):
    """DataTable with different bindings + will be putting custom scrolling logic
    here to work with the data lazy loading"""

    DEFAULT_CSS = """
        PiqueTable {
            scrollbar-gutter: stable;
        }
    """  # need stable gutter to avoid weird off-by-one error


class DataViewport(containers.Container):
    filename: Path
    frame: pl.LazyFrame
    rows = reactive(1)
    start_row = reactive(0)

    BINDINGS = [
        Binding("k,up", "cursor_up", "Cursor Up", show=True, priority=True),
        Binding("j,down", "cursor_down", "Cursor Down", show=True, priority=True),
        Binding("h,left", "cursor_left", "Cursor Left", show=True, priority=True),
        Binding("l,right", "cursor_right", "Cursor Right", show=True, priority=True),
        Binding("ctrl+u,pageup", "page_up", "Page Up", show=True, priority=True),
        Binding("ctrl+d,pagedown", "page_down", "Page Down", show=True, priority=True),
    ]

    _ROW_HEIGHT_OFFSET = -2
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
    def table(self) -> PiqueTable:
        table = self.query_one(PiqueTable)
        return table

    def compose(self) -> ComposeResult:
        yield Msg("PENDING")
        yield PiqueTable(zebra_stripes=True, cell_padding=1)

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

        msg = self.query_one(Msg)
        msg.update(self.table.CellSelected.handler_name)

    def render_cursor_msg(self) -> None:
        msg = f"Cursor: {self.table.cursor_coordinate}; StartRow: {self.start_row}; NumRows: {self.calc_rows_for_viewport_height()}"
        self.query_one(Msg).update(msg)

    def calc_rows_for_viewport_height(self, height: int | None = None):
        """Determines number of rows that would fit inside the viewport given a
        height."""
        if height is None:
            height = self.content_size.height

        return max(height + self._ROW_HEIGHT_OFFSET, 1)

    def on_mount(self) -> None:
        self.rows = self.calc_rows_for_viewport_height()
        self.render_table()
        self.table.focus()

    def on_data_table_cell_selected(self, event: DataTable.CellSelected):
        self.render_cursor_msg()

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted):
        self.render_cursor_msg()

    def on_resize(self, event: events.Resize) -> None:
        self.rows = self.calc_rows_for_viewport_height(event.size.height)

    def action_cursor_up(self) -> None:
        """Move cursor up, if we are at the top of the page, move start rown up"""
        if self.table.cursor_coordinate.row == 0:
            col = self.table.cursor_coordinate.column
            self.start_row = max(self.start_row - 1, 0)
            self.table.move_cursor(row=0, column=col)

        self.table.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down, if we are at the bottom of the page, move start rown down"""
        max_row_coord = self.calc_rows_for_viewport_height() - 1
        if self.table.cursor_coordinate.row >= max_row_coord:
            col = self.table.cursor_coordinate.column
            self.start_row = self.start_row + 1
            self.table.move_cursor(row=max_row_coord, column=col)

        self.table.action_cursor_down()

    def action_cursor_left(self) -> None:
        self.table.action_cursor_left()

    def action_cursor_right(self) -> None:
        self.table.action_cursor_right()

    def action_page_up(self) -> None:
        self.table.action_page_up()

    def action_page_down(self) -> None:
        self.table.action_page_down()


class Pique(App):
    filename: Path
    dark: bool

    def __init__(self, *args, **kwargs) -> None:
        parsed_args = parse_args()
        self.filename = parsed_args.filename
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield DataViewport(filename=self.filename)
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
