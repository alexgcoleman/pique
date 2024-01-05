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

    # need stable gutter to avoid weird off-by-one error due to horizontal scrollbar
    # TODO: investigate overflow CSS
    DEFAULT_CSS = """
        PiqueTable {
            scrollbar-gutter: stable;
            overflow-y: auto;
            overflow-x: auto;
        }
    """


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
        Binding("b", "bottom", "Bottop", show=True, priority=True),
        Binding("t", "top", "Top", show=True, priority=True),
    ]

    _ROW_HEIGHT_OFFSET = -3
    """Subtract from the container size to get number of rows to display,
    accounts for the table header + horizontal scrollbar + displays"""

    def watch_rows(self, old_rows: int, new_rows: int) -> None:
        self.render_table_rows()

    def watch_start_row(self, old_start_row: int, new_start_row: int) -> None:
        self.render_table_rows()

    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.frame = engine.reader(self.filename)
        self.data_rows = self.frame.select(pl.count()).collect().item()

        super().__init__()

    @property
    def table(self) -> PiqueTable:
        table = self.query_one(PiqueTable)
        return table

    def compose(self) -> ComposeResult:
        yield Msg("PENDING")
        yield PiqueTable(zebra_stripes=True, cell_padding=1)

    def render_table_columns(self) -> None:
        self.table.clear(columns=True)

        cols = self.frame.columns
        self.table.add_columns(*cols)

    def render_table_rows(self) -> None:
        if len(self.table.columns) == 0:
            self.render_table_columns()

        self.table.clear(columns=False)

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

        self.table.add_rows(rows)

    def render_cursor_msg(self) -> None:
        msg = (
            f"Cursor: {self.table.cursor_coordinate}; "
            f"StartRow: {self.start_row}; "
            f"NumRows: {self.page_size}; "
            f"DataRows: {self.data_rows}; "
        )
        self.query_one(Msg).update(msg)

    def calc_rows_for_viewport_height(self) -> int:
        """Determines number of rows that would fit inside the viewport given a
        height."""
        height = self.content_size.height

        return max(height + self._ROW_HEIGHT_OFFSET, 1)

    @property
    def page_size(self) -> int:
        return self.calc_rows_for_viewport_height()

    @property
    def max_start_row(self) -> int:
        """The maximum start_row, leaves room for a blank row at the bottom to
        indicate the end of data"""
        return self.data_rows - self.page_size + 1

    def on_mount(self) -> None:
        self.rows = self.calc_rows_for_viewport_height()
        self.render_table_columns()
        self.render_table_rows()
        self.table.focus()
        self.render_cursor_msg()

    def on_data_table_cell_selected(self, event: DataTable.CellSelected):
        self.render_cursor_msg()

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted):
        self.render_cursor_msg()

    def on_resize(self, event: events.Resize) -> None:
        self.rows = self.calc_rows_for_viewport_height()

    def action_bottom(self) -> None:
        """Move cursor and viewport to the bottom"""
        cursor_col = self.table.cursor_coordinate.column
        self.start_row = self.max_start_row
        self.table.move_cursor(row=self.page_size, column=cursor_col)

    def action_cursor_up(self) -> None:
        """Move cursor up, if we are at the top of the page, move start rown up"""
        if self.table.cursor_coordinate.row == 0:
            col = self.table.cursor_coordinate.column
            self.start_row = max(self.start_row - 1, 0)
            self.table.move_cursor(row=0, column=col)

        self.table.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down, if we are at the bottom of the page, move start rown down"""
        max_row_coord = self.page_size - 1
        if self.table.cursor_coordinate.row >= max_row_coord:
            # scrolls the veiw down if we are already at the bottom
            # don't need to worry about going over the number of rows in the frame,
            # as the table will have 'scrolled' to reveal an empty row at the
            # bottom, preventing the cursor from exceeding max_row_coord, so this
            # block doesn't hit
            col = self.table.cursor_coordinate.column

            self.start_row = self.start_row + 1
            self.table.move_cursor(row=max_row_coord, column=col)

        self.table.action_cursor_down()

    def action_cursor_left(self) -> None:
        self.table.action_cursor_left()

    def action_cursor_right(self) -> None:
        self.table.action_cursor_right()

    def action_page_up(self) -> None:
        """Move the viewport if we are not at the top, otherwise move the cursor
        to the top"""
        cursor_coord = self.table.cursor_coordinate

        if self.start_row > 0:
            # Moving viewport, preserve the relative cursor postition
            self.start_row = max(self.start_row - self.page_size + 1, 0)
            self.table.move_cursor(row=cursor_coord.row, column=cursor_coord.column)
        else:
            self.table.action_page_up()

    def action_page_down(self) -> None:
        """Move the viewport if we are not at the bottom, otherwise move the cursor
        to the bottom"""
        cursor_coord = self.table.cursor_coordinate
        max_row = self.max_start_row

        if self.start_row < max_row:
            # Moving viewport, preserve the relative cursor postition
            self.start_row = min(self.start_row + self.page_size - 1, max_row)
            self.table.move_cursor(row=cursor_coord.row, column=cursor_coord.column)
        else:
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
