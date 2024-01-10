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
from textual.coordinate import Coordinate
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    ContentSwitcher,
    DataTable,
    Static,
)

from .engine import Engine


@dataclass
class DTypeFormat:
    colour: str = "white"
    justify: str = "center"
    overflow: str | None = None


DTYPE_STYLE: dict[Type, DTypeFormat] = {
    datatypes.String: DTypeFormat(colour="green", justify="left", overflow="ellipsis"),
    datatypes.Float64: DTypeFormat(colour="magenta", justify="right"),
    datatypes.Int64: DTypeFormat(colour="blue", justify="right"),
    datatypes.UInt32: DTypeFormat(colour="blue", justify="right"),
}


class Msg(Static):
    """Msg"""


class PiqueTable(DataTable):
    """DataTable with different bindings + will be putting custom scrolling logic
    here to work with the data lazy loading"""

    # need stable gutter to avoid weird off-by-one error due to horizontal scrollbar
    DEFAULT_CSS = """
        PiqueTable {
            scrollbar-gutter: stable;
        }
    """

    BINDINGS = [
        Binding("k,up", "cursor_up", "Cursor Up", show=True, priority=True),
        Binding("j,down", "cursor_down", "Cursor Down", show=True, priority=True),
        Binding("h,left", "cursor_left", "Cursor Left", show=True, priority=True),
        Binding("l,right", "cursor_right", "Cursor Right", show=True, priority=True),
        Binding("ctrl+u,pageup", "page_up", "Page Up", show=True, priority=True),
        Binding("ctrl+d,pagedown", "page_down", "Page Down", show=True, priority=True),
        Binding("b", "bottom", "Bottom", show=True, priority=True),
        Binding("t", "top", "Top", show=True, priority=True),
    ]


class ColumnSelector(PiqueTable):
    engine: Engine
    hidden_columns: list[str]
    column_stats: pl.DataFrame

    _ROW_HEIGHT_OFFSET = -2

    class ColumnVisibilityChanged(Message):
        name: str
        is_hidden: bool

        def __init__(self, name: str, is_hidden: bool) -> None:
            self.name = name
            self.is_hidden = is_hidden
            super().__init__()

    def __init__(self, *args, engine: Engine, **kwargs) -> None:
        self.engine = engine
        self.column_stats = self.engine.column_stats()

        super().__init__(*args, **kwargs)

        self.hidden_columns = []

    def render_table(self) -> None:
        self.clear(columns=True)

        stats = self.column_stats

        for col_name in ["column", "is_visible"] + stats.columns[1:]:
            self.add_column(label=col_name)

        def formatted_row(row_idx: int) -> list[Text]:
            row = stats[row_idx]
            col_name = row["column_name"].item()

            dtype_format = DTYPE_STYLE.get(self.engine.schema[col_name], DTypeFormat())

            is_hidden = col_name in self.hidden_columns

            bold_style = "bold" if not is_hidden else ""
            colour_style = "white" if not is_hidden else "grey46"
            name_style = bold_style + colour_style

            name_cell = Text(col_name, style=name_style, justify="left")

            is_visible_cell = Text(
                "visible" if col_name not in self.hidden_columns else "hidden",
                style=colour_style,
            )

            dtype_cell = Text(
                row["dtype"].item(), style=dtype_format.colour, justify="center"
            )

            count_cells: list[Text] = [
                Text(str(row[val_type].item()), justify="right")
                for val_type in ["n_unique", "null_count"]
            ]

            min_max_cells: list[Text] = [
                Text(
                    row[val_type].item(),
                    style=dtype_format.colour,
                    justify=dtype_format.justify,
                    overflow="ellipsis",
                )
                for val_type in ["min_value", "max_value"]
            ]

            return (
                [name_cell]
                + [is_visible_cell]
                + [dtype_cell]
                + count_cells
                + min_max_cells
            )

        for row_idx in range(len(stats)):
            row = formatted_row(row_idx)
            self.add_row(*row, height=None)

        self.fixed_columns = 2
        self.cursor_type = "row"

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        col = str(self.get_cell_at(Coordinate(row=event.cursor_row, column=0)))

        self.toggle_hidden(col)
        self.move_cursor(row=event.cursor_row)

    def toggle_hidden(self, col_name: str) -> None:
        is_hidden = col_name in self.hidden_columns

        if is_hidden:
            self.hidden_columns.remove(col_name)
        else:
            self.hidden_columns.append(col_name)

        self.post_message(
            self.ColumnVisibilityChanged(name=col_name, is_hidden=is_hidden)
        )

        self.render_table()

    def on_mount(self) -> None:
        self.render_table()


class DataViewport(containers.Container):
    engine: Engine
    rows = reactive(1)
    start_row = reactive(0)
    hidden_columns: list[str] = reactive([])

    BINDINGS = [
        Binding("k,up", "cursor_up", "Cursor Up", show=True, priority=True),
        Binding("j,down", "cursor_down", "Cursor Down", show=True, priority=True),
        Binding("h,left", "cursor_left", "Cursor Left", show=True, priority=True),
        Binding("l,right", "cursor_right", "Cursor Right", show=True, priority=True),
        Binding("ctrl+u,pageup", "page_up", "Page Up", show=True, priority=True),
        Binding("ctrl+d,pagedown", "page_down", "Page Down", show=True, priority=True),
        Binding("b", "bottom", "Bottom", show=True, priority=True),
        Binding("t", "top", "Top", show=True, priority=True),
    ]

    _ROW_HEIGHT_OFFSET = -3
    """Subtract from the container size to get number of rows to display,
    accounts for the table header + horizontal scrollbar + displays"""

    def __init__(self, engine: Engine, **kwargs) -> None:
        self.engine = engine

        super().__init__(**kwargs)

    def watch_rows(self, old_rows: int, new_rows: int) -> None:
        self.render_table_rows()

    def watch_start_row(self, old_start_row: int, new_start_row: int) -> None:
        self.log(f"{old_start_row=}, {new_start_row=}")
        self.render_table_rows()

    @property
    def table(self) -> PiqueTable:
        table = self.query_one(PiqueTable)
        return table

    def compose(self) -> ComposeResult:
        yield Msg("PENDING")
        yield PiqueTable(zebra_stripes=True, cell_padding=1)

    @property
    def visible_cols(self) -> list:
        return [col for col in self.engine.columns if col not in self.hidden_columns]

    def render_table_columns(self) -> None:
        self.table.clear(columns=True)
        self.table.add_columns(*self.visible_cols)

    def render_table_rows(self) -> None:
        if len(self.table.columns) == 0:
            self.render_table_columns()

        self.table.clear(columns=False)

        cols = self.visible_cols

        df = self.engine.view_slice(self.start_row, self.rows)

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
            f"DataRows: {self.engine.row_count}; "
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
        return self.engine.row_count - self.page_size + 1

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

    def action_top(self) -> None:
        """Move cursor and viewport to the top"""
        cursor_col = self.table.cursor_coordinate.column
        self.start_row = 0
        self.table.move_cursor(row=0, column=cursor_col)

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
            # scrolls the view down if we are already at the bottom
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

    def toggle_hidden(self, col_name: str) -> None:
        is_hidden = col_name in self.hidden_columns

        if is_hidden:
            self.hidden_columns.remove(col_name)
        else:
            self.hidden_columns.append(col_name)

        self.render_table_columns()
        self.render_table_rows()


class Pique(App):
    filename: Path
    engine: Engine
    dark: bool

    BINDINGS = [
        Binding("c", "switch_content", "Switch"),
        Binding("d", "debug_log", "debug log"),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        parsed_args = parse_args()
        self.filename = parsed_args.filename
        self.engine = Engine(filepath=self.filename)

    def action_debug_log(self) -> None:
        self.log.warning(f"{self.focused=}")

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with ContentSwitcher(name="content", initial="data"):
            yield DataViewport(engine=self.engine, id="data")
            yield ColumnSelector(engine=self.engine, id="column-selector")

    def action_switch_content(self) -> None:
        if self.query_one(ContentSwitcher).current == "column-selector":
            self.focus_data()
        else:
            self.focus_selector()

    def focus_selector(self) -> None:
        self.query_one(ContentSwitcher).current = "column-selector"
        self.set_focus(self.query_one(ColumnSelector))

    def focus_data(self) -> None:
        self.query_one(ContentSwitcher).current = "data"
        self.set_focus(self.query_one(DataViewport).query_one(PiqueTable))

    def on_column_selector_column_visibility_changed(
        self, message: ColumnSelector.ColumnVisibilityChanged
    ) -> None:
        self.log.warning(f"{self}.on_column_selector_column_visibility_changed")
        self.log.warning(f"{message=}")

        self.query_one(DataViewport).toggle_hidden(message.name)


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
