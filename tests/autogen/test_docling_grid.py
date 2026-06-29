"""Self-check for docling structured-table grid reconstruction."""

from docling_core.types.doc.document import TableCell, TableData

from src.autogen.extraction.docling_ext import _grid_from_table_data


def _cell(text, r0, r1, c0, c1):
    return TableCell(
        text=text,
        start_row_offset_idx=r0, end_row_offset_idx=r1,
        start_col_offset_idx=c0, end_col_offset_idx=c1,
    )


def test_span_expansion_and_no_tuples():
    # 2-col header where "Txn" spans both columns, then a data row.
    data = TableData(
        num_rows=2, num_cols=2,
        table_cells=[
            _cell("Txn", 0, 1, 0, 2),  # col-spanning header
            _cell("01/01/2024", 1, 2, 0, 1),
            _cell("100.00", 1, 2, 1, 2),
        ],
    )
    grid = _grid_from_table_data(data)
    assert grid == [["Txn", "Txn"], ["01/01/2024", "100.00"]]
    assert all(isinstance(c, str) for row in grid for c in row)


def test_internal_newlines_collapsed():
    data = TableData(
        num_rows=1, num_cols=1,
        table_cells=[_cell("UPI/foo\n/bar  baz", 0, 1, 0, 1)],
    )
    assert _grid_from_table_data(data) == [["UPI/foo /bar baz"]]


def test_degenerate_table_returns_empty():
    assert _grid_from_table_data(TableData(num_rows=0, num_cols=0, table_cells=[])) == []


if __name__ == "__main__":
    test_span_expansion_and_no_tuples()
    test_degenerate_table_returns_empty()
    print("ok")
