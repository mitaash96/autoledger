"""Tests for the four library-specific PDF table extractors.

All library calls are mocked — no real PDFs or network access.
"""

from __future__ import annotations

import logging
import textwrap
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.autogen.exceptions import ExtractionError
from src.autogen.models import ExtractedTable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[list[str]], columns: list[str] | None = None) -> pd.DataFrame:
    """Build a small DataFrame for mocked table exports."""
    cols = columns or [f"col{i}" for i in range(len(rows[0]))] if rows else []
    return pd.DataFrame(rows, columns=cols)


# ===========================================================================
# PyMuPDF Extractor
# ===========================================================================

class TestPyMuPdfExtractor:
    """Tests for src.autogen.extraction.pymupdf_ext.PyMuPdfExtractor."""

    def _make_page_with_tables(self, table_rows: list[list[list[str]]]) -> MagicMock:
        """Return a mock fitz page whose find_tables() yields tables."""
        page = MagicMock()
        tab_list = MagicMock()
        mock_tables = []
        for rows in table_rows:
            t = MagicMock()
            t.extract.return_value = rows
            mock_tables.append(t)
        tab_list.tables = mock_tables
        page.find_tables.return_value = tab_list
        return page

    # --- happy path ---

    def test_happy_path_returns_extracted_tables(self):
        rows_p1 = [["H1", "H2"], ["a", "b"]]
        rows_p2 = [["X", "Y", "Z"]]

        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__iter__ = lambda self: iter([
            self._pages[0], self._pages[1]
        ])

        page1 = self._make_page_with_tables([rows_p1])
        page2 = self._make_page_with_tables([rows_p2])

        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.load_page = MagicMock(side_effect=[page1, page2])

        with patch("src.autogen.extraction.pymupdf_ext.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc

            from src.autogen.extraction.pymupdf_ext import PyMuPdfExtractor
            extractor = PyMuPdfExtractor()
            tables = extractor.extract("file.pdf", "")

        assert len(tables) == 2
        assert tables[0].rows == [["H1", "H2"], ["a", "b"]]
        assert tables[0].page == 1
        assert tables[1].rows == [["X", "Y", "Z"]]
        assert tables[1].page == 2
        mock_doc.close.assert_called_once()

    def test_no_password_needed(self):
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__len__ = MagicMock(return_value=0)

        with patch("src.autogen.extraction.pymupdf_ext.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc

            from src.autogen.extraction.pymupdf_ext import PyMuPdfExtractor
            tables = PyMuPdfExtractor().extract("file.pdf", "")

        assert tables == []
        mock_doc.authenticate.assert_not_called()

    # --- password failure ---

    def test_wrong_password_raises_extraction_error(self):
        mock_doc = MagicMock()
        mock_doc.needs_pass = True
        mock_doc.authenticate.return_value = 0  # falsey → failure

        with patch("src.autogen.extraction.pymupdf_ext.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc

            from src.autogen.extraction.pymupdf_ext import PyMuPdfExtractor
            with pytest.raises(ExtractionError):
                PyMuPdfExtractor().extract("file.pdf", "wrong")

        mock_doc.close.assert_called_once()

    # --- library generic error → [] + warning ---

    def test_library_error_returns_empty_list(self, caplog):
        with patch("src.autogen.extraction.pymupdf_ext.fitz") as mock_fitz:
            mock_fitz.open.side_effect = RuntimeError("corrupt pdf")

            from src.autogen.extraction.pymupdf_ext import PyMuPdfExtractor
            with caplog.at_level(logging.WARNING):
                tables = PyMuPdfExtractor().extract("bad.pdf", "")

        assert tables == []
        assert any("corrupt pdf" in r.message or "corrupt pdf" in str(r.args)
                   for r in caplog.records)

    def test_find_tables_error_returns_partial_results(self, caplog):
        """Error on page 2 should still return page 1 tables."""
        page1 = self._make_page_with_tables([[["a", "b"]]])
        page2 = MagicMock()
        page2.find_tables.side_effect = RuntimeError("page error")

        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.load_page = MagicMock(side_effect=[page1, page2])

        with patch("src.autogen.extraction.pymupdf_ext.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc

            from src.autogen.extraction.pymupdf_ext import PyMuPdfExtractor
            with caplog.at_level(logging.WARNING):
                tables = PyMuPdfExtractor().extract("file.pdf", "")

        assert len(tables) == 1
        assert tables[0].page == 1
        mock_doc.close.assert_called_once()


# ===========================================================================
# PdfPlumber Extractor
# ===========================================================================

class TestPdfPlumberExtractor:
    """Tests for src.autogen.extraction.pdfplumber_ext.PdfPlumberExtractor."""

    def _mock_pdf(self, pages_tables: list[list[list[list[str]]]]) -> MagicMock:
        """Build a mock pdfplumber PDF context manager."""
        mock_pages = []
        for tables in pages_tables:
            page = MagicMock()
            page.extract_tables.return_value = tables
            mock_pages.append(page)

        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        return mock_pdf

    # --- happy path ---

    def test_happy_path_returns_extracted_tables(self):
        # Page 1 has 2 tables; page 2 has 1 table
        pages_tables = [
            [
                [["H1", "H2"], ["v1", "v2"]],
                [["A", "B"]],
            ],
            [
                [["X", None, "Z"]],
            ],
        ]
        mock_pdf = self._mock_pdf(pages_tables)

        with patch("src.autogen.extraction.pdfplumber_ext.pdfplumber") as mock_pp:
            mock_pp.open.return_value = mock_pdf

            from src.autogen.extraction.pdfplumber_ext import PdfPlumberExtractor
            tables = PdfPlumberExtractor().extract("file.pdf", "secret")

        assert len(tables) == 3
        assert tables[0].rows == [["H1", "H2"], ["v1", "v2"]]
        assert tables[0].page == 1
        assert tables[1].rows == [["A", "B"]]
        assert tables[1].page == 1
        assert tables[2].rows == [["X", "", "Z"]]
        assert tables[2].page == 2

    # --- password failure ---

    def test_wrong_password_raises_extraction_error(self):
        with patch("src.autogen.extraction.pdfplumber_ext.pdfplumber") as mock_pp:
            mock_pp.open.side_effect = Exception("PDFPasswordIncorrect")

            from src.autogen.extraction.pdfplumber_ext import PdfPlumberExtractor
            with pytest.raises(ExtractionError):
                PdfPlumberExtractor().extract("file.pdf", "wrong")

    # --- library generic error → [] + warning ---

    def test_library_error_returns_empty_list(self, caplog):
        with patch("src.autogen.extraction.pdfplumber_ext.pdfplumber") as mock_pp:
            mock_pp.open.side_effect = OSError("file not found")

            from src.autogen.extraction.pdfplumber_ext import PdfPlumberExtractor
            with caplog.at_level(logging.WARNING):
                tables = PdfPlumberExtractor().extract("missing.pdf", "")

        assert tables == []
        assert any("file not found" in r.message or "file not found" in str(r.args)
                   for r in caplog.records)

    def test_extract_tables_error_returns_partial(self, caplog):
        """Error on page 2 should still yield page 1 tables."""
        page1 = MagicMock()
        page1.extract_tables.return_value = [[["a", "b"]]]
        page2 = MagicMock()
        page2.extract_tables.side_effect = RuntimeError("corrupt")

        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("src.autogen.extraction.pdfplumber_ext.pdfplumber") as mock_pp:
            mock_pp.open.return_value = mock_pdf

            from src.autogen.extraction.pdfplumber_ext import PdfPlumberExtractor
            with caplog.at_level(logging.WARNING):
                tables = PdfPlumberExtractor().extract("file.pdf", "")

        assert len(tables) == 1
        assert tables[0].page == 1


# ===========================================================================
# Camelot Extractor
# ===========================================================================

class TestCamelotExtractor:
    """Tests for src.autogen.extraction.camelot_ext.CamelotExtractor."""

    def _make_table(self, rows: list[list[str]], page: int | None = 1) -> MagicMock:
        t = MagicMock()
        t.df = pd.DataFrame(rows)
        t.page = page
        return t

    def _make_table_list(self, tables: list[MagicMock]) -> MagicMock:
        tl = MagicMock()
        tl.__len__ = MagicMock(return_value=len(tables))
        tl.__iter__ = MagicMock(return_value=iter(tables))
        return tl

    # --- happy path ---

    def test_happy_path_returns_tables(self):
        row_data = [["H1", "H2"], ["r1", "r2"]]
        t1 = self._make_table(row_data, page=1)
        lattice_list = self._make_table_list([t1])
        stream_list = self._make_table_list([])  # fewer tables

        with patch("src.autogen.extraction.camelot_ext.camelot") as mock_camelot:
            mock_camelot.read_pdf.side_effect = [lattice_list, stream_list]

            from src.autogen.extraction.camelot_ext import CamelotExtractor
            tables = CamelotExtractor().extract("file.pdf", "")

        assert len(tables) == 1
        assert tables[0].rows == [["H1", "H2"], ["r1", "r2"]]
        assert tables[0].page == 1

    def test_stream_wins_when_more_tables(self):
        """stream returns 3 tables, lattice returns 1 → stream chosen."""
        t_lattice = self._make_table([["A", "B"]], page=1)
        lattice_list = self._make_table_list([t_lattice])

        t_s1 = self._make_table([["X"]], page=1)
        t_s2 = self._make_table([["Y"]], page=2)
        t_s3 = self._make_table([["Z"]], page=3)
        stream_list = self._make_table_list([t_s1, t_s2, t_s3])

        with patch("src.autogen.extraction.camelot_ext.camelot") as mock_camelot:
            mock_camelot.read_pdf.side_effect = [lattice_list, stream_list]

            from src.autogen.extraction.camelot_ext import CamelotExtractor
            tables = CamelotExtractor().extract("file.pdf", "")

        assert len(tables) == 3
        pages = {t.page for t in tables}
        assert pages == {1, 2, 3}

    # --- password failure ---

    def test_decryption_error_raises_extraction_error(self):
        with patch("src.autogen.extraction.camelot_ext.camelot") as mock_camelot:
            mock_camelot.read_pdf.side_effect = Exception("cannot decrypt pdf")

            from src.autogen.extraction.camelot_ext import CamelotExtractor
            with pytest.raises(ExtractionError):
                CamelotExtractor().extract("file.pdf", "wrong")

    # --- library generic error → [] + warning ---

    def test_library_error_returns_empty_list(self, caplog):
        with patch("src.autogen.extraction.camelot_ext.camelot") as mock_camelot:
            mock_camelot.read_pdf.side_effect = Exception("some other read error")

            from src.autogen.extraction.camelot_ext import CamelotExtractor
            with caplog.at_level(logging.WARNING):
                tables = CamelotExtractor().extract("bad.pdf", "")

        # Generic errors → [] (not ExtractionError)
        assert tables == []

    def test_lattice_error_stream_fallback(self, caplog):
        """lattice raises → fall back to stream; stream has tables."""
        t_s1 = self._make_table([["A", "B"]], page=1)
        stream_list = self._make_table_list([t_s1])

        with patch("src.autogen.extraction.camelot_ext.camelot") as mock_camelot:
            mock_camelot.read_pdf.side_effect = [
                Exception("lattice fail"),
                stream_list,
            ]

            from src.autogen.extraction.camelot_ext import CamelotExtractor
            with caplog.at_level(logging.WARNING):
                tables = CamelotExtractor().extract("file.pdf", "")

        assert len(tables) == 1


# ===========================================================================
# Docling Extractor
# ===========================================================================

class TestDoclingExtractor:
    """Tests for src.autogen.extraction.docling_ext.DoclingExtractor."""

    def _mock_pikepdf_pdf(self) -> MagicMock:
        """Return a mock pikepdf PDF context manager that succeeds."""
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        return mock_pdf

    def _table_data(self, grid: list[list[str]]):
        """Build a docling TableData from a dense list[list[str]] grid."""
        from docling_core.types.doc.document import TableCell, TableData

        cells = [
            TableCell(
                text=text,
                start_row_offset_idx=r, end_row_offset_idx=r + 1,
                start_col_offset_idx=c, end_col_offset_idx=c + 1,
            )
            for r, row in enumerate(grid)
            for c, text in enumerate(row)
        ]
        return TableData(
            num_rows=len(grid), num_cols=len(grid[0]) if grid else 0,
            table_cells=cells,
        )

    def _mock_converter_result(self, grids: list[list[list[str]]]) -> MagicMock:
        """Build a mock ConversionResult with structured TableItem mocks."""
        mock_tables = []
        for grid in grids:
            t = MagicMock()
            t.data = self._table_data(grid)
            mock_tables.append(t)

        mock_doc = MagicMock()
        mock_doc.tables = mock_tables

        mock_result = MagicMock()
        mock_result.document = mock_doc
        return mock_result

    # --- happy path ---

    def test_happy_path_returns_extracted_tables(self, tmp_path):
        mock_result = self._mock_converter_result(
            [[["H1", "H2"], ["r1c1", "r1c2"], ["r2c1", "r2c2"]]]
        )
        mock_pikepdf_pdf = self._mock_pikepdf_pdf()

        with (
            patch("src.autogen.extraction.docling_ext.pikepdf") as mock_pikepdf,
            patch("src.autogen.extraction.docling_ext._CONVERTER") as mock_converter,
            patch("src.autogen.extraction.docling_ext.tempfile") as mock_tempfile,
        ):
            mock_pikepdf.open.return_value = mock_pikepdf_pdf
            mock_pikepdf.PasswordError = Exception  # placeholder class

            # mock tempfile
            mock_tmp = MagicMock()
            mock_tmp.name = str(tmp_path / "temp_decrypted.pdf")
            mock_tempfile.NamedTemporaryFile.return_value.__enter__ = MagicMock(return_value=mock_tmp)
            mock_tempfile.NamedTemporaryFile.return_value.__exit__ = MagicMock(return_value=False)

            mock_converter.convert.return_value = mock_result

            from src.autogen.extraction.docling_ext import DoclingExtractor
            tables = DoclingExtractor().extract("file.pdf", "")

        assert len(tables) == 1
        assert tables[0].page is None
        assert tables[0].rows == [["H1", "H2"], ["r1c1", "r1c2"], ["r2c1", "r2c2"]]

    # --- password failure ---

    def test_wrong_password_raises_extraction_error(self):
        class FakePasswordError(Exception):
            pass

        with (
            patch("src.autogen.extraction.docling_ext.pikepdf") as mock_pikepdf,
            patch("src.autogen.extraction.docling_ext.DocumentConverter"),
        ):
            mock_pikepdf.PasswordError = FakePasswordError
            mock_pikepdf.open.side_effect = FakePasswordError("bad password")

            from src.autogen.extraction.docling_ext import DoclingExtractor
            with pytest.raises(ExtractionError):
                DoclingExtractor().extract("file.pdf", "wrong")

    # --- library generic error → [] + warning ---

    def test_converter_error_returns_empty_list(self, caplog, tmp_path):
        mock_pikepdf_pdf = self._mock_pikepdf_pdf()

        with (
            patch("src.autogen.extraction.docling_ext.pikepdf") as mock_pikepdf,
            patch("src.autogen.extraction.docling_ext._CONVERTER") as mock_converter,
            patch("src.autogen.extraction.docling_ext.tempfile") as mock_tempfile,
        ):
            mock_pikepdf.open.return_value = mock_pikepdf_pdf
            mock_pikepdf.PasswordError = Exception

            mock_tmp = MagicMock()
            mock_tmp.name = str(tmp_path / "temp_decrypted.pdf")
            mock_tempfile.NamedTemporaryFile.return_value.__enter__ = MagicMock(return_value=mock_tmp)
            mock_tempfile.NamedTemporaryFile.return_value.__exit__ = MagicMock(return_value=False)

            mock_converter.convert.side_effect = RuntimeError("docling crash")

            from src.autogen.extraction.docling_ext import DoclingExtractor
            with caplog.at_level(logging.WARNING):
                tables = DoclingExtractor().extract("file.pdf", "")

        assert tables == []
        assert any("docling crash" in r.message or "docling crash" in str(r.args)
                   for r in caplog.records)
