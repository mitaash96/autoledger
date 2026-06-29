"""PyMuPDF-based PDF table extractor."""

from __future__ import annotations

import fitz  # pymupdf

from src import config as cfg
from src.autogen.exceptions import ExtractionError
from src.autogen.extraction.base import BaseExtractor, normalize_rows
from src.autogen.models import ExtractedTable
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])


class PyMuPdfExtractor(BaseExtractor):
    """Extract tables from PDFs using PyMuPDF (fitz)."""

    name = "pymupdf"

    def extract(self, physical_file: str, password: str) -> list[ExtractedTable]:
        """Return a list of ExtractedTable objects found in *physical_file*.

        Raises ExtractionError on password/decryption failure.
        All other library exceptions are caught; tables extracted so far are
        returned (or [] if none succeeded).
        """
        doc = None
        try:
            doc = fitz.open(physical_file)
        except Exception as exc:
            logger.warning("pymupdf: failed to open %s: %s", physical_file, exc)
            return []

        tables: list[ExtractedTable] = []
        try:
            if doc.needs_pass:
                if not doc.authenticate(password):
                    raise ExtractionError(
                        f"pymupdf: wrong password for {physical_file}"
                    )

            for i in range(len(doc)):
                try:
                    page = doc.load_page(i)
                    tab_list = page.find_tables()
                    for t in tab_list.tables:
                        rows = t.extract()
                        tables.append(
                            ExtractedTable(
                                name=None,
                                rows=normalize_rows(rows),
                                page=i + 1,
                            )
                        )
                except ExtractionError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "pymupdf: error on page %d of %s: %s", i + 1, physical_file, exc
                    )
        finally:
            doc.close()

        return tables
