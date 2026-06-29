"""pdfplumber-based PDF table extractor."""

from __future__ import annotations

import pdfplumber

from src import config as cfg
from src.autogen.exceptions import ExtractionError
from src.autogen.extraction.base import BaseExtractor, normalize_rows
from src.autogen.models import ExtractedTable
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

# pdfplumber surfaces password failures as a pdfminer exception; the exact
# class can vary, but the message always contains "password" (case-insensitive).
_PASSWORD_KEYWORDS = ("password", "decrypt", "encrypted", "PDFPasswordIncorrect")


def _is_password_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw.lower() in msg for kw in _PASSWORD_KEYWORDS)


class PdfPlumberExtractor(BaseExtractor):
    """Extract tables from PDFs using pdfplumber."""

    name = "pdfplumber"

    def extract(self, physical_file: str, password: str) -> list[ExtractedTable]:
        """Return a list of ExtractedTable objects found in *physical_file*.

        Raises ExtractionError on password/decryption failure.
        All other library exceptions are caught; tables extracted so far are
        returned (or [] if none succeeded).
        """
        try:
            pdf_ctx = pdfplumber.open(physical_file, password=password)
        except Exception as exc:
            if _is_password_error(exc):
                raise ExtractionError(
                    f"pdfplumber: wrong password for {physical_file}"
                ) from exc
            logger.warning("pdfplumber: failed to open %s: %s", physical_file, exc)
            return []

        tables: list[ExtractedTable] = []
        with pdf_ctx as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    for tbl in page.extract_tables():
                        tables.append(
                            ExtractedTable(
                                name=None,
                                rows=normalize_rows(tbl),
                                page=i + 1,
                            )
                        )
                except Exception as exc:
                    logger.warning(
                        "pdfplumber: error on page %d of %s: %s",
                        i + 1,
                        physical_file,
                        exc,
                    )

        return tables
