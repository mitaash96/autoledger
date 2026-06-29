"""Camelot-based PDF table extractor."""

from __future__ import annotations

import camelot

from src import config as cfg
from src.autogen.exceptions import ExtractionError
from src.autogen.extraction.base import BaseExtractor, normalize_rows
from src.autogen.models import ExtractedTable
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

_DECRYPT_KEYWORDS = ("decrypt", "password", "encrypted", "wrong password")


def _is_decrypt_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw.lower() in msg for kw in _DECRYPT_KEYWORDS)


class CamelotExtractor(BaseExtractor):
    """Extract tables from PDFs using Camelot.

    Runs both lattice and stream flavors, then keeps the result with more
    tables. A password/decryption error from *both* flavors raises
    ExtractionError; other errors log a warning and return [].
    """

    name = "camelot"

    def extract(self, physical_file: str, password: str) -> list[ExtractedTable]:
        """Return a list of ExtractedTable objects found in *physical_file*.

        Raises ExtractionError on password/decryption failure.
        All other library exceptions are caught and logged; [] returned.
        """
        lattice_result = None
        stream_result = None
        errors: list[Exception] = []

        for flavor in ("lattice", "stream"):
            try:
                result = camelot.read_pdf(
                    physical_file,
                    pages="all",
                    flavor=flavor,
                    password=password,
                )
                if flavor == "lattice":
                    lattice_result = result
                else:
                    stream_result = result
            except Exception as exc:
                logger.warning(
                    "camelot %s flavor failed for %s: %s", flavor, physical_file, exc
                )
                errors.append(exc)

        # Both failed → decide whether it's a password issue or generic
        if lattice_result is None and stream_result is None:
            decrypt_err = next((e for e in errors if _is_decrypt_error(e)), None)
            if decrypt_err is not None:
                raise ExtractionError(
                    f"camelot: wrong password for {physical_file}"
                ) from decrypt_err
            return []

        # Pick the result with more tables; prefer stream on tie
        def _len(tl) -> int:
            return len(tl) if tl is not None else -1

        chosen = (
            stream_result
            if _len(stream_result) >= _len(lattice_result)
            else lattice_result
        )
        if chosen is None:
            return []

        tables: list[ExtractedTable] = []
        for table in chosen:
            try:
                df = table.df
                rows = df.values.tolist()
                tables.append(
                    ExtractedTable(
                        name=None,
                        rows=normalize_rows(rows),
                        page=getattr(table, "page", None),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "camelot: error converting table in %s: %s", physical_file, exc
                )

        return tables
