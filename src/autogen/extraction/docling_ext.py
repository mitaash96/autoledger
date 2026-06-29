"""Docling-based PDF table extractor.

Uses pikepdf to pre-decrypt password-protected PDFs (docling has no password
param), then runs docling's DocumentConverter on the decrypted temp file.
Runs inference on CUDA (GPU), OCR disabled (text-layer PDFs only), tables read
from docling's structured cell model (span-aware) rather than a pandas export.
"""

from __future__ import annotations

import os
import tempfile

import pikepdf
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from src import config as cfg
from src.autogen.exceptions import ExtractionError
from src.autogen.extraction.base import BaseExtractor, normalize_rows
from src.autogen.models import ExtractedTable
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

# Build the converter once at module load; it is thread-safe for reads.
_PDF_OPTS = PdfPipelineOptions()
_PDF_OPTS.do_ocr = False
_PDF_OPTS.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
# Explicit ACCURATE mode guards against docling default drift (cf. OCR engine).
_PDF_OPTS.table_structure_options = TableStructureOptions(
    mode=TableFormerMode.ACCURATE, do_cell_matching=True
)

_CONVERTER = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_PDF_OPTS)}
)


def _grid_from_table_data(data) -> list[list[str]]:
    """Reconstruct a span-expanded num_rows x num_cols grid from a TableData.

    Each cell's text is written into every (row, col) it covers via its offset
    indices, so merged/spanning cells repeat across the spanned positions.
    Returns [] for a degenerate (0-row or 0-col) table.
    """
    n_rows, n_cols = data.num_rows, data.num_cols
    if n_rows <= 0 or n_cols <= 0:
        return []
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for cell in data.table_cells:
        # Collapse wrapped/multi-line cell text to single-spaced, matching the
        # other text-layer extractors so reconciliation lines up.
        text = " ".join(cell.text.split())
        for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
            for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    grid[r][c] = text
    return grid


class DoclingExtractor(BaseExtractor):
    """Extract tables from PDFs using Docling (CUDA/GPU, OCR off).

    Encrypted PDFs are pre-decrypted via pikepdf into a temporary file.
    pikepdf.PasswordError raises ExtractionError; all other errors log and
    return []. Tables are read from docling's structured cell model.
    """

    name = "docling"

    def extract(self, physical_file: str, password: str) -> list[ExtractedTable]:
        """Return a list of ExtractedTable objects found in *physical_file*.

        Raises ExtractionError on password/decryption failure.
        """
        tmp_path: str | None = None
        try:
            # Pre-decrypt (pikepdf round-trip works for both encrypted and
            # plain PDFs; for plain PDFs the password is simply ignored).
            try:
                with pikepdf.open(physical_file, password=password) as pdf:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_f:
                        tmp_path = tmp_f.name
                        pdf.save(tmp_f.name)
            except pikepdf.PasswordError as exc:
                raise ExtractionError(
                    f"docling: wrong password for {physical_file}"
                ) from exc

            # Run docling on the (possibly decrypted) temp file, reusing the
            # module-level converter (docling setup is expensive).
            try:
                result = _CONVERTER.convert(tmp_path)
            except Exception as exc:
                logger.warning(
                    "docling: conversion failed for %s: %s", physical_file, exc
                )
                return []

            tables: list[ExtractedTable] = []
            for table in result.document.tables:
                try:
                    rows = _grid_from_table_data(table.data)
                    if not rows:
                        continue
                    tables.append(
                        ExtractedTable(
                            name=None,
                            rows=normalize_rows(rows),
                            page=None,
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "docling: error exporting table in %s: %s", physical_file, exc
                    )

            return tables

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
