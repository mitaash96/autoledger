"""Extraction runner — coordinates all extractor classes in parallel.

Public API
----------
run_all_extractors      cache-aware coordinator (textlayer inline + 4 library processes)
extract_one             single-extractor helper; top-level for pickle safety
"""

from __future__ import annotations

import concurrent.futures
import contextlib
from concurrent.futures import ProcessPoolExecutor

from src import config as cfg
from src.autogen.extraction import cache
from src.autogen.extraction.base import BaseExtractor, promote_header
from src.autogen.extraction.camelot_ext import CamelotExtractor
from src.autogen.extraction.docling_ext import DoclingExtractor
from src.autogen.extraction.pdfplumber_ext import PdfPlumberExtractor
from src.autogen.extraction.pymupdf_ext import PyMuPdfExtractor
from src.autogen.extraction.textlayer import TextLayerReference
from src.autogen.models import ExtractedTable
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REFERENCE_NAME = "textlayer"
LIBRARY_NAMES = ["docling", "camelot", "pymupdf", "pdfplumber"]

EXTRACTOR_CLASSES: dict[str, type[BaseExtractor]] = {
    "textlayer": TextLayerReference,
    "docling": DoclingExtractor,
    "camelot": CamelotExtractor,
    "pymupdf": PyMuPdfExtractor,
    "pdfplumber": PdfPlumberExtractor,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Single-extractor helper (top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------


def extract_one(
    extractor_name: str,
    physical_file: str,
    password: str,
) -> list[ExtractedTable]:
    """Instantiate one extractor and return its tables.

    Kept as a module-level function so it is picklable and can be submitted
    to a ProcessPoolExecutor.  Developers can also call it directly to
    inspect one extractor's output for a single PDF.

    Args:
        extractor_name: Must be a key in EXTRACTOR_CLASSES.
        physical_file:  Absolute path to the PDF on disk.
        password:       Decryption password (empty string if not encrypted).

    Returns:
        list[ExtractedTable] — may be empty if the PDF has no tables.

    Raises:
        ValueError:        If extractor_name is not registered.
        ExtractionError:   If the PDF cannot be decrypted (propagated from the
                           extractor).
    """
    cls = EXTRACTOR_CLASSES.get(extractor_name)
    if cls is None:
        raise ValueError(
            f"Unknown extractor '{extractor_name}'. "
            f"Valid names: {sorted(EXTRACTOR_CLASSES)}"
        )
    tables = cls().extract(physical_file, password)
    return [ExtractedTable(t.name, promote_header(t.rows), t.page) for t in tables]


# ---------------------------------------------------------------------------
# Parallel coordinator
# ---------------------------------------------------------------------------


def run_all_extractors(
    attachment_id: str,
    physical_file: str,
    password: str,
    timeout_seconds: int = 120,
    use_cache: bool = True,
) -> dict[str, list[ExtractedTable]]:
    """Run all 5 extractors (textlayer inline, libraries in processes).

    Cached results are returned without re-running; misses run and cache on
    success; timeouts/exceptions map to [].
    """
    all_names: list[str] = [REFERENCE_NAME] + LIBRARY_NAMES
    result: dict[str, list[ExtractedTable]] = {}
    to_run: list[str] = []

    if use_cache:
        for name in all_names:
            cached = cache.load(attachment_id, name)
            if cached is not None:
                result[name] = cached
            else:
                to_run.append(name)
    else:
        to_run = list(all_names)

    if not to_run:
        logger.info("All extractors cached for %s", attachment_id)
        return result

    ref_to_run = [n for n in to_run if n == REFERENCE_NAME]
    lib_to_run = [n for n in to_run if n != REFERENCE_NAME]

    # Run textlayer inline (pure-CPU, millisecond-fast — no thread/process overhead)
    for name in ref_to_run:
        try:
            tables = extract_one(name, physical_file, password)
            result[name] = tables
            if use_cache:
                cache.save(attachment_id, name, tables)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Extractor '%s' failed for attachment '%s': %s: %s",
                name,
                attachment_id,
                type(exc).__name__,
                exc,
            )
            result[name] = []

    # Only spin up a ProcessPoolExecutor when there are library extractors to run
    proc_cm = ProcessPoolExecutor() if lib_to_run else contextlib.nullcontext()

    with proc_cm as proc_pool:
        futures: dict[str, concurrent.futures.Future] = {}
        for name in lib_to_run:
            futures[name] = proc_pool.submit(extract_one, name, physical_file, password)

        for name, future in futures.items():
            try:
                tables = future.result(timeout=timeout_seconds)
                result[name] = tables
                if use_cache:
                    cache.save(attachment_id, name, tables)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Extractor '%s' timed out after %ds for attachment '%s'",
                    name,
                    timeout_seconds,
                    attachment_id,
                )
                result[name] = []
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Extractor '%s' failed for attachment '%s': %s: %s",
                    name,
                    attachment_id,
                    type(exc).__name__,
                    exc,
                )
                result[name] = []

    return result


