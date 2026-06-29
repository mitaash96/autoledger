"""Text-layer reference extractor — reads the embedded text layer from digital PDFs."""

from __future__ import annotations

import re

import fitz

from src import config as cfg
from src.autogen.exceptions import ExtractionError
from src.autogen.extraction.base import BaseExtractor
from src.autogen.models import ExtractedTable
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])


def tokenize(text: str) -> list[str]:
    """Extract meaningful tokens from raw page text (lowercase, with duplicates)."""
    return re.findall(r"[a-z0-9][a-z0-9,./:\-]*", text.lower())


def is_value_token(tok: str) -> bool:
    """Return True if the token contains any digit."""
    return any(c.isdigit() for c in tok)


class TextLayerReference(BaseExtractor):
    name = "textlayer"

    def extract(self, physical_file: str, password: str) -> list[ExtractedTable]:
        """Extract all text tokens from the PDF text layer. Raises ExtractionError on wrong password."""
        doc = fitz.open(physical_file)
        try:
            if doc.needs_pass and not doc.authenticate(password):
                raise ExtractionError(f"Failed to decrypt '{physical_file}': wrong password.")

            tokens: list[str] = []
            for page in doc:
                tokens.extend(tokenize(page.get_text()))

            if not tokens:
                return []
            return [ExtractedTable(name="textlayer", rows=[[t] for t in tokens], page=None)]
        finally:
            doc.close()
