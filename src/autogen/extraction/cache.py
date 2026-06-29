"""Persistent JSON cache of extraction results.

Cache directory: data/control/extraction_cache/

The cache is never auto-invalidated. Delete the directory to force
re-extraction.
"""

import json
from pathlib import Path

from src import config as cfg
from src.autogen.models import ExtractedTable, table_from_dict, table_to_dict
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

CACHE_DIR = "data/control/extraction_cache"


def _cache_path(attachment_id: str, extractor_name: str) -> Path:
    """Return the path to the cache file for the given parameters."""
    return Path(CACHE_DIR) / f"{attachment_id}_{extractor_name}.json"


def save(
    attachment_id: str,
    extractor_name: str,
    tables: list[ExtractedTable],
) -> None:
    """Persist extraction results to the cache.

    Creates CACHE_DIR if it does not exist.
    """
    path = _cache_path(attachment_id, extractor_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [table_to_dict(t) for t in tables]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger.debug("Cache saved: %s", path)


def load(
    attachment_id: str,
    extractor_name: str,
) -> list[ExtractedTable] | None:
    """Load cached extraction results.

    Returns list[ExtractedTable] on hit, None on miss or corrupt file.
    """
    path = _cache_path(attachment_id, extractor_name)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            logger.warning("Cache file has unexpected shape (not a list): %s", path)
            return None
        return [table_from_dict(d) for d in raw]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Corrupt cache file %s (%s) — treating as miss", path, exc)
        return None
