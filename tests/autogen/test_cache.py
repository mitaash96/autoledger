"""Tests for src/autogen/extraction/cache.py"""

import json
import pytest
from src.autogen.models import ExtractedTable
import src.autogen.extraction.cache as cache_module
from src.autogen.extraction.cache import (
    _cache_path,
    save,
    load,
)


@pytest.fixture(autouse=True)
def patch_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DIR", str(tmp_path))


# ---------------------------------------------------------------------------
# _cache_path
# ---------------------------------------------------------------------------

def test_cache_path(tmp_path):
    path = _cache_path("att1", "camelot")
    assert path.name == "att1_camelot.json"


# ---------------------------------------------------------------------------
# Round-trip save → load
# ---------------------------------------------------------------------------

def _make_tables():
    return [
        ExtractedTable(name="header", rows=[["A", "B"], ["1", "2"]], page=1),
        ExtractedTable(name=None, rows=[], page=None),
    ]


def test_round_trip():
    tables = _make_tables()
    save("att1", "camelot", tables)
    result = load("att1", "camelot")
    assert result is not None
    assert len(result) == 2
    assert result[0].name == "header"
    assert result[0].rows == [["A", "B"], ["1", "2"]]
    assert result[0].page == 1
    assert result[1].name is None
    assert result[1].rows == []
    assert result[1].page is None


# ---------------------------------------------------------------------------
# Cache miss — missing file
# ---------------------------------------------------------------------------

def test_load_missing_returns_none():
    result = load("nonexistent", "camelot")
    assert result is None


# ---------------------------------------------------------------------------
# Corrupt JSON → None, no exception
# ---------------------------------------------------------------------------

def test_load_corrupt_json_returns_none(tmp_path):
    path = _cache_path("att1", "camelot")
    with open(path, "w") as f:
        f.write("{not valid json{{{{")
    result = load("att1", "camelot")
    assert result is None


def test_load_corrupt_json_does_not_raise(tmp_path):
    path = _cache_path("att1", "camelot")
    with open(path, "w") as f:
        f.write("null")  # valid JSON but wrong shape
    # Should not raise; returns None or empty list gracefully
    result = load("att1", "camelot")
    # null deserializes to None which is a list check failure → None
    assert result is None


# ---------------------------------------------------------------------------
# Save creates CACHE_DIR if absent
# ---------------------------------------------------------------------------

def test_save_creates_cache_dir(tmp_path, monkeypatch):
    nested = str(tmp_path / "deep" / "nested")
    monkeypatch.setattr(cache_module, "CACHE_DIR", nested)
    tables = _make_tables()
    save("att1", "camelot", tables)
    result = load("att1", "camelot")
    assert result is not None
