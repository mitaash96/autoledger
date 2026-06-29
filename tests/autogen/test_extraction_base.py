"""Tests for src/autogen/extraction/base.py"""

import pytest
from src.autogen.extraction.base import BaseExtractor, normalize_rows
from src.autogen.models import ExtractedTable
from src.autogen.exceptions import ExtractionError


# ---------------------------------------------------------------------------
# Concrete stub for testing
# ---------------------------------------------------------------------------

class DummyExtractor(BaseExtractor):
    name = "dummy"

    def extract(self, physical_file: str, password: str) -> list[ExtractedTable]:
        return [ExtractedTable(name="t", rows=[["a", "b"]], page=1)]


class PasswordFailExtractor(BaseExtractor):
    name = "pwfail"

    def extract(self, physical_file: str, password: str) -> list[ExtractedTable]:
        raise ExtractionError("bad password")


# ---------------------------------------------------------------------------
# ABC instantiation guard
# ---------------------------------------------------------------------------

def test_base_extractor_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseExtractor()  # type: ignore[abstract]


def test_concrete_subclass_instantiates():
    e = DummyExtractor()
    assert e.name == "dummy"


def test_concrete_extract_returns_tables():
    e = DummyExtractor()
    tables = e.extract("file.pdf", "")
    assert len(tables) == 1
    assert tables[0].name == "t"


def test_password_fail_raises_extraction_error():
    e = PasswordFailExtractor()
    with pytest.raises(ExtractionError):
        e.extract("file.pdf", "wrong")


# ---------------------------------------------------------------------------
# normalize_rows
# ---------------------------------------------------------------------------

def test_normalize_rows_empty_input():
    assert normalize_rows([]) == []


def test_normalize_rows_ints_and_floats():
    result = normalize_rows([[1, 2.5, 3]])
    assert result == [["1", "2.5", "3"]]


def test_normalize_rows_none_becomes_empty_string():
    result = normalize_rows([[None, "hello", None]])
    assert result == [["", "hello", ""]]


def test_normalize_rows_strips_whitespace():
    result = normalize_rows([[" padded ", "\ttabbed\n"]])
    assert result == [["padded", "tabbed"]]


def test_normalize_rows_mixed_types():
    result = normalize_rows([[0, None, " text ", 3.14]])
    assert result == [["0", "", "text", "3.14"]]


def test_normalize_rows_ragged_rows_preserved():
    result = normalize_rows([[1, 2, 3], [4]])
    assert result == [["1", "2", "3"], ["4"]]


def test_normalize_rows_tuples_accepted():
    result = normalize_rows([(1, 2), (3, 4)])
    assert result == [["1", "2"], ["3", "4"]]


def test_normalize_rows_empty_rows():
    result = normalize_rows([[], []])
    assert result == [[], []]
