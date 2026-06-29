"""Tests for src.autogen.codegen.prompt."""

import json

import pytest

from src.autogen.codegen.prompt import build_prompt, trim_tables
from src.autogen.models import ExtractedTable


def _make_table(n_cols: int, n_rows: int = 3, page: int = 1) -> ExtractedTable:
    header = [f"col{i}" for i in range(n_cols)]
    rows = [header] + [[f"v{r}c{c}" for c in range(n_cols)] for r in range(n_rows)]
    return ExtractedTable(name=f"table_{n_cols}cols", rows=rows, page=page)


BANK = "TestBank"
INSTRUMENT = "credit_card"
WINNER_EXTRACTOR = "pdfplumber_extractor"
WINNER_LIBRARY = "pdfplumber"
TARGET_SCHEMA = {"date": "Utf8", "amount": "Float64", "description": "Utf8"}
TABLES = [_make_table(3)]


def test_build_prompt_contains_bank():
    prompt = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    assert BANK in prompt


def test_build_prompt_contains_instrument():
    prompt = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    assert INSTRUMENT in prompt


def test_build_prompt_contains_target_schema():
    prompt = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    # Each key and value must appear (via json.dumps embedding)
    for key in TARGET_SCHEMA:
        assert key in prompt
    assert json.dumps(TARGET_SCHEMA) in prompt


def test_build_prompt_contains_transform():
    prompt = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    assert "transform" in prompt


def test_build_prompt_contains_polars():
    prompt = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    assert "pl.DataFrame" in prompt or "polars" in prompt


def test_build_prompt_contains_winner_library():
    prompt = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    assert WINNER_LIBRARY in prompt


def test_build_prompt_contains_no_hardcoded_data_rule():
    prompt = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    assert "hardcoded" in prompt.lower() or "No hardcoded" in prompt


def test_build_prompt_contains_only_python():
    prompt = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    assert "ONLY Python" in prompt or "only Python" in prompt


def test_build_prompt_no_markdown_fence():
    prompt = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    assert "```" not in prompt


def test_trim_tables_caps_at_15():
    tables = [_make_table(3) for _ in range(20)]
    result = trim_tables(tables)
    assert len(result) == 15


def test_trim_tables_prefers_wider_tables():
    # Mix of narrow (2-col) and wide (10-col) tables
    narrow = [_make_table(2) for _ in range(12)]
    wide = [_make_table(10) for _ in range(10)]
    result = trim_tables(narrow + wide, max_tables=15)
    assert len(result) == 15
    # All 10 wide tables must survive
    wide_names = {t.name for t in wide}
    result_names = [t.name for t in result]
    wide_in_result = sum(1 for n in result_names if n in wide_names)
    assert wide_in_result == 10


def test_trim_tables_caps_rows():
    table = _make_table(3, n_rows=30)
    result = trim_tables([table], max_rows=10)
    assert len(result) == 1
    # header + max_rows data rows
    assert len(result[0].rows) == 11


def test_trim_tables_default_args():
    tables = [_make_table(3) for _ in range(5)]
    result = trim_tables(tables)
    assert len(result) == 5


def test_build_prompt_feedback_none_unchanged():
    """With feedback=None, output equals the same call without the argument (backward-compatible)."""
    prompt_without = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    prompt_with_none = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA, feedback=None)
    assert prompt_without == prompt_with_none


def test_build_prompt_feedback_appends_block():
    """With a feedback string, output contains the feedback text and the heading AFTER the requirements."""
    feedback_text = "Fix the schema mapping issue in the transform function."
    prompt = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA, feedback=feedback_text)
    assert "# Previous attempt failed — fix these issues and return the corrected full script:" in prompt
    assert feedback_text in prompt
    # Verify feedback appears after requirements block (which ends with requirement 8)
    assert prompt.index("# Previous attempt failed") > prompt.index("8. No hardcoded")


def test_build_prompt_feedback_empty_unchanged():
    """With feedback='' or whitespace-only, no feedback block is appended (same as None)."""
    prompt_without = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA)
    prompt_with_empty = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA, feedback="")
    prompt_with_whitespace = build_prompt(BANK, INSTRUMENT, WINNER_EXTRACTOR, WINNER_LIBRARY, TABLES, TARGET_SCHEMA, feedback="   ")
    assert prompt_without == prompt_with_empty
    assert prompt_without == prompt_with_whitespace
