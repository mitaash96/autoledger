"""Tests for src.autogen.codegen.pii_guard."""

import pytest

from src.autogen.codegen.pii_guard import find_violations, scan
from src.autogen.exceptions import PiiLeakError
from src.autogen.models import ExtractedTable


CLEAN_SOURCE = """\
import polars as pl

def transform(attachment, password):
    return pl.DataFrame({"date": [], "amount": []})
"""


def _table(header: list[str], data_rows: list[list[str]]) -> ExtractedTable:
    return ExtractedTable(name="t", rows=[header] + data_rows, page=1)


# --- clean source ---

def test_clean_source_no_violations():
    assert find_violations(CLEAN_SOURCE) == []


def test_clean_source_scan_does_not_raise():
    scan(CLEAN_SOURCE)  # must not raise


# --- password ---

def test_password_literal_in_source_is_violation():
    source = CLEAN_SOURCE + '\nPASSWORD = "s3cr3tP@ss"\n'
    findings = find_violations(source, password="s3cr3tP@ss")
    assert len(findings) >= 1


def test_password_literal_scan_raises():
    source = CLEAN_SOURCE + '\nPASSWORD = "s3cr3tP@ss"\n'
    with pytest.raises(PiiLeakError):
        scan(source, password="s3cr3tP@ss")


def test_empty_password_not_flagged():
    # empty password → skip check
    assert find_violations(CLEAN_SOURCE, password="") == []


# --- email ---

def test_email_in_source_raises():
    source = CLEAN_SOURCE + '\n# customer@example.com\n'
    with pytest.raises(PiiLeakError):
        scan(source)


def test_email_in_source_is_violation():
    source = CLEAN_SOURCE + '\n# user.name+tag@mail.example.org\n'
    findings = find_violations(source)
    assert len(findings) >= 1


# --- digit runs ---

def test_eight_digits_is_violation():
    source = CLEAN_SOURCE + '\n# ref 12345678\n'
    findings = find_violations(source)
    assert len(findings) >= 1


def test_eight_digits_scan_raises():
    source = CLEAN_SOURCE + '\n# ref 12345678\n'
    with pytest.raises(PiiLeakError):
        scan(source)


def test_seven_digits_is_clean():
    source = CLEAN_SOURCE + '\n# ref 1234567\n'
    assert find_violations(source) == []


# --- verbatim data cell ---

def test_verbatim_data_cell_long_raises():
    table = _table(["Date", "Description", "Amount"], [["2024-01-15", "GROCERIES STORE", "150.00"]])
    source = CLEAN_SOURCE + '\n# GROCERIES STORE\n'
    with pytest.raises(PiiLeakError):
        scan(source, input_tables=[table])


def test_verbatim_data_cell_long_is_violation():
    table = _table(["Date", "Description", "Amount"], [["2024-01-15", "GROCERIES STORE", "150.00"]])
    source = CLEAN_SOURCE + '\n# GROCERIES STORE\n'
    findings = find_violations(source, input_tables=[table])
    assert len(findings) >= 1


def test_header_cell_in_source_not_flagged():
    # "Description" is a column name (row 0) — must NOT be flagged
    table = _table(["Date", "Description", "Amount"], [["2024-01-15", "GROCERIES STORE", "150.00"]])
    source = CLEAN_SOURCE + '\ndf = df.rename({"Description": "desc"})\n'
    # Only check header cell "Description" — it should NOT produce a violation by itself
    findings = find_violations(source, input_tables=[table])
    # There may be violations from the data cells if they appear, but "Description" alone must not add one.
    # Source does not contain data-row cells, so should be clean.
    assert len(findings) == 0


def test_short_data_cell_not_flagged():
    # "A" or "12.5" (len < 6) should not trigger
    table = _table(["X"], [["12.50"]])  # "12.50" has len 5
    source = CLEAN_SOURCE + '\n# 12.50\n'
    assert find_violations(source, input_tables=[table]) == []


def test_none_input_tables_clean():
    assert find_violations(CLEAN_SOURCE, input_tables=None) == []
