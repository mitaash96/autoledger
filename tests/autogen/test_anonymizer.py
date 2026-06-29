"""Tests for src/autogen/anonymization/anonymizer.py — TDD."""

from __future__ import annotations

from datetime import datetime

from src.autogen.anonymization.anonymizer import TableAnonymizer
from src.autogen.models import ExtractedTable


def _table(rows: list[list[str]]) -> ExtractedTable:
    return ExtractedTable(name=None, rows=rows, page=None)


# ---------------------------------------------------------------------------
# _detect_type
# ---------------------------------------------------------------------------


def test_detect_date():
    assert TableAnonymizer()._detect_type("15/01/2024") == "date"


def test_detect_date_dmy_dash():
    assert TableAnonymizer()._detect_type("15-01-2024") == "date"


def test_detect_long_digits_code():
    assert TableAnonymizer()._detect_type("123456789012") == "code"


def test_detect_alphanumeric_code():
    assert TableAnonymizer()._detect_type("TXN1234AB") == "code"


def test_detect_numeric_comma():
    assert TableAnonymizer()._detect_type("1,234.56") == "numeric"


def test_detect_negative_numeric():
    assert TableAnonymizer()._detect_type("-42.00") == "numeric"


def test_detect_plain_integer():
    assert TableAnonymizer()._detect_type("100") == "numeric"


def test_detect_name():
    assert TableAnonymizer()._detect_type("John Smith") == "name"


def test_detect_keyword_single_text():
    assert TableAnonymizer()._detect_type("Balance") == "text"


def test_detect_opening_balance_text():
    assert TableAnonymizer()._detect_type("Opening Balance") == "text"


# ---------------------------------------------------------------------------
# numeric strategy
# ---------------------------------------------------------------------------


def test_numeric_scaled_in_range():
    a = TableAnonymizer(seed=42)
    result = a.anonymize([_table([["Amount"], ["100.00"]])])
    assert 80.0 <= float(result[0].rows[1][0]) <= 120.0


def test_numeric_preserves_two_decimals():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Amount"], ["100.00"]])])[0].rows[1][0]
    assert "." in cell and len(cell.split(".")[1]) == 2


def test_numeric_preserves_sign():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Amount"], ["-42.00"]])])[0].rows[1][0]
    assert cell.startswith("-")


def test_numeric_comma_and_two_decimals():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Amount"], ["1,234.56"]])])[0].rows[1][0]
    assert "," in cell
    assert len(cell.split(".")[1]) == 2


def test_numeric_no_comma_when_original_had_none():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Amount"], ["50.00"]])])[0].rows[1][0]
    assert "." in cell
    assert "," not in cell


# ---------------------------------------------------------------------------
# date strategy
# ---------------------------------------------------------------------------


def test_date_valid_dmy_format():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Date"], ["15/01/2024"]])])[0].rows[1][0]
    datetime.strptime(cell, "%d/%m/%Y")  # must parse back
    assert cell != "15/01/2024"


def test_date_ordering_preserved():
    a = TableAnonymizer(seed=42)
    rows = a.anonymize([_table([["Date"], ["01/01/2024"], ["15/01/2024"]])])[0].rows
    d1 = datetime.strptime(rows[1][0], "%d/%m/%Y")
    d2 = datetime.strptime(rows[2][0], "%d/%m/%Y")
    assert d1 < d2


def test_date_identical_input_identical_output():
    a = TableAnonymizer(seed=42)
    rows = a.anonymize([_table([["Date"], ["15/01/2024"], ["15/01/2024"]])])[0].rows
    assert rows[1][0] == rows[2][0]


# ---------------------------------------------------------------------------
# code strategy
# ---------------------------------------------------------------------------


def test_code_all_digits_preserved_length():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Account"], ["1234567890123"]])])[0].rows[1][0]
    assert len(cell) == 13
    assert cell.isdigit()
    assert cell != "1234567890123"


def test_code_alphanumeric_structure():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Code"], ["AB-12-CD"]])])[0].rows[1][0]
    assert len(cell) == 8
    assert cell[2] == "-" and cell[5] == "-"
    assert cell[0].isupper() and cell[1].isupper()
    assert cell[3].isdigit() and cell[4].isdigit()
    assert cell[6].isupper() and cell[7].isupper()


def test_code_consistent_within_run():
    a = TableAnonymizer(seed=42)
    rows = a.anonymize([_table([["Code"], ["AB-12-CD"], ["AB-12-CD"]])])[0].rows
    assert rows[1][0] == rows[2][0]


# ---------------------------------------------------------------------------
# name strategy
# ---------------------------------------------------------------------------


def test_name_two_title_tokens():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Name"], ["John Smith"]])])[0].rows[1][0]
    tokens = cell.split()
    assert len(tokens) == 2
    assert all(t[0].isupper() and t[1:].islower() for t in tokens)


def test_name_changed():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Name"], ["John Smith"]])])[0].rows[1][0]
    assert cell != "John Smith"


def test_name_consistent_within_run():
    a = TableAnonymizer(seed=42)
    rows = a.anonymize([_table([["Name"], ["John Smith"], ["John Smith"]])])[0].rows
    assert rows[1][0] == rows[2][0]


# ---------------------------------------------------------------------------
# header and text pass-through
# ---------------------------------------------------------------------------


def test_header_row_preserved():
    a = TableAnonymizer(seed=42)
    header = ["Date", "Description", "Amount"]
    result = a.anonymize([_table([header, ["15/01/2024", "John Smith", "1,234.56"]])])
    assert result[0].rows[0] == header


def test_keywords_kept_non_pii():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Desc"], ["Opening Balance"]])])[0].rows[1][0]
    assert cell == "Opening Balance"  # both are non-PII keywords → kept


def test_generic_text_separators_preserved():
    a = TableAnonymizer(seed=42)
    cell = a.anonymize([_table([["Info"], ["N/A"]])])[0].rows[1][0]
    assert len(cell) == 3 and cell[1] == "/"  # not PII; structure preserved


def test_narration_pii_scrubbed():
    """Free-text narration must not leak names, handles, or digit runs verbatim."""
    a = TableAnonymizer(seed=42)
    narration = "UPI-ANASUA BASU-anasuabasu1998@okhdfcbank-119616912998-wee Ref 33218196001338"
    cell = a.anonymize([_table([["Narration"], [narration]])])[0].rows[1][0]
    for leak in ("ANASUA", "anasuabasu1998", "119616912998", "33218196001338"):
        assert leak not in cell


def test_indicators_and_domain_preserved():
    """Balance: txn-type indicators, keyword anchors and PSP domain survive anonymization."""
    a = TableAnonymizer(seed=42)
    narration = "UPI-ANASUA BASU-anasuabasu1998@okhdfcbank-119616912998-wee Ref 33218196001338"
    cell = a.anonymize([_table([["Narration"], [narration]])])[0].rows[1][0]
    for kept in ("UPI", "Ref", "@okhdfcbank", "wee"):
        assert kept in cell


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_determinism_same_seed():
    table = _table([["Name", "Amount", "Date"], ["John Smith", "1,234.56", "15/01/2024"]])
    r1 = TableAnonymizer(seed=42).anonymize([table])
    r2 = TableAnonymizer(seed=42).anonymize([table])
    assert r1[0].rows == r2[0].rows


# ---------------------------------------------------------------------------
# immutability
# ---------------------------------------------------------------------------


def test_returns_new_table_object():
    table = _table([["Date"], ["15/01/2024"]])
    result = TableAnonymizer(seed=42).anonymize([table])
    assert result[0] is not table


def test_original_rows_untouched():
    table = _table([["Date"], ["15/01/2024"]])
    original_rows = [list(r) for r in table.rows]
    TableAnonymizer(seed=42).anonymize([table])
    assert table.rows == original_rows
