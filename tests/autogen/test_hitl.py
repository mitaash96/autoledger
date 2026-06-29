"""Tests for HITL reviewer — render_preview_html and review_and_confirm."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.autogen.models import ExtractedTable
from src.autogen.hitl.reviewer import render_preview_html, review_and_confirm

BANNER = "This data will be sent to OpenRouter API. Verify no sensitive values are visible."

_ORIG = ExtractedTable(name="Transactions", rows=[["Date", "Amount"], ["2024-01", "100"]], page=1)
_TABLE = ExtractedTable(name="Transactions", rows=[["Date", "Amount"], ["2024-05", "120"]], page=1)
_ORIG_TABLES = [_ORIG]
_ANON_TABLES = [_TABLE]
# 2-tier: level 1 = extractor, level 2 = attachment (attachment_id, originals, anons).
_GROUPS = [("pdfplumber", [("PDF 1", _ORIG_TABLES, _ANON_TABLES)])]


# ---------------------------------------------------------------------------
# render_preview_html
# ---------------------------------------------------------------------------

def test_render_contains_banner():
    html = render_preview_html(_GROUPS)
    assert BANNER in html


def test_render_contains_cell_values():
    html = render_preview_html(_GROUPS)
    assert "Date" in html
    assert "Amount" in html
    assert "2024-01" in html  # original value
    assert "2024-05" in html  # anonymized value


def test_render_contains_details_and_table_tags():
    html = render_preview_html(_GROUPS)
    assert "<details" in html
    assert "<table" in html


def test_render_side_by_side_columns():
    html = render_preview_html(_GROUPS)
    assert "class='compare'" in html
    assert "class='panel'" in html
    assert "Non-Anonymized" in html
    assert "Anonymized" in html


def test_render_highlights_changed_cells():
    html = render_preview_html(_GROUPS)
    assert "class='changed'" in html


def test_render_no_external_links():
    html = render_preview_html(_GROUPS)
    assert "http://" not in html
    assert "https://" not in html


def test_render_html_escaping():
    evil_orig = ExtractedTable(name="Evil", rows=[["Header"], ["<i>y</i>"]], page=1)
    evil = ExtractedTable(name="Evil", rows=[["Header"], ["<b>x</b>"]], page=1)
    html = render_preview_html([("pdfplumber", [("PDF 1", [evil_orig], [evil])])])
    assert "&lt;b&gt;" in html
    assert "<b>x</b>" not in html


def test_render_unnamed_table():
    unnamed = ExtractedTable(name=None, rows=[["col"]], page=2)
    html = render_preview_html([("pdfplumber", [("PDF 1", [unnamed], [unnamed])])])
    assert "<details" in html  # still renders


# ---------------------------------------------------------------------------
# review_and_confirm — interactive, confirmed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("user_input", ["y", "Y", " y ", " Y "])
def test_confirmed(tmp_path, monkeypatch, user_input):
    preview = str(tmp_path / "preview.html")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    with (
        patch("webbrowser.open") as mock_browser,
        patch("builtins.input", return_value=user_input),
    ):
        result = review_and_confirm(_GROUPS, preview)
    assert result is True
    assert Path(preview).exists()
    mock_browser.assert_called_once()


@pytest.mark.parametrize("user_input", ["n", "", "x", "no", "yes"])
def test_declined(tmp_path, monkeypatch, user_input):
    preview = str(tmp_path / "preview.html")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    with (
        patch("webbrowser.open"),
        patch("builtins.input", return_value=user_input),
    ):
        result = review_and_confirm(_GROUPS, preview)
    assert result is False


def test_file_written_before_browser(tmp_path, monkeypatch):
    preview = str(tmp_path / "sub" / "preview.html")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    calls = []

    def fake_open(uri):
        calls.append(Path(preview).exists())

    with (
        patch("webbrowser.open", side_effect=fake_open),
        patch("builtins.input", return_value="n"),
    ):
        review_and_confirm(_GROUPS, preview)

    # file must exist by the time browser.open is called
    assert calls == [True]


# ---------------------------------------------------------------------------
# review_and_confirm — non-interactive guard
# ---------------------------------------------------------------------------

def test_non_interactive_raises(tmp_path, monkeypatch):
    preview = str(tmp_path / "preview.html")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with (
        patch("webbrowser.open") as mock_browser,
    ):
        with pytest.raises(RuntimeError, match="interactive terminal"):
            review_and_confirm(_GROUPS, preview)
    # guard fires before file write or browser open
    assert not Path(preview).exists()
    mock_browser.assert_not_called()


def test_render_two_tier_tabs_and_numbering():
    from src.autogen.models import ExtractedTable

    t1 = ExtractedTable(None, [["DATE", "BAL"], ["01-01-2020", "5"]], 1)
    t2 = ExtractedTable(None, [["A", "B"], ["x", "9"]], 2)
    groups = [
        ("pdfplumber", [("PDF_A", [t1, t2], [t1, t2]), ("PDF_B", [t1], [t1])]),
        ("docling", [("PDF_A", [t1], [t1])]),
    ]
    html = render_preview_html(groups)
    # 2 extractor tabs + (2 + 1) attachment tabs = 5 radios.
    assert html.count("type='radio'") == 5
    # First extractor open + first attachment of each extractor open = 3 checked.
    assert html.count(" checked") == 3
    assert "pdfplumber" in html and "docling" in html  # level-1 (extractor) tabs
    assert "PDF_A" in html and "PDF_B" in html  # level-2 (attachment) tabs
    assert "Table 1" in html and "Table 2" in html  # numbered within an attachment
