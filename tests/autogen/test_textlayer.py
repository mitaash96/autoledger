"""Tests for TextLayerReference. fitz is mocked — no real PDF is opened."""

from unittest.mock import MagicMock, patch

import pytest

from src.autogen.exceptions import ExtractionError
from src.autogen.extraction.textlayer import TextLayerReference, is_value_token, tokenize


def _make_fake_page(text: str = "") -> MagicMock:
    page = MagicMock()
    page.get_text.return_value = text
    return page


def _make_fake_doc(pages, needs_pass=False, auth_ok=True) -> MagicMock:
    doc = MagicMock()
    doc.needs_pass = needs_pass
    doc.authenticate.return_value = 1 if auth_ok else 0
    doc.__iter__ = lambda self: iter(pages)
    doc.__len__ = lambda self: len(pages)
    return doc


def _run(doc, password="pw"):
    with patch("src.autogen.extraction.textlayer.fitz") as mock_fitz:
        mock_fitz.open.return_value = doc
        return TextLayerReference().extract("/fake/path.pdf", password)


class TestTokenize:
    def test_amount_kept_as_single_token(self):
        assert "67,719.47" in tokenize("Total 67,719.47 INR")

    def test_date_kept_as_single_token(self):
        assert "30/04/2026" in tokenize("Date: 30/04/2026")

    def test_account_number_kept(self):
        assert "47004974" in tokenize("Acct 47004974")

    def test_lowercases(self):
        assert "inr" in tokenize("INR")

    def test_duplicates_preserved(self):
        toks = tokenize("INR inr")
        assert toks.count("inr") == 2

    def test_splits_words(self):
        toks = tokenize("hello world")
        assert "hello" in toks and "world" in toks


class TestIsValueToken:
    def test_amount(self):
        assert is_value_token("67,719.47") is True

    def test_date(self):
        assert is_value_token("30/04/2026") is True

    def test_account_number(self):
        assert is_value_token("47004974") is True

    def test_word_without_digit(self):
        assert is_value_token("savings") is False


class TestHappyPath:
    def test_single_page_returns_one_table(self):
        # "hello world 123" → 3 tokens
        doc = _make_fake_doc([_make_fake_page("hello world 123")])
        tables = _run(doc)
        assert len(tables) == 1
        assert tables[0].name == "textlayer"
        assert tables[0].page is None

    def test_row_count_matches_token_count(self):
        text = "hello world 123"
        expected = len(tokenize(text))
        doc = _make_fake_doc([_make_fake_page(text)])
        tables = _run(doc)
        assert len(tables[0].rows) == expected

    def test_each_row_is_single_cell(self):
        doc = _make_fake_doc([_make_fake_page("hello world")])
        tables = _run(doc)
        assert all(len(row) == 1 for row in tables[0].rows)

    def test_row_values_match_tokens(self):
        text = "hello world"
        doc = _make_fake_doc([_make_fake_page(text)])
        tables = _run(doc)
        assert [r[0] for r in tables[0].rows] == tokenize(text)


class TestMultiPage:
    def test_tokens_aggregated_across_pages(self):
        p1 = _make_fake_page("hello world")
        p2 = _make_fake_page("foo bar")
        doc = _make_fake_doc([p1, p2])
        tables = _run(doc)
        expected = tokenize("hello world") + tokenize("foo bar")
        assert [r[0] for r in tables[0].rows] == expected


class TestEmptyText:
    def test_no_meaningful_tokens_returns_empty(self):
        doc = _make_fake_doc([_make_fake_page("   \n\t  ")])
        assert _run(doc) == []

    def test_empty_string_returns_empty(self):
        doc = _make_fake_doc([_make_fake_page("")])
        assert _run(doc) == []


class TestEncryptedPdf:
    def test_wrong_password_raises(self):
        doc = _make_fake_doc([], needs_pass=True, auth_ok=False)
        with pytest.raises(ExtractionError):
            _run(doc)

    def test_wrong_password_closes_doc(self):
        doc = _make_fake_doc([], needs_pass=True, auth_ok=False)
        with pytest.raises(ExtractionError):
            _run(doc)
        doc.close.assert_called_once()

    def test_correct_password_proceeds(self):
        doc = _make_fake_doc([_make_fake_page("hello")], needs_pass=True, auth_ok=True)
        tables = _run(doc)
        doc.authenticate.assert_called_once_with("pw")
        assert len(tables) == 1


class TestDocClose:
    def test_close_called_on_success(self):
        doc = _make_fake_doc([_make_fake_page("hello")])
        _run(doc)
        doc.close.assert_called_once()
