"""Tests for src/autogen/validation/scorer.py — TDD.

The scorer counts only *value* tokens (numbers/dates/codes) of the text-layer
reference; plain word tokens are excluded from the denominator.
"""

from __future__ import annotations

import pytest

from src.autogen.exceptions import ScoringError
from src.autogen.models import ExtractedTable, ScoringResult
from src.autogen.validation.scorer import WINNER_THRESHOLD, score_extractors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref(tokens: list[str]) -> list[ExtractedTable]:
    """Build a textlayer reference table from a token list."""
    return [ExtractedTable("textlayer", [[t] for t in tokens], None)]


def _lib(rows: list[list[str]], name: str = "t", page: int = 1) -> list[ExtractedTable]:
    return [ExtractedTable(name, rows, page)]


def _single_pdf(
    ref_tables: list[ExtractedTable],
    lib_tables: dict[str, list[ExtractedTable]],
    attachment_id: str = "pdf1",
) -> dict[str, dict[str, list[ExtractedTable]]]:
    return {attachment_id: {"textlayer": ref_tables, **lib_tables}}


# ---------------------------------------------------------------------------
# 1. Exact-content library → recall 1.0, below_threshold False
# ---------------------------------------------------------------------------


def test_exact_content_scores_1():
    # Library cell tokenizes to all of the reference value tokens.
    result = score_extractors(
        _single_pdf(_ref(["100", "200", "300"]), {"docling": _lib([["100 200 300"]])})
    )

    assert isinstance(result, ScoringResult)
    assert result.winner == "docling"
    assert result.winner_score == pytest.approx(1.0)
    assert result.below_threshold is False
    assert result.scores["docling"] == pytest.approx(1.0)
    assert result.per_pdf_scores["pdf1"]["docling"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Prose (non-value) tokens are excluded from the denominator
# ---------------------------------------------------------------------------


def test_prose_tokens_excluded():
    """Reference has prose + value tokens; a library capturing only the value
    tokens still scores 1.0 because prose is not in the denominator."""
    ref = _ref(["kolkata", "balance", "67,719.47", "30/04/2026"])
    # Library captures only the two value tokens, no prose.
    lib = _lib([["67,719.47 30/04/2026"]])

    result = score_extractors(_single_pdf(ref, {"docling": lib}))

    assert result.scores["docling"] == pytest.approx(1.0)
    assert result.below_threshold is False


# ---------------------------------------------------------------------------
# 3. Partial match ordering — complete beats incomplete
# ---------------------------------------------------------------------------


def test_partial_match_ordering():
    """Library missing half the value tokens scores lower than the complete one."""
    tokens = ["100", "200", "300", "400"]

    good = _lib([["100 200 300 400"]])   # all four value tokens
    bad = _lib([["100 200"]])            # half the value tokens → recall 0.5

    data = _single_pdf(_ref(tokens), {"good": good, "bad": bad})
    result = score_extractors(data)

    assert result.scores["good"] > result.scores["bad"]
    assert result.scores["bad"] == pytest.approx(0.5)
    assert result.winner == "good"


# ---------------------------------------------------------------------------
# 4. Winner is highest recall among multiple libraries
# ---------------------------------------------------------------------------


def test_winner_is_highest_recall():
    ref = _ref(["123", "456"])
    lib_a = _lib([["123 456"]])   # recall 1.0
    lib_b = _lib([["999"]])       # value token, but not in ref → recall 0.0

    data = _single_pdf(ref, {"lib_a": lib_a, "lib_b": lib_b})
    result = score_extractors(data)

    assert result.winner == "lib_a"
    assert result.scores["lib_a"] > result.scores["lib_b"]


# ---------------------------------------------------------------------------
# 5. below_threshold True when best recall < WINNER_THRESHOLD
# ---------------------------------------------------------------------------


def test_below_threshold():
    # Five reference value tokens; library captures only one → recall 0.2.
    ref = _ref(["100", "200", "300", "400", "500"])
    lib = _lib([["100"]])

    data = _single_pdf(ref, {"bad": lib})
    result = score_extractors(data)

    assert result.scores["bad"] == pytest.approx(0.2)
    assert result.below_threshold is True
    assert result.winner_score < WINNER_THRESHOLD


# ---------------------------------------------------------------------------
# 6. No reference value tokens on any PDF → ScoringError
# ---------------------------------------------------------------------------


def test_no_reference_tokens_raises():
    # Empty reference and prose-only reference are both unscorable.
    data = {
        "pdf_empty": {"textlayer": [], "docling": _lib([["100"]])},
        "pdf_prose": {"textlayer": _ref(["hello", "world"]), "docling": _lib([["100"]])},
    }
    with pytest.raises(ScoringError, match="no reference tokens"):
        score_extractors(data)


# ---------------------------------------------------------------------------
# 7. All libraries empty on all PDFs → ScoringError
# ---------------------------------------------------------------------------


def test_all_lib_extractors_empty_raises():
    data = {
        "pdf1": {
            "textlayer": _ref(["123"]),
            "docling": [],
            "camelot": [],
        }
    }
    with pytest.raises(ScoringError, match="all library extractors failed"):
        score_extractors(data)


# ---------------------------------------------------------------------------
# 8. PDF with no reference value tokens is skipped; others still scored
# ---------------------------------------------------------------------------


def test_empty_ref_pdf_skipped():
    data = {
        "pdf_no_ref": {
            "textlayer": [],   # no tokens → not scorable
            "docling": _lib([["100"]]),
        },
        "pdf_good": {
            "textlayer": _ref(["100"]),
            "docling": _lib([["100"]]),
        },
    }
    result = score_extractors(data)

    assert "pdf_no_ref" not in result.per_pdf_scores
    assert "pdf_good" in result.per_pdf_scores
    assert result.scores["docling"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. Tie-break: equal recall → more tables wins; equal tables → alphabetical
# ---------------------------------------------------------------------------


def test_tiebreak_more_tables_wins():
    """Two libraries with identical recall; the one with more ExtractedTable objects wins."""
    ref = _ref(["100", "200"])

    # lib_more: same value tokens split across 2 ExtractedTable objects
    lib_more = [
        ExtractedTable("t", [["100"]], 1),
        ExtractedTable("t", [["200"]], 2),
    ]
    # lib_fewer: same value tokens in 1 ExtractedTable
    lib_fewer = [ExtractedTable("t", [["100 200"]], 1)]

    data = _single_pdf(ref, {"lib_more": lib_more, "lib_fewer": lib_fewer})
    result = score_extractors(data)

    assert result.scores["lib_more"] == pytest.approx(1.0)
    assert result.scores["lib_fewer"] == pytest.approx(1.0)
    assert result.winner == "lib_more"


def test_tiebreak_alphabetical():
    """Equal recall, equal table count → alphabetically earlier name wins."""
    ref = _ref(["100"])
    lib_z = _lib([["100"]])
    lib_a = _lib([["100"]])

    data = _single_pdf(ref, {"z_extractor": lib_z, "a_extractor": lib_a})
    result = score_extractors(data)

    assert result.winner == "a_extractor"
