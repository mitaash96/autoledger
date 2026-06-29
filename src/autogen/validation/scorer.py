"""Extractor scorer — ranks library PDF-table extractors by content recall of the
text-layer reference's *value* tokens (numbers, dates, codes).

Only value tokens are scored: prose on the page (addresses, footnotes, marketing
text) is not what a table extractor is expected to capture, so counting it would
penalise correct extractors and make the winner threshold meaningless. Recall over
value tokens directly answers "did the extractor drop any financial data?"."""

from __future__ import annotations

from collections import Counter

from src import config as cfg
from src.autogen.exceptions import ScoringError
from src.autogen.extraction.runner import REFERENCE_NAME
from src.autogen.extraction.textlayer import is_value_token, tokenize
from src.autogen.models import ExtractedTable, ScoringResult
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

WINNER_THRESHOLD: float = 0.70

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reference_tokens(ref_tables: list[ExtractedTable]) -> Counter:
    # Reference cells are already bare tokens; keep only value tokens.
    return Counter(
        cell
        for table in ref_tables
        for row in table.rows
        for cell in row
        if is_value_token(cell)
    )


def _library_tokens(lib_tables: list[ExtractedTable]) -> Counter:
    # Tokenize each cell, then keep only value tokens to match the reference.
    return Counter(
        t
        for table in lib_tables
        for row in table.rows
        for cell in row
        for t in tokenize(cell)
        if is_value_token(t)
    )


def _recall(ref: Counter, lib: Counter) -> float:
    """Recall of lib value tokens against the ref value-token multiset."""
    num = sum(min(lib.get(tok, 0), cnt) for tok, cnt in ref.items())
    den = sum(ref.values())
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_extractors(
    per_pdf_extractions: dict[str, dict[str, list[ExtractedTable]]],
) -> ScoringResult:
    """Rank library extractors by weighted content recall against the text-layer reference.

    Raises ScoringError if no reference tokens exist or all libraries are empty.
    """
    # 1. Build reference Counters; scorable PDFs have non-empty reference.
    ref_counters: dict[str, Counter] = {
        aid: _reference_tokens(extractors.get(REFERENCE_NAME, []))
        for aid, extractors in per_pdf_extractions.items()
    }
    scorable_pdfs = [aid for aid, cnt in ref_counters.items() if cnt]

    if not scorable_pdfs:
        raise ScoringError("no reference tokens across dev PDFs; cannot score")

    # 2. Collect all library names (any key that is not the reference).
    all_lib_names: set[str] = set()
    for extractors in per_pdf_extractions.values():
        all_lib_names.update(k for k in extractors if k != REFERENCE_NAME)

    # 3. Drop libraries that produced no tokens on every scorable PDF.
    def _failed(name: str) -> bool:
        return all(not _library_tokens(per_pdf_extractions[aid].get(name, [])) for aid in scorable_pdfs)

    active_lib_names = sorted(name for name in all_lib_names if not _failed(name))

    if not active_lib_names:
        raise ScoringError("all library extractors failed to produce tables")

    # 4. Score each active library over scorable PDFs.
    scores: dict[str, float] = {}
    per_pdf_scores: dict[str, dict[str, float]] = {}

    for name in active_lib_names:
        recalls: list[float] = []
        for aid in scorable_pdfs:
            recall = _recall(ref_counters[aid], _library_tokens(per_pdf_extractions[aid].get(name, [])))
            recalls.append(recall)
            per_pdf_scores.setdefault(aid, {})[name] = recall
        scores[name] = sum(recalls) / len(recalls)

    # 5. Winner: best mean recall; tie-break by total table count (more is better), then name asc.
    def _total_tables(name: str) -> int:
        return sum(len(per_pdf_extractions[aid].get(name, [])) for aid in scorable_pdfs)

    winner = min(active_lib_names, key=lambda n: (-scores[n], -_total_tables(n), n))
    logger.info("scored extractors %s — winner=%s", scores, winner)

    # 6. Threshold check and result.
    winner_score = scores[winner]
    below_threshold = winner_score < WINNER_THRESHOLD

    if below_threshold:
        logger.warning(
            "Best extractor '%s' scored %.3f which is below the winner threshold %.2f. "
            "Proceeding with caution — manual review recommended.",
            winner,
            winner_score,
            WINNER_THRESHOLD,
        )

    return ScoringResult(
        scores=scores,
        per_pdf_scores=per_pdf_scores,
        winner=winner,
        winner_score=winner_score,
        below_threshold=below_threshold,
    )
