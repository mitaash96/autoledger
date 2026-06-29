"""Tests for src.autogen.reporting.html_report (Task 12b)."""

import re
from pathlib import Path

import pytest

from src.autogen.models import Attachment, ExtractedTable, TestResult
from src.autogen.reporting.html_report import render_report, write_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_attachment(idx: int) -> Attachment:
    return Attachment(
        attachment_id=f"att_{idx}",
        raw_attachment_id=f"raw_{idx}",
        name=f"statement_{idx}.pdf",
        email_id=f"email_{idx}",
        date=None,
        physical_file=f"/data/pdf/statement_{idx}.pdf",
    )


def _make_test_result(success: bool) -> TestResult:
    return TestResult(
        attachment_id="att_1",
        name="statement_1.pdf",
        success=success,
        row_count=42,
        columns=["date", "amount", "description"],
        schema_conforms=True,
        null_rates={"date": 0.0, "amount": 0.01, "description": 0.05},
        error=None,
        parquet_path="/data/out.parquet",
    )


@pytest.fixture()
def full_ctx():
    return {
        "bank": "HDFC",
        "instrument": "credit_card",
        "timestamp": "2024-01-15T10:30:00",
        "success": True,
        "winner_extractor": "camelot",
        "winner_score": 0.92,
        "codegen_backend": "local",
        "anonymization_applied": False,
        "hitl_approved": None,
        "below_threshold": False,
        "scores": {"camelot": 0.92, "pdfplumber": 0.75, "docling": 0.60},
        "dev_samples": [_make_attachment(1), _make_attachment(2)],
        "test_samples": [_make_attachment(3)],
        "anonymized_tables": None,
        "generated_source": "def extract():\n    pass\n",
        "test_results": [_make_test_result(True)],
        "criteria": [
            ("C1", "Rows extracted > 0", True),
            ("C2", "Schema conforms", True),
            ("C3", "No nulls in date", False),
        ],
    }


@pytest.fixture()
def minimal_ctx():
    return {"bank": "SBI", "instrument": "savings", "success": False}


# ---------------------------------------------------------------------------
# render_report — structural
# ---------------------------------------------------------------------------


def test_render_returns_string(full_ctx):
    html = render_report(full_ctx)
    assert isinstance(html, str)


def test_render_is_html_document(full_ctx):
    html = render_report(full_ctx)
    assert "<!DOCTYPE html>" in html or "<html" in html


def test_render_contains_bank_and_instrument(full_ctx):
    html = render_report(full_ctx)
    assert "HDFC" in html
    assert "credit_card" in html


def test_render_contains_winner_extractor(full_ctx):
    html = render_report(full_ctx)
    assert "camelot" in html


def test_render_contains_generated_source(full_ctx):
    html = render_report(full_ctx)
    assert "def extract():" in html


def test_render_contains_test_sample_name(full_ctx):
    html = render_report(full_ctx)
    assert "statement_3.pdf" in html


def test_render_contains_score_extractor_names(full_ctx):
    html = render_report(full_ctx)
    assert "pdfplumber" in html
    assert "docling" in html


# ---------------------------------------------------------------------------
# render_report — self-contained (no external links)
# ---------------------------------------------------------------------------


def test_render_no_external_urls(full_ctx):
    html = render_report(full_ctx)
    assert "http://" not in html
    assert "https://" not in html


# ---------------------------------------------------------------------------
# render_report — XSS / escaping
# ---------------------------------------------------------------------------


def test_render_escapes_malicious_generated_source():
    ctx = {
        "bank": "X",
        "instrument": "Y",
        "success": True,
        "generated_source": "<script>alert('xss')</script>",
    }
    html = render_report(ctx)
    assert "<script>alert('xss')</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_escapes_malicious_anonymized_table():
    table = ExtractedTable(
        name="Evil",
        rows=[["<script>evil()</script>", "normal"]],
        page=1,
    )
    ctx = {
        "bank": "X",
        "instrument": "Y",
        "success": True,
        "anonymized_tables": [table],
    }
    html = render_report(ctx)
    assert "<script>evil()</script>" not in html
    assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# render_report — below_threshold warning
# ---------------------------------------------------------------------------


def test_below_threshold_shows_warning(full_ctx):
    full_ctx["below_threshold"] = True
    html = render_report(full_ctx)
    assert "below" in html.lower() or "threshold" in html.lower() or "warning" in html.lower()


def test_not_below_threshold_no_warning_banner(full_ctx):
    marker = "below the 0.70 threshold"
    assert marker not in render_report(dict(full_ctx, below_threshold=False))
    assert marker in render_report(dict(full_ctx, below_threshold=True))


# ---------------------------------------------------------------------------
# render_report — anonymization section
# ---------------------------------------------------------------------------


_ANON_BANNER = "Cells anonymized"


def test_no_anonymized_tables_hides_section(full_ctx):
    full_ctx["anonymized_tables"] = None
    html = render_report(full_ctx)
    assert _ANON_BANNER not in html


def test_anonymized_tables_present_shows_banner():
    table = ExtractedTable(name="T1", rows=[["A", "B"]], page=1)
    ctx = {
        "bank": "X",
        "instrument": "Y",
        "success": True,
        "anonymized_tables": [table],
    }
    html = render_report(ctx)
    assert _ANON_BANNER in html


# ---------------------------------------------------------------------------
# render_report — minimal ctx does not raise
# ---------------------------------------------------------------------------


def test_minimal_ctx_renders_without_error(minimal_ctx):
    html = render_report(minimal_ctx)
    assert "SBI" in html
    assert "savings" in html


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------


def test_write_report_returns_matching_path(full_ctx, tmp_path):
    path = write_report(full_ctx, out_dir=str(tmp_path))
    fname = Path(path).name
    assert re.match(r"autogen_HDFC_credit_card_\d{8}_\d{6}\.html", fname)


def test_write_report_file_exists(full_ctx, tmp_path):
    path = write_report(full_ctx, out_dir=str(tmp_path))
    assert Path(path).exists()


def test_write_report_content_matches_render(full_ctx, tmp_path):
    path = write_report(full_ctx, out_dir=str(tmp_path))
    written = Path(path).read_text(encoding="utf-8")
    # Body carries no auto-timestamp (it comes from ctx), so it must equal render output.
    assert written == render_report(full_ctx)
