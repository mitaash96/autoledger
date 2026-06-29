"""Tests for src.autogen.models — dataclasses and serialization helpers."""

from src.autogen.models import (
    ExtractedTable,
    ExtractionResult,
    GenerationResult,
    SampleSet,
    ScoringResult,
    TestResult,
    table_from_dict,
    table_to_dict,
)
from src.email.schemas import Attachment


# ---------------------------------------------------------------------------
# table_to_dict / table_from_dict round-trip
# ---------------------------------------------------------------------------


def _make_attachment(**kwargs) -> Attachment:
    defaults = dict(
        attachment_id="att-1",
        raw_attachment_id="raw-1",
        name="statement.pdf",
        email_id="email-1",
    )
    defaults.update(kwargs)
    return Attachment(**defaults)


class TestTableRoundTrip:
    def test_round_trip_with_all_fields(self):
        t = ExtractedTable(
            name="Transactions",
            rows=[["Date", "Amount"], ["2024-01-01", "100.00"]],
            page=3,
        )
        d = table_to_dict(t)
        assert isinstance(d, dict)
        restored = table_from_dict(d)
        assert restored.name == t.name
        assert restored.rows == t.rows
        assert restored.page == t.page

    def test_round_trip_with_none_fields(self):
        t = ExtractedTable(name=None, rows=[], page=None)
        d = table_to_dict(t)
        restored = table_from_dict(d)
        assert restored.name is None
        assert restored.rows == []
        assert restored.page is None

    def test_dict_is_json_serializable(self):
        import json

        t = ExtractedTable(name="T", rows=[["a", "b"]], page=1)
        d = table_to_dict(t)
        # must not raise
        json.dumps(d)

    def test_to_dict_keys(self):
        t = ExtractedTable(name="T", rows=[], page=2)
        d = table_to_dict(t)
        assert set(d.keys()) == {"name", "rows", "page"}


# ---------------------------------------------------------------------------
# Dataclass construction sanity checks
# ---------------------------------------------------------------------------


class TestExtractedTable:
    def test_basic_construction(self):
        t = ExtractedTable(name="Test", rows=[["A"]], page=1)
        assert t.name == "Test"
        assert t.rows == [["A"]]
        assert t.page == 1

    def test_defaults_optional_fields(self):
        t = ExtractedTable(name=None, rows=[], page=None)
        assert t.name is None
        assert t.page is None


class TestExtractionResult:
    def test_construction(self):
        tables = [ExtractedTable(name=None, rows=[], page=1)]
        r = ExtractionResult(
            extractor="pdfplumber",
            attachment_id="att-1",
            tables=tables,
            error=None,
        )
        assert r.extractor == "pdfplumber"
        assert r.attachment_id == "att-1"
        assert len(r.tables) == 1
        assert r.error is None

    def test_with_error(self):
        r = ExtractionResult(
            extractor="camelot",
            attachment_id="att-2",
            tables=[],
            error="DecryptionError",
        )
        assert r.error == "DecryptionError"
        assert r.tables == []


class TestSampleSet:
    def test_construction(self):
        a1 = _make_attachment(attachment_id="a1", raw_attachment_id="r1")
        a2 = _make_attachment(attachment_id="a2", raw_attachment_id="r2")
        ss = SampleSet(dev=[a1], test=[a2])
        assert len(ss.dev) == 1
        assert len(ss.test) == 1

    def test_empty(self):
        ss = SampleSet(dev=[], test=[])
        assert ss.dev == []
        assert ss.test == []


class TestScoringResult:
    def test_construction(self):
        sr = ScoringResult(
            scores={"pdfplumber": 0.85, "camelot": 0.60},
            per_pdf_scores={"att-1": {"pdfplumber": 0.85, "camelot": 0.60}},
            winner="pdfplumber",
            winner_score=0.85,
            below_threshold=False,
        )
        assert sr.winner == "pdfplumber"
        assert sr.below_threshold is False

    def test_below_threshold(self):
        sr = ScoringResult(
            scores={"pdfplumber": 0.50},
            per_pdf_scores={},
            winner="pdfplumber",
            winner_score=0.50,
            below_threshold=True,
        )
        assert sr.below_threshold is True


class TestTestResult:
    def test_success(self):
        tr = TestResult(
            attachment_id="att-1",
            name="statement.pdf",
            success=True,
            row_count=42,
            columns=["date", "amount"],
            schema_conforms=True,
            null_rates={"date": 0.0, "amount": 0.02},
            error=None,
            parquet_path="/data/out.parquet",
        )
        assert tr.success is True
        assert tr.row_count == 42

    def test_failure(self):
        tr = TestResult(
            attachment_id="att-2",
            name="broken.pdf",
            success=False,
            row_count=None,
            columns=None,
            schema_conforms=None,
            null_rates=None,
            error="ParseError",
            parquet_path=None,
        )
        assert tr.success is False
        assert tr.error == "ParseError"

    def test_new_fields_default_to_none(self):
        # construction without error_type / column_dtypes must still work
        tr = TestResult(
            attachment_id="att-3",
            name="stmt.pdf",
            success=True,
            row_count=5,
            columns=["date", "amount"],
            schema_conforms=True,
            null_rates={"date": 0.0, "amount": 0.0},
            error=None,
            parquet_path="/tmp/out.parquet",
        )
        assert tr.error_type is None
        assert tr.column_dtypes is None


class TestGenerationResult:
    def test_construction(self):
        gr = GenerationResult(
            pipeline_path="/pipelines/hdfc_cc.py",
            winner_extractor="pdfplumber",
            winner_score=0.85,
            codegen_backend="local",
            anonymization_applied=True,
            hitl_approved=None,
            test_results=[],
            report_path="/reports/run.json",
            success=True,
        )
        assert gr.codegen_backend == "local"
        assert gr.hitl_approved is None
        assert gr.success is True


class TestAttachmentReExport:
    """Attachment must be importable from src.autogen.models."""

    def test_import(self):
        from src.autogen.models import Attachment as ModelAttachment

        assert ModelAttachment is Attachment
