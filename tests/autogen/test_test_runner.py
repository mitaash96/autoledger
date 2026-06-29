"""Tests for src/autogen/runner.py — TDD (Task 12a)."""

from __future__ import annotations

import textwrap

import polars as pl
import pytest

import src.autogen.runner as runner
from src.autogen.models import Attachment, TestResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TARGET_SCHEMA = {"date": "datetime", "amount": "float", "narration": "str"}


def _attachment(attachment_id: str, physical_file: str = "dummy.pdf") -> Attachment:
    return Attachment(
        attachment_id=attachment_id,
        raw_attachment_id=attachment_id,
        name=f"{attachment_id}.pdf",
        email_id="email-1",
        physical_file=physical_file,
    )


def _good_pipeline(tmp_path) -> str:
    """Write a valid pipeline that returns a tiny df matching TARGET_SCHEMA."""
    code = textwrap.dedent("""\
        import polars as pl
        from src.autogen.models import Attachment

        def transform(attachment, password):
            if getattr(attachment, "physical_file", None) == "BAD":
                raise ValueError("bad file")
            return pl.DataFrame({
                "date": pl.Series([None], dtype=pl.Datetime),
                "amount": pl.Series([1.5], dtype=pl.Float64),
                "narration": pl.Series(["tx1"], dtype=pl.Utf8),
            })
    """)
    path = tmp_path / "good.py"
    path.write_text(code)
    return str(path)


def _syntax_error_pipeline(tmp_path) -> str:
    path = tmp_path / "bad.py"
    path.write_text("def transform(attachment, password):\n  return ???\n")
    return str(path)


# ---------------------------------------------------------------------------
# load_pipeline
# ---------------------------------------------------------------------------


def test_load_pipeline_returns_module_with_transform(tmp_path):
    path = _good_pipeline(tmp_path)
    mod = runner.load_pipeline(path)
    assert callable(mod.transform)


def test_load_pipeline_raises_on_syntax_error(tmp_path):
    path = _syntax_error_pipeline(tmp_path)
    with pytest.raises(SyntaxError):
        runner.load_pipeline(path)


# ---------------------------------------------------------------------------
# dtype_conforms
# ---------------------------------------------------------------------------


def _df_matching() -> pl.DataFrame:
    return pl.DataFrame({
        "date": pl.Series([None], dtype=pl.Datetime),
        "amount": pl.Series([1.5], dtype=pl.Float64),
        "narration": pl.Series(["x"], dtype=pl.Utf8),
    })


def test_dtype_conforms_true_for_matching_schema():
    assert runner.dtype_conforms(_df_matching(), TARGET_SCHEMA) is True


def test_dtype_conforms_false_missing_column():
    df = pl.DataFrame({"date": pl.Series([None], dtype=pl.Datetime), "amount": pl.Series([1.0])})
    assert runner.dtype_conforms(df, TARGET_SCHEMA) is False


def test_dtype_conforms_false_wrong_dtype():
    df = pl.DataFrame({
        "date": pl.Series([None], dtype=pl.Datetime),
        "amount": pl.Series(["1.5"], dtype=pl.Utf8),  # str instead of float
        "narration": pl.Series(["x"], dtype=pl.Utf8),
    })
    assert runner.dtype_conforms(df, TARGET_SCHEMA) is False


def test_dtype_conforms_accepts_pl_string_for_str():
    # pl.String and pl.Utf8 are aliases; create with pl.String explicitly
    df = pl.DataFrame({
        "date": pl.Series([None], dtype=pl.Datetime),
        "amount": pl.Series([1.5], dtype=pl.Float32),
        "narration": pl.Series(["x"]).cast(pl.String),
    })
    assert runner.dtype_conforms(df, {"date": "datetime", "amount": "float", "narration": "str"}) is True


def test_dtype_conforms_accepts_pl_float32():
    df = pl.DataFrame({
        "date": pl.Series([None], dtype=pl.Date),
        "amount": pl.Series([1.5], dtype=pl.Float32),
        "narration": pl.Series(["x"], dtype=pl.Utf8),
    })
    assert runner.dtype_conforms(df, TARGET_SCHEMA) is True


# ---------------------------------------------------------------------------
# null_rates
# ---------------------------------------------------------------------------


def test_null_rates_no_nulls():
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    rates = runner.null_rates(df)
    assert rates == {"a": 0.0, "b": 0.0}


def test_null_rates_partial_nulls():
    df = pl.DataFrame({"a": [1, None, None], "b": [None, "y", "z"]})
    rates = runner.null_rates(df)
    assert abs(rates["a"] - 2 / 3) < 1e-9
    assert abs(rates["b"] - 1 / 3) < 1e-9


def test_null_rates_empty_df():
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Utf8)})
    rates = runner.null_rates(df)
    assert rates == {"a": 0.0, "b": 0.0}


# ---------------------------------------------------------------------------
# execute_on_test_pdfs — happy path
# ---------------------------------------------------------------------------


def test_execute_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TEST_OUTPUT_DIR", str(tmp_path))
    pipeline_path = _good_pipeline(tmp_path)
    samples = [_attachment("att-1"), _attachment("att-2")]

    results = runner.execute_on_test_pdfs(pipeline_path, samples, "", TARGET_SCHEMA)

    assert len(results) == 2
    for i, r in enumerate(results):
        assert r.attachment_id == samples[i].attachment_id
        assert r.success is True
        assert r.row_count == 1
        assert r.columns == ["date", "amount", "narration"]
        assert r.schema_conforms is True
        assert r.error is None
        assert r.parquet_path is not None
        assert (tmp_path / f"{samples[i].attachment_id}.parquet").exists()


# ---------------------------------------------------------------------------
# execute_on_test_pdfs — single sample failure does not abort others
# ---------------------------------------------------------------------------


def test_execute_partial_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TEST_OUTPUT_DIR", str(tmp_path))
    pipeline_path = _good_pipeline(tmp_path)
    # "BAD" physical_file causes transform to raise; "good.pdf" succeeds
    samples = [_attachment("att-bad", physical_file="BAD"), _attachment("att-ok")]

    results = runner.execute_on_test_pdfs(pipeline_path, samples, "", TARGET_SCHEMA)

    assert len(results) == 2
    bad, ok = results
    assert bad.success is False
    assert bad.error == "bad file"
    assert ok.success is True


# ---------------------------------------------------------------------------
# execute_on_test_pdfs — pipeline import failure
# ---------------------------------------------------------------------------


def test_execute_import_failure_returns_all_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TEST_OUTPUT_DIR", str(tmp_path))
    pipeline_path = _syntax_error_pipeline(tmp_path)
    samples = [_attachment("att-1"), _attachment("att-2")]

    results = runner.execute_on_test_pdfs(pipeline_path, samples, "", TARGET_SCHEMA)

    assert len(results) == 2
    for r in results:
        assert r.success is False
        assert "pipeline import failed" in r.error
        assert r.row_count is None
        assert r.columns is None


# ---------------------------------------------------------------------------
# execute_on_test_pdfs — new fields populated
# ---------------------------------------------------------------------------


def test_execute_happy_path_sets_column_dtypes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TEST_OUTPUT_DIR", str(tmp_path))
    pipeline_path = _good_pipeline(tmp_path)
    results = runner.execute_on_test_pdfs(pipeline_path, [_attachment("att-1")], "", TARGET_SCHEMA)
    r = results[0]
    assert r.success is True
    assert r.column_dtypes is not None
    assert set(r.column_dtypes.keys()) == {"date", "amount", "narration"}
    assert r.error_type is None


def test_execute_per_sample_failure_sets_error_type(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TEST_OUTPUT_DIR", str(tmp_path))
    pipeline_path = _good_pipeline(tmp_path)
    results = runner.execute_on_test_pdfs(
        pipeline_path, [_attachment("att-bad", physical_file="BAD")], "", TARGET_SCHEMA
    )
    r = results[0]
    assert r.success is False
    assert r.error_type == "ValueError"
    assert r.column_dtypes is None


def test_execute_import_failure_sets_error_type(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TEST_OUTPUT_DIR", str(tmp_path))
    pipeline_path = _syntax_error_pipeline(tmp_path)
    results = runner.execute_on_test_pdfs(pipeline_path, [_attachment("att-1")], "", TARGET_SCHEMA)
    r = results[0]
    assert r.success is False
    assert r.error_type == "SyntaxError"


# ---------------------------------------------------------------------------
# summarize_failures
# ---------------------------------------------------------------------------

_RAW_ERROR = "could not parse '12345678901 ACME SALARY' as Float64"


def _failed_result(**kwargs) -> TestResult:
    defaults = dict(
        attachment_id="att-1",
        name="stmt.pdf",
        success=False,
        row_count=None,
        columns=None,
        schema_conforms=None,
        null_rates=None,
        error=_RAW_ERROR,
        parquet_path=None,
        error_type="InvalidOperationError",
        column_dtypes=None,
    )
    defaults.update(kwargs)
    return TestResult(**defaults)


def _success_result() -> TestResult:
    return TestResult(
        attachment_id="att-ok",
        name="ok.pdf",
        success=True,
        row_count=3,
        columns=["date", "amount", "narration"],
        schema_conforms=True,
        null_rates={"date": 0.0, "amount": 0.0, "narration": 0.0},
        error=None,
        parquet_path="/tmp/ok.parquet",
    )


def test_summarize_failures_returns_none_when_all_succeeded():
    result = runner.summarize_failures([_success_result()], TARGET_SCHEMA, backend="cloud")
    assert result is None


def test_summarize_failures_cloud_excludes_raw_values():
    distinctive_filename = "DISTINCTIVE_TEST_FILENAME_xyz789.pdf"
    failed = _failed_result(name=distinctive_filename, columns=["narration"])  # missing date and amount
    summary = runner.summarize_failures([failed], TARGET_SCHEMA, backend="cloud")
    assert summary is not None
    assert "InvalidOperationError" in summary
    assert "ACME SALARY" not in summary
    assert "12345678901" not in summary
    assert distinctive_filename not in summary


def test_summarize_failures_cloud_includes_missing_columns():
    failed = _failed_result(columns=["narration"])
    summary = runner.summarize_failures([failed], TARGET_SCHEMA, backend="cloud")
    assert "date" in summary
    assert "amount" in summary


def test_summarize_failures_local_includes_raw_error():
    failed = _failed_result()
    summary = runner.summarize_failures([failed], TARGET_SCHEMA, backend="local")
    assert summary is not None
    assert _RAW_ERROR in summary


def test_summarize_failures_dtype_mismatch_line():
    schema = {"Txn Date": "datetime"}
    failed = _failed_result(
        columns=["Txn Date"],
        column_dtypes={"Txn Date": "String"},
    )
    summary = runner.summarize_failures([failed], schema, backend="cloud")
    assert summary is not None
    assert "Txn Date" in summary
    assert "datetime" in summary
    assert "String" in summary


def test_summarize_failures_no_mismatch_when_dtype_conforms():
    schema = {"amount": "float"}
    failed = _failed_result(
        columns=["amount"],
        column_dtypes={"amount": "Float64"},
    )
    summary = runner.summarize_failures([failed], schema, backend="cloud")
    # error_type header present but no mismatch line
    assert summary is not None
    assert "expected float but got" not in summary


def test_summarize_failures_all_missing_when_columns_none():
    # columns=None means we have no column info — all target cols count as missing
    failed = _failed_result(columns=None)
    summary = runner.summarize_failures([failed], TARGET_SCHEMA, backend="cloud")
    assert summary is not None
    for col in TARGET_SCHEMA:
        assert col in summary


def test_summarize_failures_cloud_deduplicates_identical_lines():
    """Multiple failures sharing the same error_type must not repeat identical lines."""
    failed1 = _failed_result(attachment_id="att-1", error_type="ValueError")
    failed2 = _failed_result(attachment_id="att-2", error_type="ValueError")
    summary = runner.summarize_failures([failed1, failed2], TARGET_SCHEMA, backend="cloud")
    assert summary is not None
    assert summary.count("error_type=ValueError") == 1


def test_summarize_failures_local_preserves_per_file_lines():
    """Local path must NOT deduplicate — each file's line is distinct."""
    failed1 = _failed_result(attachment_id="att-1", name="a.pdf", error_type="ValueError")
    failed2 = _failed_result(attachment_id="att-2", name="b.pdf", error_type="ValueError")
    summary = runner.summarize_failures([failed1, failed2], TARGET_SCHEMA, backend="local")
    assert summary is not None
    assert summary.count("error_type=ValueError") == 2


# ---------------------------------------------------------------------------
# lint_pipeline
# ---------------------------------------------------------------------------


def test_lint_pipeline_flags_violations(tmp_path):
    # E402: module import not at top of file → ruff must report at least one finding.
    path = tmp_path / "dirty.py"
    path.write_text("x = 1\nimport os\n")
    findings = runner.lint_pipeline(str(path))
    assert findings


def test_lint_pipeline_clean(tmp_path):
    path = tmp_path / "clean.py"
    path.write_text("x = 1\n")
    assert runner.lint_pipeline(str(path)) == []
