"""Test-execution runner: load a generated pipeline and validate it against test PDFs."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import polars as pl

from src import config as cfg
from src.autogen.models import Attachment, TestResult
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

TEST_OUTPUT_DIR = "data/tmp/autogen"

DTYPE_OK: dict[str, tuple] = {
    "datetime": (pl.Datetime, pl.Date),
    "float": (pl.Float32, pl.Float64),
    "str": (pl.String,),
}


def load_pipeline(pipeline_path: str):
    """Load a generated pipeline module from a file path (fresh, no sys.path mutation)."""
    spec = importlib.util.spec_from_file_location("_autogen_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot locate pipeline: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dtype_conforms(df: pl.DataFrame, target_schema: dict[str, str]) -> bool:
    """True iff every target column exists in df and its dtype matches the expected family."""
    for col, type_key in target_schema.items():
        if col not in df.columns:
            return False
        if not isinstance(df[col].dtype, DTYPE_OK[type_key]):
            return False
    return True


def null_rates(df: pl.DataFrame) -> dict[str, float]:
    """Per-column null fraction (0.0–1.0); returns 0.0 for each column in an empty df."""
    if df.height == 0:
        return {col: 0.0 for col in df.columns}
    return {col: df[col].null_count() / df.height for col in df.columns}


def execute_on_test_pdfs(
    pipeline_path: str,
    test_samples: list[Attachment],
    password: str,
    target_schema: dict[str, str],
) -> list[TestResult]:
    """Run a generated pipeline over the held-out test samples (one TestResult per sample)."""
    return _execute_on_pdfs(pipeline_path, test_samples, password, target_schema)


def execute_on_dev_pdfs(
    pipeline_path: str,
    dev_samples: list[Attachment],
    password: str,
    target_schema: dict[str, str],
) -> list[TestResult]:
    """Run a generated pipeline over the dev/sample attachments (one TestResult per sample).

    Used by the opencode backend as the end-to-end success gate: it validates the composed
    pipeline against the same attachments the anonymized fixtures were derived from.
    """
    return _execute_on_pdfs(pipeline_path, dev_samples, password, target_schema)


def _execute_on_pdfs(
    pipeline_path: str,
    samples: list[Attachment],
    password: str,
    target_schema: dict[str, str],
) -> list[TestResult]:
    """Load a generated pipeline and run its transform over each attachment."""
    try:
        module = load_pipeline(pipeline_path)
    except Exception as exc:
        error = f"pipeline import failed: {exc}"
        return [
            TestResult(
                attachment_id=a.attachment_id,
                name=a.name,
                success=False,
                row_count=None,
                columns=None,
                schema_conforms=None,
                null_rates=None,
                error=error,
                parquet_path=None,
                error_type=type(exc).__name__,
            )
            for a in samples
        ]

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    results: list[TestResult] = []
    for attachment in samples:
        try:
            df: pl.DataFrame = module.transform(attachment, password)
            path = f"{TEST_OUTPUT_DIR}/{attachment.attachment_id}.parquet"
            df.write_parquet(path)
            results.append(
                TestResult(
                    attachment_id=attachment.attachment_id,
                    name=attachment.name,
                    success=True,
                    row_count=df.height,
                    columns=df.columns,
                    schema_conforms=dtype_conforms(df, target_schema),
                    null_rates=null_rates(df),
                    error=None,
                    parquet_path=path,
                    column_dtypes={c: str(df[c].dtype) for c in df.columns},
                )
            )
        except Exception as exc:
            results.append(
                TestResult(
                    attachment_id=attachment.attachment_id,
                    name=attachment.name,
                    success=False,
                    row_count=None,
                    columns=None,
                    schema_conforms=None,
                    null_rates=None,
                    error=str(exc),
                    parquet_path=None,
                    error_type=type(exc).__name__,
                )
            )
    return results


def lint_pipeline(pipeline_path: str) -> list[str]:
    """Run `ruff check` on a generated pipeline; return finding lines ([] = clean). Never raises."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--quiet", "--output-format=concise", pipeline_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def summarize_failures(
    results: list[TestResult],
    target_schema: dict[str, str],
    backend: str,
) -> str | None:
    """Build a PII-safe failure summary for codegen prompt feedback."""
    failures = [r for r in results if r.success is False]
    if not failures:
        return None

    lines: list[str] = []
    for r in failures:
        if backend == "local":
            lines.append(f"[{r.name}] error_type={r.error_type or 'unknown'}")
        else:
            lines.append(f"error_type={r.error_type or 'unknown'}")
        if backend == "local" and r.error:
            lines.append(f"  detail: {r.error}")

        present = set(r.columns) if r.columns else set()
        missing = [col for col in target_schema if col not in present]
        if missing:
            lines.append(f"  missing columns: {', '.join(missing)}")

        if r.column_dtypes:
            for col, type_key in target_schema.items():
                if col in r.column_dtypes:
                    actual = r.column_dtypes[col]
                    expected_names = {t.__name__ for t in DTYPE_OK.get(type_key, [])}
                    if expected_names and not any(actual.startswith(name) for name in expected_names):
                        lines.append(f"  {col}: expected {type_key} but got {actual}")

    if backend != "local":
        seen: set[str] = set()
        deduped: list[str] = []
        for line in lines:
            if line not in seen:
                deduped.append(line)
                seen.add(line)
        lines = deduped

    return "\n".join(lines)
