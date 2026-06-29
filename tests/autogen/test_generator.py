"""Tests for src.autogen.generator.run orchestrator (Task 12c)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.autogen.exceptions import PiiLeakError, UserAbortError
from src.autogen.models import (
    ExtractedTable,
    GenerationResult,
    SampleSet,
    ScoringResult,
    TestResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PIPELINE = textwrap.dedent("""\
    import polars as pl

    def transform(attachment, password=""):
        return pl.DataFrame()
""")

TARGET_SCHEMA = {"date": "datetime", "amount": "float"}


def _make_attachment(aid: str = "a1") -> MagicMock:
    a = MagicMock()
    a.attachment_id = aid
    a.name = f"{aid}.pdf"
    a.physical_file = f"/tmp/{aid}.pdf"
    return a


def _make_sample_set() -> SampleSet:
    return SampleSet(dev=[_make_attachment("dev1")], test=[_make_attachment("tst1")])


def _make_scoring(winner: str = "pdfplumber") -> ScoringResult:
    return ScoringResult(
        scores={winner: 0.85},
        per_pdf_scores={"dev1": {winner: 0.85}},
        winner=winner,
        winner_score=0.85,
        below_threshold=False,
    )


def _make_table() -> ExtractedTable:
    return ExtractedTable(name="T", rows=[["date", "amount"], ["2024-01-01", "100"]], page=1)


def _make_test_result(success: bool = True, conforms: bool = True) -> TestResult:
    return TestResult(
        attachment_id="tst1",
        name="tst1.pdf",
        success=success,
        row_count=1,
        columns=list(TARGET_SCHEMA.keys()),
        schema_conforms=conforms,
        null_rates={"date": 0.0, "amount": 0.0},
        error=None,
        parquet_path=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline_file(tmp_path: Path) -> str:
    p = tmp_path / "bank_account.py"
    p.write_text(_VALID_PIPELINE)
    return str(p)


@pytest.fixture()
def _base_patches(pipeline_file):
    """Common happy-path patches for local backend."""
    table = _make_table()
    extraction_result = {"pdfplumber": [table], "textlayer": [table]}

    with (
        patch("src.autogen.generator.ollama_client") as mock_oc,
        patch("src.autogen.generator.load_sample_set", return_value=_make_sample_set()) as mock_ss,
        patch("src.autogen.generator.extraction_runner") as mock_runner,
        patch("src.autogen.generator.score_extractors", return_value=_make_scoring()) as mock_score,
        patch("src.autogen.generator.build_prompt", return_value="PROMPT") as mock_prompt,
        patch("src.autogen.generator.generate_and_write", return_value=pipeline_file) as mock_gen,
        patch(
            "src.autogen.generator.execute_on_test_pdfs", return_value=[_make_test_result()]
        ) as mock_exec,
        patch(
            "src.autogen.generator.write_report", return_value="/data/control/report.html"
        ) as mock_report,
        patch("src.autogen.generator.load_pipeline") as mock_load_pipeline,
    ):
        mock_oc.is_reachable.return_value = True
        mock_runner.run_all_extractors.return_value = extraction_result
        yield {
            "oc": mock_oc,
            "ss": mock_ss,
            "runner": mock_runner,
            "score": mock_score,
            "prompt": mock_prompt,
            "gen": mock_gen,
            "exec": mock_exec,
            "report": mock_report,
            "load_pipeline": mock_load_pipeline,
            "table": table,
            "pipeline_file": pipeline_file,
        }


# ---------------------------------------------------------------------------
# Pre-flight tests
# ---------------------------------------------------------------------------


class TestPreFlight:
    def test_cloud_without_key_raises_value_error(self):
        with patch("src.autogen.generator.ollama_client") as mock_oc:
            mock_oc.is_reachable.return_value = True
            from src.autogen.generator import run

            with pytest.raises(ValueError, match="openrouter_api_key"):
                run("bank", "account", "", TARGET_SCHEMA, codegen_backend="cloud")
            # nothing beyond pre-flight should have been called
            mock_oc.ensure_models.assert_not_called()

    def test_ollama_unreachable_raises_runtime_error(self):
        with patch("src.autogen.generator.ollama_client") as mock_oc:
            mock_oc.is_reachable.return_value = False
            from src.autogen.generator import run

            with pytest.raises(RuntimeError, match="[Oo]llama"):
                run("bank", "account", "", TARGET_SCHEMA)

    def test_ensure_models_failure_propagates(self):
        with patch("src.autogen.generator.ollama_client") as mock_oc:
            mock_oc.is_reachable.return_value = True
            mock_oc.ensure_models.side_effect = RuntimeError("model missing")
            from src.autogen.generator import run

            with pytest.raises(RuntimeError, match="model missing"):
                run("bank", "account", "", TARGET_SCHEMA)

    def test_local_backend_requests_local_model(self, _base_patches):
        from src.autogen.generator import run, LOCAL_MODEL

        run("bank", "account", "", TARGET_SCHEMA, codegen_backend="local")
        _base_patches["oc"].ensure_models.assert_called_once_with([LOCAL_MODEL])

    def test_cloud_backend_does_not_request_local_model(self, pipeline_file):
        table = _make_table()
        extraction_result = {"pdfplumber": [table], "textlayer": [table]}
        with (
            patch("src.autogen.generator.ollama_client") as mock_oc,
            patch("src.autogen.generator.load_sample_set", return_value=_make_sample_set()),
            patch("src.autogen.generator.extraction_runner") as mock_runner,
            patch("src.autogen.generator.score_extractors", return_value=_make_scoring()),
            patch("src.autogen.generator.build_prompt", return_value="PROMPT"),
            patch("src.autogen.generator.generate_and_write", return_value=pipeline_file),
            patch("src.autogen.generator.execute_on_test_pdfs", return_value=[_make_test_result()]),
            patch("src.autogen.generator.write_report", return_value="/data/control/r.html"),
            patch("src.autogen.generator.load_pipeline"),
            patch("src.autogen.generator.TableAnonymizer") as mock_anon,
            patch("src.autogen.generator.review_and_confirm", return_value=True),
        ):
            mock_oc.is_reachable.return_value = True
            mock_runner.run_all_extractors.return_value = extraction_result
            mock_anon.return_value.anonymize.return_value = [table]
            from src.autogen.generator import run

            run(
                "bank",
                "account",
                "",
                TARGET_SCHEMA,
                codegen_backend="cloud",
                openrouter_api_key="key",
            )
            called_models = mock_oc.ensure_models.call_args[0][0]
            from src.autogen.codegen.writer import LOCAL_MODEL

            assert LOCAL_MODEL not in called_models


# ---------------------------------------------------------------------------
# Local happy-path
# ---------------------------------------------------------------------------


class TestLocalHappyPath:
    def test_success_true(self, _base_patches):
        from src.autogen.generator import run

        result = run("bank", "account", "pw", TARGET_SCHEMA, codegen_backend="local")
        assert isinstance(result, GenerationResult)
        assert result.success is True

    def test_no_anonymization(self, _base_patches):
        from src.autogen.generator import run

        result = run("bank", "account", "pw", TARGET_SCHEMA, codegen_backend="local")
        assert result.anonymization_applied is False
        assert result.hitl_approved is None

    def test_write_report_called(self, _base_patches):
        from src.autogen.generator import run

        run("bank", "account", "pw", TARGET_SCHEMA, codegen_backend="local")
        _base_patches["report"].assert_called_once()

    def test_generate_and_write_backend_local(self, _base_patches):
        from src.autogen.generator import run

        run("bank", "account", "pw", TARGET_SCHEMA, codegen_backend="local")
        _base_patches["gen"].assert_called_once()
        _, kwargs = _base_patches["gen"].call_args
        assert kwargs.get("backend") == "local" or _base_patches["gen"].call_args[0][3] == "local"

    def test_generate_and_write_receives_raw_winner_tables(self, _base_patches):
        """input_tables must be the raw (unanonymized) winner tables."""
        from src.autogen.generator import run

        table = _base_patches["table"]
        run("bank", "account", "pw", TARGET_SCHEMA, codegen_backend="local")
        call_kwargs = _base_patches["gen"].call_args[1]
        assert call_kwargs.get("input_tables") == [table]

    def test_winner_library_mapped_for_pymupdf(self, pipeline_file):
        """Winner 'pymupdf' must map to library import 'fitz'."""
        table = _make_table()
        extraction_result = {"pymupdf": [table], "textlayer": [table]}
        with (
            patch("src.autogen.generator.ollama_client") as mock_oc,
            patch("src.autogen.generator.load_sample_set", return_value=_make_sample_set()),
            patch("src.autogen.generator.extraction_runner") as mock_runner,
            patch("src.autogen.generator.score_extractors", return_value=_make_scoring("pymupdf")),
            patch("src.autogen.generator.build_prompt", return_value="PROMPT") as mock_prompt,
            patch("src.autogen.generator.generate_and_write", return_value=pipeline_file),
            patch("src.autogen.generator.execute_on_test_pdfs", return_value=[_make_test_result()]),
            patch("src.autogen.generator.write_report", return_value="/data/control/r.html"),
            patch("src.autogen.generator.load_pipeline"),
        ):
            mock_oc.is_reachable.return_value = True
            mock_runner.run_all_extractors.return_value = extraction_result
            from src.autogen.generator import run

            run("bank", "account", "", TARGET_SCHEMA)
            # 4th positional arg to build_prompt is winner_library
            call_args = mock_prompt.call_args[0]
            assert call_args[3] == "fitz", f"Expected 'fitz', got {call_args[3]!r}"

    def test_result_has_correct_winner_extractor(self, _base_patches):
        from src.autogen.generator import run

        result = run("bank", "account", "", TARGET_SCHEMA)
        assert result.winner_extractor == "pdfplumber"

    def test_result_has_winner_score(self, _base_patches):
        from src.autogen.generator import run

        result = run("bank", "account", "", TARGET_SCHEMA)
        assert result.winner_score == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Cloud approved
# ---------------------------------------------------------------------------


class TestCloudApproved:
    @pytest.fixture()
    def _cloud_patches(self, pipeline_file):
        table = _make_table()
        anon_table = _make_table()
        extraction_result = {"pdfplumber": [table], "textlayer": [table]}
        with (
            patch("src.autogen.generator.ollama_client") as mock_oc,
            patch("src.autogen.generator.load_sample_set", return_value=_make_sample_set()),
            patch("src.autogen.generator.extraction_runner") as mock_runner,
            patch("src.autogen.generator.score_extractors", return_value=_make_scoring()),
            patch("src.autogen.generator.TableAnonymizer") as mock_anon_cls,
            patch("src.autogen.generator.review_and_confirm", return_value=True) as mock_review,
            patch("src.autogen.generator.build_prompt", return_value="PROMPT") as mock_prompt,
            patch(
                "src.autogen.generator.generate_and_write", return_value=pipeline_file
            ) as mock_gen,
            patch("src.autogen.generator.execute_on_test_pdfs", return_value=[_make_test_result()]),
            patch("src.autogen.generator.write_report", return_value="/r.html"),
            patch("src.autogen.generator.load_pipeline"),
        ):
            mock_oc.is_reachable.return_value = True
            mock_runner.run_all_extractors.return_value = extraction_result
            mock_anon_cls.return_value.anonymize.return_value = [anon_table]
            yield {
                "oc": mock_oc,
                "anon_cls": mock_anon_cls,
                "anon_table": anon_table,
                "raw_table": table,
                "review": mock_review,
                "prompt": mock_prompt,
                "gen": mock_gen,
            }

    def test_anonymization_applied(self, _cloud_patches):
        from src.autogen.generator import run

        result = run("b", "i", "", TARGET_SCHEMA, codegen_backend="cloud", openrouter_api_key="k")
        assert result.anonymization_applied is True

    def test_hitl_approved(self, _cloud_patches):
        from src.autogen.generator import run

        result = run("b", "i", "", TARGET_SCHEMA, codegen_backend="cloud", openrouter_api_key="k")
        assert result.hitl_approved is True

    def test_prompt_built_with_anonymized_tables(self, _cloud_patches):
        from src.autogen.generator import run

        run("b", "i", "", TARGET_SCHEMA, codegen_backend="cloud", openrouter_api_key="k")
        anon_table = _cloud_patches["anon_table"]
        call_args = _cloud_patches["prompt"].call_args[0]
        assert call_args[4] == [anon_table]

    def test_generate_and_write_receives_raw_tables(self, _cloud_patches):
        from src.autogen.generator import run

        run("b", "i", "", TARGET_SCHEMA, codegen_backend="cloud", openrouter_api_key="k")
        raw_table = _cloud_patches["raw_table"]
        call_kwargs = _cloud_patches["gen"].call_args[1]
        assert call_kwargs.get("input_tables") == [raw_table]

    def test_success_true(self, _cloud_patches):
        from src.autogen.generator import run

        result = run("b", "i", "", TARGET_SCHEMA, codegen_backend="cloud", openrouter_api_key="k")
        assert result.success is True


# ---------------------------------------------------------------------------
# opencode backend
# ---------------------------------------------------------------------------


class _DummyCtx:
    """Stand-in for opencode_runner.ToolBroker as a context manager."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestOpencodeBackend:
    @pytest.fixture()
    def _oc_patches(self, pipeline_file, tmp_path):
        table = _make_table()
        anon_table = _make_table()
        extraction_result = {"pdfplumber": [table], "textlayer": [table]}
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with (
            patch("src.autogen.generator.ollama_client") as mock_oc,
            patch("src.autogen.generator.os.path.exists", return_value=True),
            patch("src.autogen.generator.load_sample_set", return_value=_make_sample_set()),
            patch("src.autogen.generator.extraction_runner") as mock_runner,
            patch("src.autogen.generator.score_extractors") as mock_score,
            patch("src.autogen.generator.TableAnonymizer") as mock_anon_cls,
            patch("src.autogen.generator.review_and_confirm", return_value=True),
            patch("src.autogen.generator.opencode_runner.make_run_dir", return_value=run_dir),
            patch("src.autogen.generator.opencode_runner.prepare_sandbox") as mock_prep,
            patch("src.autogen.generator.opencode_runner.ToolBroker", _DummyCtx),
            patch(
                "src.autogen.generator.opencode_runner.run_round",
                return_value=("pdfplumber", pipeline_file),
            ) as mock_round,
            patch("src.autogen.generator.execute_on_test_pdfs", return_value=[_make_test_result()]),
            patch("src.autogen.generator.lint_pipeline", return_value=[]),
            patch("src.autogen.generator.write_report", return_value="/r.html"),
            patch("src.autogen.generator.load_pipeline"),
        ):
            mock_oc.is_reachable.return_value = True
            mock_runner.run_all_extractors.return_value = extraction_result
            mock_anon_cls.return_value.anonymize.return_value = [anon_table]
            yield {
                "round": mock_round,
                "prep": mock_prep,
                "score": mock_score,
                "anon_table": anon_table,
            }

    def test_default_model_fallback(self, _oc_patches):
        from src import config as cfg
        from src.autogen.generator import run

        run("b", "i", "", TARGET_SCHEMA, codegen_backend="opencode")
        # run_round(run_dir, bank, instrument, model, ...) — model is the 4th positional arg.
        assert _oc_patches["round"].call_args.args[3] == cfg.opencode["default_model"]

    def test_explicit_model_used(self, _oc_patches):
        from src.autogen.generator import run

        run(
            "b",
            "i",
            "",
            TARGET_SCHEMA,
            codegen_backend="opencode",
            codegen_model="opencode/mimo-v2.5-free",
        )
        assert _oc_patches["round"].call_args.args[3] == "opencode/mimo-v2.5-free"

    def test_opencode_skips_scorer(self, _oc_patches):
        """The agent selects the extractor in the sandbox — the host scorer must not run."""
        from src.autogen.generator import run

        run("b", "i", "", TARGET_SCHEMA, codegen_backend="opencode")
        _oc_patches["score"].assert_not_called()

    def test_opencode_anonymizes_all_extractors(self, _oc_patches):
        from src.autogen.generator import run

        run("b", "i", "", TARGET_SCHEMA, codegen_backend="opencode")
        # prepare_sandbox(run_dir, dev_anon, schema) — dev_anon is attachment -> extractor -> tables.
        dev_anon = _oc_patches["prep"].call_args.args[1]
        assert dev_anon == {
            "dev1": {
                "pdfplumber": [_oc_patches["anon_table"]],
                "textlayer": [_oc_patches["anon_table"]],
            }
        }

    def test_winner_from_run_round(self, _oc_patches):
        from src.autogen.generator import run

        result = run("b", "i", "", TARGET_SCHEMA, codegen_backend="opencode")
        assert result.winner_extractor == "pdfplumber"

    def test_run_round_called_once_on_success(self, _oc_patches):
        from src.autogen.generator import run

        run("b", "i", "", TARGET_SCHEMA, codegen_backend="opencode")
        assert _oc_patches["round"].call_count == 1

    def test_ralph_loops_until_pass(self, pipeline_file, tmp_path):
        """First host-test round fails, second passes → run_round called twice, success True."""
        table = _make_table()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with (
            patch("src.autogen.generator.ollama_client") as mock_oc,
            patch("src.autogen.generator.os.path.exists", return_value=True),
            patch("src.autogen.generator.load_sample_set", return_value=_make_sample_set()),
            patch("src.autogen.generator.extraction_runner") as mock_runner,
            patch("src.autogen.generator.score_extractors"),
            patch("src.autogen.generator.TableAnonymizer") as mock_anon_cls,
            patch("src.autogen.generator.review_and_confirm", return_value=True),
            patch("src.autogen.generator.opencode_runner.make_run_dir", return_value=run_dir),
            patch("src.autogen.generator.opencode_runner.prepare_sandbox"),
            patch("src.autogen.generator.opencode_runner.ToolBroker", _DummyCtx),
            patch(
                "src.autogen.generator.opencode_runner.run_round",
                return_value=("pdfplumber", pipeline_file),
            ) as mock_round,
            patch(
                "src.autogen.generator.execute_on_test_pdfs",
                side_effect=[[_make_test_result(success=False)], [_make_test_result(success=True)]],
            ),
            patch("src.autogen.generator.summarize_failures", return_value="error_type=runtime"),
            patch("src.autogen.generator.lint_pipeline", return_value=[]),
            patch("src.autogen.generator.write_report", return_value="/r.html"),
            patch("src.autogen.generator.load_pipeline"),
        ):
            mock_oc.is_reachable.return_value = True
            mock_runner.run_all_extractors.return_value = {"pdfplumber": [table]}
            mock_anon_cls.return_value.anonymize.return_value = [table]
            from src.autogen.generator import run

            result = run("b", "i", "", TARGET_SCHEMA, codegen_backend="opencode")
        assert mock_round.call_count == 2
        # Second round must carry the PII-safe structural feedback (7th positional arg).
        assert mock_round.call_args_list[1].args[6] == "error_type=runtime"
        assert result.success is True

    def test_ralph_bounded_by_max_rounds(self, pipeline_file, tmp_path):
        """All host-test rounds fail → run_round called max_ralph_rounds times, success False."""
        from src import config as cfg

        table = _make_table()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with (
            patch("src.autogen.generator.ollama_client") as mock_oc,
            patch("src.autogen.generator.os.path.exists", return_value=True),
            patch("src.autogen.generator.load_sample_set", return_value=_make_sample_set()),
            patch("src.autogen.generator.extraction_runner") as mock_runner,
            patch("src.autogen.generator.score_extractors"),
            patch("src.autogen.generator.TableAnonymizer") as mock_anon_cls,
            patch("src.autogen.generator.review_and_confirm", return_value=True),
            patch("src.autogen.generator.opencode_runner.make_run_dir", return_value=run_dir),
            patch("src.autogen.generator.opencode_runner.prepare_sandbox"),
            patch("src.autogen.generator.opencode_runner.ToolBroker", _DummyCtx),
            patch(
                "src.autogen.generator.opencode_runner.run_round",
                return_value=("pdfplumber", pipeline_file),
            ) as mock_round,
            patch(
                "src.autogen.generator.execute_on_test_pdfs",
                return_value=[_make_test_result(success=False)],
            ),
            patch("src.autogen.generator.summarize_failures", return_value="error_type=runtime"),
            patch("src.autogen.generator.lint_pipeline", return_value=[]),
            patch("src.autogen.generator.write_report", return_value="/r.html"),
            patch("src.autogen.generator.load_pipeline"),
        ):
            mock_oc.is_reachable.return_value = True
            mock_runner.run_all_extractors.return_value = {"pdfplumber": [table]}
            mock_anon_cls.return_value.anonymize.return_value = [table]
            from src.autogen.generator import run

            result = run("b", "i", "", TARGET_SCHEMA, codegen_backend="opencode")
        assert mock_round.call_count == cfg.opencode["max_ralph_rounds"]
        assert result.success is False


# ---------------------------------------------------------------------------
# Cloud aborted
# ---------------------------------------------------------------------------


class TestCloudAborted:
    def _run_abort(self, review_return):
        table = _make_table()
        extraction_result = {"pdfplumber": [table], "textlayer": [table]}
        with (
            patch("src.autogen.generator.ollama_client") as mock_oc,
            patch("src.autogen.generator.load_sample_set", return_value=_make_sample_set()),
            patch("src.autogen.generator.extraction_runner") as mock_runner,
            patch("src.autogen.generator.score_extractors", return_value=_make_scoring()),
            patch("src.autogen.generator.TableAnonymizer") as mock_anon_cls,
            patch("src.autogen.generator.review_and_confirm", side_effect=review_return) as _,
            patch("src.autogen.generator.build_prompt", return_value="PROMPT"),
            patch("src.autogen.generator.generate_and_write") as mock_gen,
            patch("src.autogen.generator.execute_on_test_pdfs"),
            patch("src.autogen.generator.write_report", return_value="/r.html") as mock_report,
            patch("src.autogen.generator.load_pipeline"),
        ):
            mock_oc.is_reachable.return_value = True
            mock_runner.run_all_extractors.return_value = extraction_result
            mock_anon_cls.return_value.anonymize.return_value = [table]
            from src.autogen.generator import run

            result = run(
                "b", "i", "", TARGET_SCHEMA, codegen_backend="cloud", openrouter_api_key="k"
            )
            return result, mock_gen, mock_report

    def test_returns_result_not_exception_on_false(self):
        result, _, _ = self._run_abort([False])
        assert isinstance(result, GenerationResult)

    def test_success_false_on_abort(self):
        result, _, _ = self._run_abort([False])
        assert result.success is False

    def test_hitl_approved_false_on_abort(self):
        result, _, _ = self._run_abort([False])
        assert result.hitl_approved is False

    def test_generate_and_write_not_called_on_abort(self):
        _, mock_gen, _ = self._run_abort([False])
        mock_gen.assert_not_called()

    def test_report_still_written_on_abort(self):
        _, _, mock_report = self._run_abort([False])
        mock_report.assert_called_once()

    def test_user_abort_error_also_returns_result(self):
        result, mock_gen, mock_report = self._run_abort(UserAbortError("aborted"))
        assert isinstance(result, GenerationResult)
        assert result.success is False
        assert result.hitl_approved is False
        mock_gen.assert_not_called()
        mock_report.assert_called_once()


# ---------------------------------------------------------------------------
# Partial success (test failure)
# ---------------------------------------------------------------------------


class TestSuccessCriteria:
    def test_success_false_when_test_result_fails(self, _base_patches):
        _base_patches["exec"].return_value = [_make_test_result(success=False)]
        from src.autogen.generator import run

        result = run("bank", "account", "", TARGET_SCHEMA)
        assert result.success is False

    def test_result_returned_even_on_test_failure(self, _base_patches):
        _base_patches["exec"].return_value = [_make_test_result(success=False)]
        from src.autogen.generator import run

        result = run("bank", "account", "", TARGET_SCHEMA)
        assert isinstance(result, GenerationResult)

    def test_report_written_on_test_failure(self, _base_patches):
        _base_patches["exec"].return_value = [_make_test_result(success=False)]
        from src.autogen.generator import run

        run("bank", "account", "", TARGET_SCHEMA)
        _base_patches["report"].assert_called_once()

    def test_success_false_when_schema_not_conforms(self, _base_patches):
        _base_patches["exec"].return_value = [_make_test_result(success=True, conforms=False)]
        from src.autogen.generator import run

        result = run("bank", "account", "", TARGET_SCHEMA)
        assert result.success is False

    def test_success_false_when_no_test_results(self, _base_patches):
        _base_patches["exec"].return_value = []
        from src.autogen.generator import run

        result = run("bank", "account", "", TARGET_SCHEMA)
        assert result.success is False


# ---------------------------------------------------------------------------
# Refinement loop (FB-3)
# ---------------------------------------------------------------------------


class TestRefinementLoop:
    def test_retry_on_failure_succeeds_second_round(self, _base_patches):
        """First attempt fails, second succeeds → gen called twice, success True."""
        _base_patches["exec"].side_effect = [
            [_make_test_result(success=False)],
            [_make_test_result(success=True)],
        ]
        with patch("src.autogen.generator.summarize_failures", return_value="error_type=runtime"):
            from src.autogen.generator import run

            result = run("bank", "account", "", TARGET_SCHEMA)
        assert _base_patches["gen"].call_count == 2
        assert result.success is True

    def test_all_rounds_fail_bounded(self, _base_patches):
        """All attempts fail → gen called MAX_REFINE_ROUNDS+1 times, success False."""
        from src.autogen.generator import MAX_REFINE_ROUNDS

        _base_patches["exec"].return_value = [_make_test_result(success=False)]
        with patch("src.autogen.generator.summarize_failures", return_value="error_type=runtime"):
            from src.autogen.generator import run

            result = run("bank", "account", "", TARGET_SCHEMA)
        assert _base_patches["gen"].call_count == MAX_REFINE_ROUNDS + 1
        assert result.success is False

    def test_no_refinement_when_first_succeeds(self, _base_patches):
        """First attempt already succeeds → gen called exactly once."""
        from src.autogen.generator import run

        result = run("bank", "account", "", TARGET_SCHEMA)
        assert _base_patches["gen"].call_count == 1
        assert result.success is True

    def test_cloud_refinement_uses_structural_feedback_only(self, pipeline_file):
        """Real summarize_failures cloud branch must strip PII from refinement prompt."""
        table = _make_table()
        anon_table = _make_table()
        extraction_result = {"pdfplumber": [table], "textlayer": [table]}

        raw_error = "could not parse '12345678901 ACME SALARY' as Float64"
        failing_result = TestResult(
            attachment_id="tst1",
            name="SECRET_STATEMENT_xyz.pdf",
            success=False,
            row_count=None,
            columns=None,
            schema_conforms=None,
            null_rates=None,
            error=raw_error,
            parquet_path=None,
            error_type="InvalidOperationError",
            column_dtypes=None,
        )

        with (
            patch("src.autogen.generator.ollama_client") as mock_oc,
            patch("src.autogen.generator.load_sample_set", return_value=_make_sample_set()),
            patch("src.autogen.generator.extraction_runner") as mock_runner,
            patch("src.autogen.generator.score_extractors", return_value=_make_scoring()),
            patch("src.autogen.generator.TableAnonymizer") as mock_anon_cls,
            patch("src.autogen.generator.review_and_confirm", return_value=True),
            patch("src.autogen.generator.build_prompt", return_value="PROMPT") as mock_prompt,
            patch("src.autogen.generator.generate_and_write", return_value=pipeline_file),
            patch(
                "src.autogen.generator.execute_on_test_pdfs",
                side_effect=[
                    [failing_result],
                    [_make_test_result(success=True)],
                ],
            ),
            patch("src.autogen.generator.write_report", return_value="/r.html"),
            patch("src.autogen.generator.load_pipeline"),
            # summarize_failures is NOT mocked — we exercise the real cloud branch
        ):
            mock_oc.is_reachable.return_value = True
            mock_runner.run_all_extractors.return_value = extraction_result
            mock_anon_cls.return_value.anonymize.return_value = [anon_table]
            from src.autogen.generator import run

            result = run(
                "b", "i", "", TARGET_SCHEMA, codegen_backend="cloud", openrouter_api_key="key"
            )

        assert result.success is True
        assert mock_prompt.call_count == 2
        feedback = mock_prompt.call_args_list[1][1]["feedback"]
        assert "12345678901" not in feedback
        assert "ACME SALARY" not in feedback
        assert "SECRET_STATEMENT_xyz" not in feedback
        assert "InvalidOperationError" in feedback

    def test_cloud_pii_backstop_propagates(self, pipeline_file):
        """Cloud backend: feedback containing 8+ digit run raises PiiLeakError out of run."""
        table = _make_table()
        anon_table = _make_table()
        extraction_result = {"pdfplumber": [table], "textlayer": [table]}
        with (
            patch("src.autogen.generator.ollama_client") as mock_oc,
            patch("src.autogen.generator.load_sample_set", return_value=_make_sample_set()),
            patch("src.autogen.generator.extraction_runner") as mock_runner,
            patch("src.autogen.generator.score_extractors", return_value=_make_scoring()),
            patch("src.autogen.generator.TableAnonymizer") as mock_anon_cls,
            patch("src.autogen.generator.review_and_confirm", return_value=True),
            patch("src.autogen.generator.build_prompt", return_value="PROMPT"),
            patch("src.autogen.generator.generate_and_write", return_value=pipeline_file),
            patch(
                "src.autogen.generator.execute_on_test_pdfs",
                return_value=[_make_test_result(success=False)],
            ),
            patch("src.autogen.generator.write_report", return_value="/r.html"),
            patch("src.autogen.generator.load_pipeline"),
            patch(
                "src.autogen.generator.summarize_failures",
                return_value="error_type=runtime 12345678901",
            ),
        ):
            mock_oc.is_reachable.return_value = True
            mock_runner.run_all_extractors.return_value = extraction_result
            mock_anon_cls.return_value.anonymize.return_value = [anon_table]
            from src.autogen.generator import run

            with pytest.raises(PiiLeakError):
                run("b", "i", "", TARGET_SCHEMA, codegen_backend="cloud", openrouter_api_key="key")
