"""Orchestrator: wires extraction → scoring → codegen → test → report."""

from __future__ import annotations

import ast
import os
from datetime import datetime
from pathlib import Path

from src import config as cfg
from src.autogen import ollama_client
from src.autogen.anonymization.anonymizer import TableAnonymizer
from src.autogen.codegen import opencode_runner, pii_guard
from src.autogen.codegen.prompt import build_prompt
from src.autogen.codegen.writer import LOCAL_MODEL, generate_and_write
from src.autogen.exceptions import UserAbortError
from src.autogen.extraction import runner as extraction_runner
from src.autogen.hitl.reviewer import review_and_confirm
from src.autogen.manifest import load_sample_set
from src.autogen.models import ExtractedTable, GenerationResult, ScoringResult, TestResult
from src.autogen.reporting.html_report import write_report
from src.autogen.runner import (
    execute_on_test_pdfs,
    lint_pipeline,
    load_pipeline,
    summarize_failures,
)
from src.autogen.validation.scorer import score_extractors
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

MAX_REFINE_ROUNDS = 2

LIBRARY_IMPORT: dict[str, str] = {
    "camelot": "camelot",
    "pdfplumber": "pdfplumber",
    "pymupdf": "fitz",
    "docling": "docling",
}


def run(
    bank: str,
    instrument: str,
    password: str,
    target_schema: dict[str, str],
    codegen_backend: str = "local",
    openrouter_api_key: str | None = None,
    codegen_model: str | None = None,
    use_cache: bool = True,
) -> GenerationResult:
    # 1. Pre-flight
    if codegen_backend == "cloud" and openrouter_api_key is None:
        raise ValueError("openrouter_api_key is required for cloud backend")
    if codegen_backend == "opencode":
        codegen_model = codegen_model or cfg.opencode["default_model"]
        if not codegen_model:
            raise ValueError("codegen_model (provider/model) is required for opencode backend")
        for binary in (cfg.opencode["binary"], cfg.opencode["bwrap_binary"]):
            if not os.path.exists(binary):
                raise RuntimeError(f"opencode backend requires {binary} — not found")
    if not ollama_client.is_reachable():
        raise RuntimeError("Ollama is not reachable — start the Ollama service and retry")
    required = [LOCAL_MODEL] if codegen_backend == "local" else []
    ollama_client.ensure_models(required)
    logger.info("Step 1/10: pre-flight ok (backend=%s)", codegen_backend)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Load samples
    samples = load_sample_set(bank, instrument)
    logger.info("Step 2/10: loaded samples (dev=%d, test=%d)", len(samples.dev), len(samples.test))

    # 3. Extraction
    logger.info("Step 3/10: extracting %d dev PDFs", len(samples.dev))
    per_pdf = {
        a.attachment_id: extraction_runner.run_all_extractors(
            a.attachment_id, a.physical_file, password, use_cache=use_cache
        )
        for a in samples.dev
    }
    logger.info("Step 3/10: extraction done")

    # 4. Score — opencode lets the agent pick the extractor from all anonymized inputs (no host score)
    scoring: ScoringResult | None = None
    winner = ""
    winner_library = ""
    winner_tables: list[ExtractedTable] = []
    if codegen_backend != "opencode":
        scoring = score_extractors(per_pdf)
        winner = scoring.winner
        winner_library = LIBRARY_IMPORT.get(winner, winner)
        winner_tables = [t for a in samples.dev for t in per_pdf[a.attachment_id].get(winner, [])]
        logger.info(
            "Step 4/10: winner=%s library=%s score=%.4f",
            winner,
            winner_library,
            scoring.winner_score,
        )
    else:
        logger.info("Step 4/10: scoring deferred to agent (opencode picks the extractor)")

    # 6. Anonymization + HITL gate (cloud and opencode send data off the trusted host)
    anonymization_applied = False
    hitl_approved: bool | None = None
    prompt_tables = winner_tables
    anonymized_tables = None
    # opencode: attachment_id -> extractor -> anonymized tables (all extractors, for agent selection)
    dev_anon: dict[str, dict[str, list[ExtractedTable]]] = {}

    if codegen_backend in ("cloud", "opencode"):
        anonymizer = TableAnonymizer()
        if codegen_backend == "opencode":
            # Anonymize every extractor for every dev attachment (shared anonymizer → consistent
            # masking) so the agent can compare all extractors and pick a winner in the sandbox.
            dev_anon = {
                a.attachment_id: {
                    ext: anonymizer.anonymize(tables)
                    for ext, tables in per_pdf[a.attachment_id].items()
                }
                for a in samples.dev
            }
            anonymized_tables = [
                t for per_ext in dev_anon.values() for tables in per_ext.values() for t in tables
            ]
            # 2-tier HITL groups: level 1 = extractor, level 2 = attachment.
            ext_names = list(per_pdf[samples.dev[0].attachment_id]) if samples.dev else []
            groups = [
                (
                    ext,
                    [
                        (
                            a.attachment_id,
                            per_pdf[a.attachment_id][ext],
                            dev_anon[a.attachment_id][ext],
                        )
                        for a in samples.dev
                    ],
                )
                for ext in ext_names
            ]
        else:
            anonymized_tables = anonymizer.anonymize(winner_tables)
            groups = [(winner, [("All tables", winner_tables, anonymized_tables)])]
        logger.info("Step 6/10: anonymized %d tables", len(anonymized_tables))
        preview_path = f"data/control/hitl_preview_{ts}.html"
        approved = False
        try:
            approved = review_and_confirm(groups, preview_path)
        except UserAbortError:
            approved = False

        if not approved:
            logger.info("Step 6/10: HITL rejected — aborting")
            ctx = _report_ctx(
                bank,
                instrument,
                ts,
                False,
                winner,
                scoring,
                codegen_backend,
                False,
                False,
                samples,
                anonymized_tables,
                None,
                [],
                None,
            )
            report_path = write_report(ctx)
            return GenerationResult(
                pipeline_path="",
                winner_extractor=winner,
                winner_score=scoring.winner_score if scoring else 0.0,
                codegen_backend=codegen_backend,
                anonymization_applied=False,
                hitl_approved=False,
                test_results=[],
                report_path=report_path,
                success=False,
            )

        prompt_tables = anonymized_tables
        anonymization_applied = True
        hitl_approved = True
        logger.info("Step 6/10: HITL approved")

    # 7–9. Generate → test → evaluate
    feedback: str | None = None
    pipeline_path: str = ""
    source: str = ""
    test_results: list[TestResult] = []
    criteria: list[tuple[str, str, bool]] = []
    success: bool = False
    lint_findings: list[str] | None = None

    if codegen_backend == "opencode":
        # Ralph-wiggum loop: the agent selects an extractor + generates code (iterating internally on
        # the anonymized dev fixtures); the host re-runs the composed pipeline against the real
        # held-out test set each round and re-invokes opencode with PII-safe structural feedback
        # until both test statements pass — or max_ralph_rounds is exhausted.
        max_rounds = cfg.opencode["max_ralph_rounds"]
        raw_by_extractor: dict[str, list[ExtractedTable]] = {}
        for a in samples.dev:
            for ext, tables in per_pdf[a.attachment_id].items():
                raw_by_extractor.setdefault(ext, []).extend(tables)
        eval_samples = samples.test or samples.dev
        which = "test" if samples.test else "dev (no held-out test set)"
        run_dir = opencode_runner.make_run_dir(bank, instrument)
        opencode_runner.prepare_sandbox(run_dir, dev_anon, target_schema)
        with opencode_runner.ToolBroker(run_dir, dev_anon):
            for refine_round in range(max_rounds):
                logger.info("Step 7/10: opencode ralph round %d/%d", refine_round + 1, max_rounds)
                winner, pipeline_path = opencode_runner.run_round(
                    run_dir,
                    bank,
                    instrument,
                    codegen_model or "",
                    raw_by_extractor,
                    password,
                    feedback,
                )
                source = Path(pipeline_path).read_text()
                test_results = execute_on_test_pdfs(
                    pipeline_path, eval_samples, password, target_schema
                )
                passed = sum(1 for r in test_results if r.success)
                logger.info(
                    "Step 8/10: %s validation %d/%d passed", which, passed, len(test_results)
                )
                criteria, success = _evaluate_criteria(
                    pipeline_path, source, test_results, target_schema
                )
                logger.info("Step 9/10: criteria evaluated — success=%s", success)
                if success or refine_round == max_rounds - 1:
                    break
                # Non-local branch emits only exception types, schema column names, and dtype
                # strings; run_round PII-scans it again before it enters the sandbox prompt.
                feedback = summarize_failures(test_results, target_schema, codegen_backend)
                if feedback is None:
                    break
                logger.info(
                    "Step 9/10: ralph round %d failed — re-running with host feedback",
                    refine_round + 1,
                )
        winner_library = LIBRARY_IMPORT.get(winner, winner)
        if pipeline_path:
            lint_findings = lint_pipeline(pipeline_path)
            logger.info("Step 10/10: ruff — %d finding(s)", len(lint_findings))
    else:
        for refine_round in range(MAX_REFINE_ROUNDS + 1):
            logger.info(
                "Step 7/10: generating pipeline (backend=%s, round %d/%d)",
                codegen_backend,
                refine_round + 1,
                MAX_REFINE_ROUNDS + 1,
            )
            prompt = build_prompt(
                bank,
                instrument,
                winner,
                winner_library,
                prompt_tables,
                target_schema,
                feedback=feedback,
            )
            pipeline_path = generate_and_write(
                bank,
                instrument,
                prompt,
                codegen_backend,
                openrouter_api_key=openrouter_api_key,
                input_tables=winner_tables,
                password=password,
            )
            source = Path(pipeline_path).read_text()

            test_results = execute_on_test_pdfs(
                pipeline_path, samples.test, password, target_schema
            )
            passed = sum(1 for r in test_results if r.success)
            logger.info("Step 8/10: tests %d/%d passed", passed, len(test_results))

            criteria, success = _evaluate_criteria(
                pipeline_path, source, test_results, target_schema
            )
            logger.info("Step 9/10: criteria evaluated — success=%s", success)

            if success or refine_round == MAX_REFINE_ROUNDS:
                break

            feedback = summarize_failures(test_results, target_schema, codegen_backend)
            if feedback is None:
                break
            if codegen_backend == "cloud":
                # PII safety guaranteed upstream: non-local branch emits only exception types,
                # schema column names, and dtype strings. Regex-level backstop below.
                pii_guard.scan(feedback, password=password, input_tables=winner_tables)
            logger.info(
                "Step 9/10: refining — round %d failed, regenerating with feedback",
                refine_round + 1,
            )

    # 10. Report
    ctx = _report_ctx(
        bank,
        instrument,
        ts,
        success,
        winner,
        scoring,
        codegen_backend,
        anonymization_applied,
        hitl_approved,
        samples,
        anonymized_tables if codegen_backend in ("cloud", "opencode") else None,
        source,
        test_results,
        criteria,
        lint_findings,
    )
    report_path = write_report(ctx)
    logger.info("Step 10/10: report written to %s", report_path)

    return GenerationResult(
        pipeline_path=pipeline_path,
        winner_extractor=winner,
        winner_score=scoring.winner_score if scoring else 0.0,
        codegen_backend=codegen_backend,
        anonymization_applied=anonymization_applied,
        hitl_approved=hitl_approved,
        test_results=test_results,
        report_path=report_path,
        success=success,
    )


def _evaluate_criteria(
    pipeline_path: str,
    source: str,
    test_results: list[TestResult],
    target_schema: dict[str, str],
) -> tuple[list[tuple[str, str, bool]], bool]:
    try:
        ast.parse(source)
        t1a = True
    except SyntaxError:
        t1a = False

    try:
        load_pipeline(pipeline_path)
        t1b = True
    except Exception:
        t1b = False

    t1c = bool(test_results) and all(r.success for r in test_results)
    t2a = bool(test_results) and all(
        r.columns is not None and all(col in r.columns for col in target_schema)
        for r in test_results
    )
    t2b = bool(test_results) and all(r.schema_conforms is True for r in test_results)

    criteria: list[tuple[str, str, bool]] = [
        ("T1-A", "Syntax valid", t1a),
        ("T1-B", "Import success", t1b),
        ("T1-C", "No runtime exception", t1c),
        ("T2-A", "Target columns present", t2a),
        ("T2-B", "Dtype conformance", t2b),
    ]
    return criteria, t1a and t1b and t1c and t2a and t2b


def _report_ctx(
    bank: str,
    instrument: str,
    ts: str,
    success: bool,
    winner: str,
    scoring: ScoringResult | None,
    codegen_backend: str,
    anonymization_applied: bool,
    hitl_approved: bool | None,
    samples,
    anonymized_tables,
    generated_source: str | None,
    test_results: list[TestResult],
    criteria,
    lint_findings: list[str] | None = None,
) -> dict:
    return {
        "bank": bank,
        "instrument": instrument,
        "timestamp": ts,
        "success": success,
        "winner_extractor": winner or None,
        "winner_score": scoring.winner_score if scoring else None,
        "codegen_backend": codegen_backend,
        "anonymization_applied": anonymization_applied,
        "hitl_approved": hitl_approved,
        "below_threshold": scoring.below_threshold if scoring else False,
        "scores": scoring.scores if scoring else None,
        "dev_samples": samples.dev,
        "test_samples": samples.test,
        "anonymized_tables": anonymized_tables,
        "generated_source": generated_source,
        "test_results": test_results,
        "criteria": criteria,
        "lint_findings": lint_findings,
    }


