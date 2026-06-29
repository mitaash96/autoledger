"""Tests for the sandboxed opencode codegen backend and its host broker."""

import json
import shutil
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src import config as cfg
from src.autogen.codegen import opencode_runner as oc
from src.autogen.codegen import pfc_tool_broker as broker
from src.autogen.exceptions import CodegenError, PiiLeakError
from src.autogen.models import ExtractedTable

SCHEMA = {"Txn Date": "str", "Amount": "str"}
ANON_TABLES = [
    ExtractedTable(name="txns", rows=[["Txn Date", "Amount"], ["01/01/2020", "100.00"]], page=1),
]
# attachment_id -> extractor -> anonymized tables (agent compares extractors, picks a winner).
DEV_ANON = {"dev1": {"pdfplumber": ANON_TABLES, "docling": ANON_TABLES}}
RAW_SECRET = "ACMECORPSALARYCREDIT"  # a raw value that must never leak
PASSWORD = "S3cretPass"


def _good_pipeline() -> str:
    return "import polars as pl\ndef tables_to_dataframe(tables):\n    return pl.DataFrame()\n"


# --- prepare_sandbox -------------------------------------------------------


def test_prepare_sandbox_writes_only_pii_free_assets(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "fixtures").mkdir(parents=True)
    oc.prepare_sandbox(run_dir, DEV_ANON, SCHEMA)

    for name in (
        "fixtures/anon_index.json",
        "fixtures/anon_pdfplumber_dev1.json",
        "fixtures/anon_docling_dev1.json",
        "target_schema.json",
        "selectable.json",
        "test_logic.py",
        "AGENTS.md",
        "opencode.json",
        ".opencode/tools/pfc_samples.ts",
    ):
        assert (run_dir / name).exists(), f"missing {name}"

    # No pre-baked extract_tables stub — the host composes it from the agent's winner.txt choice.
    assert not (run_dir / "extract_tables_stub.py").exists()

    index = json.loads((run_dir / "fixtures/anon_index.json").read_text())
    assert index == {"extractors": ["pdfplumber", "docling"], "attachments": ["dev1"]}
    assert json.loads((run_dir / "selectable.json").read_text()) == list(oc._VALID_WINNERS)

    agents = (run_dir / "AGENTS.md").read_text()
    assert "tables_to_dataframe" in agents
    assert "pfc_samples" in agents  # mandates the custom tool
    assert "non-empty" in agents  # mandates non-empty frames
    assert "CHECKPOINT 1" in agents and "CHECKPOINT 2" in agents  # two-checkpoint workflow
    assert "winner.txt" in agents  # agent records its extractor choice
    assert "camelot" in agents  # warns about the header-in-row-0 hazard
    assert "fixtures/anon_<extractor>_<attachment_id>.json" in agents  # file->extractor mapping
    assert "Step 1." in agents and "Step 4." in agents  # CoT workflow steps
    assert "Workflow" in agents


def test_prepare_sandbox_writes_custom_tool(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "fixtures").mkdir(parents=True)
    oc.prepare_sandbox(run_dir, DEV_ANON, SCHEMA)
    ts = (run_dir / ".opencode/tools/pfc_samples.ts").read_text()
    assert "@opencode-ai/plugin" in ts
    assert "describe_dev_set" in ts
    assert "regenerate_anonymized_tables" in ts


# --- build_bwrap_argv ------------------------------------------------------


def test_bwrap_argv_excludes_repo_data_env(tmp_path):
    argv = oc.build_bwrap_argv(tmp_path / "run", "openrouter/x")
    repo_root = str(Path(cfg.opencode["venv_dir"]).resolve().parent)
    assert "--clearenv" in argv
    assert repo_root not in argv  # repo root never bound
    assert not any(a.endswith("/data") for a in argv)  # data/ never bound
    assert not any(a.endswith(".env") for a in argv)  # .env never bound
    assert oc.WORKDIR in argv  # run dir bound at /work


def test_bwrap_argv_uses_project_venv(tmp_path):
    import os

    argv = oc.build_bwrap_argv(tmp_path / "run", "opencode/deepseek-v4-flash-free")
    venv = os.path.abspath(cfg.opencode["venv_dir"])
    assert "--ro-bind" in argv and venv in argv
    path_idx = argv.index("PATH")
    assert argv[path_idx + 1].startswith(f"{venv}/bin")
    venv_idx = argv.index("VIRTUAL_ENV")
    assert argv[venv_idx + 1] == venv


def test_bwrap_argv_does_not_leak_parent_env(monkeypatch, tmp_path):
    monkeypatch.setenv("hdfc_account", "TOPSECRET")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-allowed")
    argv = oc.build_bwrap_argv(tmp_path / "run", "openrouter/x")
    assert "TOPSECRET" not in argv
    assert "hdfc_account" not in argv
    assert "sk-allowed" in argv  # allowlisted model credential is forwarded


def test_bwrap_argv_binds_available_credentials(tmp_path, monkeypatch):
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (auth_dir / "auth.json").write_text("{}")
    (auth_dir / "account.json").write_text("{}")
    monkeypatch.setitem(cfg.opencode, "auth_dir", str(auth_dir))
    argv = oc.build_bwrap_argv(tmp_path / "run", "opencode/deepseek-v4-flash-free")
    joined = " ".join(argv)
    assert f"{oc.SANDBOX_HOME}/.local/share/opencode/auth.json" in joined
    assert f"{oc.SANDBOX_HOME}/.local/share/opencode/account.json" in joined


# --- build_final_pipeline ---------------------------------------------------


def test_build_final_pipeline_composes_correctly(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "fixtures").mkdir(parents=True)
    oc.prepare_sandbox(run_dir, DEV_ANON, SCHEMA)
    (run_dir / "pipeline.py").write_text(_good_pipeline())
    with patch.object(oc.writer, "PIPELINE_DIR", str(tmp_path / "pipelines")):
        path = oc.build_final_pipeline(run_dir, "hdfc", "account", "pdfplumber")
    result = Path(path).read_text()
    assert "def extract_tables" in result
    assert "def tables_to_dataframe" in result
    assert "def transform" in result
    assert "return tables_to_dataframe(extract_tables" in result
    assert "extract_one('pdfplumber'" in result


def test_build_final_pipeline_uses_winner_extractor(tmp_path):
    for winner in oc._VALID_WINNERS:
        run_dir = tmp_path / winner
        (run_dir / "fixtures").mkdir(parents=True)
        oc.prepare_sandbox(run_dir, DEV_ANON, SCHEMA)
        (run_dir / "pipeline.py").write_text(_good_pipeline())
        with patch.object(oc.writer, "PIPELINE_DIR", str(tmp_path / f"out_{winner}")):
            path = oc.build_final_pipeline(run_dir, "b", "i", winner)
        assert f"extract_one({winner!r}" in Path(path).read_text()


def test_build_final_pipeline_hoists_future_import(tmp_path):
    """Regression: a generated pipeline.py starting with `from __future__` must still import."""
    run_dir = tmp_path / "run"
    (run_dir / "fixtures").mkdir(parents=True)
    oc.prepare_sandbox(run_dir, DEV_ANON, SCHEMA)
    (run_dir / "pipeline.py").write_text(
        "from __future__ import annotations\nimport polars as pl\n"
        "def tables_to_dataframe(tables):\n    return pl.DataFrame()\n"
    )
    with patch.object(oc.writer, "PIPELINE_DIR", str(tmp_path / "pipelines")):
        path = oc.build_final_pipeline(run_dir, "hdfc", "account", "pdfplumber")
    text = Path(path).read_text()
    assert text.startswith("from __future__ import annotations\n")
    assert text.count("from __future__ import") == 1
    compile(text, path, "exec")  # raises SyntaxError if __future__ is misplaced


# --- _extract_tables_stub_source -------------------------------------------


def test_extract_tables_stub_source_for_each_winner():
    for winner in oc._VALID_WINNERS:
        src = oc._extract_tables_stub_source(winner)
        assert "def extract_tables" in src
        assert f"extract_one({winner!r}" in src
        assert "attachment" in src
        assert "password" in src
        assert "import extract_one" in src


def test_extract_tables_stub_source_raises_on_unknown():
    with pytest.raises(ValueError, match="unknown extraction library"):
        oc._extract_tables_stub_source("nonexistent")


# --- run_round (one ralph-wiggum round) ------------------------------------


RAW_BY_EXT = {"pdfplumber": ANON_TABLES, "docling": ANON_TABLES}


@pytest.fixture
def _round_dir(monkeypatch, tmp_path):
    """A prepared run dir + PIPELINE_DIR redirected into tmp (broker is the caller's job)."""
    monkeypatch.setattr(oc.writer, "PIPELINE_DIR", str(tmp_path / "pipelines"))
    run_dir = tmp_path / "run"
    (run_dir / "fixtures").mkdir(parents=True)
    oc.prepare_sandbox(run_dir, DEV_ANON, SCHEMA)
    return run_dir


def test_run_round_happy(_round_dir, monkeypatch):
    def fake_run(run_dir, model, timeout):
        (run_dir / "pipeline.py").write_text(_good_pipeline())
        (run_dir / "winner.txt").write_text("pdfplumber\n")

    monkeypatch.setattr(oc, "run_opencode", fake_run)
    winner, path = oc.run_round(
        _round_dir, "hdfc", "account", "openrouter/x", RAW_BY_EXT, password=PASSWORD
    )
    assert winner == "pdfplumber"
    src = Path(path).read_text()
    assert "def extract_tables" in src
    assert "def tables_to_dataframe" in src
    assert "def transform" in src
    assert "extract_one('pdfplumber'" in src


def test_run_round_requires_model(_round_dir):
    with pytest.raises(CodegenError):
        oc.run_round(_round_dir, "hdfc", "account", "", RAW_BY_EXT)


def test_run_round_missing_pipeline(_round_dir, monkeypatch):
    monkeypatch.setattr(oc, "run_opencode", lambda run_dir, model, timeout: None)
    with pytest.raises(CodegenError):
        oc.run_round(_round_dir, "hdfc", "account", "openrouter/x", RAW_BY_EXT)


def test_run_round_invalid_winner_raises(_round_dir, monkeypatch):
    def fake_run(run_dir, model, timeout):
        (run_dir / "pipeline.py").write_text(_good_pipeline())
        (run_dir / "winner.txt").write_text("not_a_real_extractor\n")

    monkeypatch.setattr(oc, "run_opencode", fake_run)
    with pytest.raises(CodegenError):
        oc.run_round(_round_dir, "hdfc", "account", "openrouter/x", RAW_BY_EXT)


def test_run_round_missing_winner_raises(_round_dir, monkeypatch):
    def fake_run(run_dir, model, timeout):
        (run_dir / "pipeline.py").write_text(_good_pipeline())  # no winner.txt

    monkeypatch.setattr(oc, "run_opencode", fake_run)
    with pytest.raises(CodegenError):
        oc.run_round(_round_dir, "hdfc", "account", "openrouter/x", RAW_BY_EXT)


def test_run_round_writes_and_clears_feedback(_round_dir, monkeypatch):
    seen: dict[str, str | None] = {}

    def fake_run(run_dir, model, timeout):
        fb = run_dir / "host_feedback.txt"
        seen["fb"] = fb.read_text() if fb.exists() else None
        (run_dir / "pipeline.py").write_text(_good_pipeline())
        (run_dir / "winner.txt").write_text("pdfplumber\n")

    monkeypatch.setattr(oc, "run_opencode", fake_run)
    oc.run_round(
        _round_dir, "hdfc", "account", "openrouter/x", RAW_BY_EXT,
        feedback="error_type=ValueError missing column",
    )
    assert seen["fb"] == "error_type=ValueError missing column"  # written before opencode runs
    # A subsequent round without feedback must clear the stale file.
    oc.run_round(_round_dir, "hdfc", "account", "openrouter/x", RAW_BY_EXT)
    assert seen["fb"] is None


def test_run_round_pii_leak_blocks_write(_round_dir, monkeypatch):
    raw = {"pdfplumber": [ExtractedTable(name="t", rows=[["h"], [RAW_SECRET]], page=1)]}
    leaky = _good_pipeline() + f"\n# leaked: {RAW_SECRET}\n"

    def fake_run(run_dir, model, timeout):
        (run_dir / "pipeline.py").write_text(leaky)
        (run_dir / "winner.txt").write_text("pdfplumber\n")

    monkeypatch.setattr(oc, "run_opencode", fake_run)
    with pytest.raises(PiiLeakError):
        oc.run_round(_round_dir, "hdfc", "account", "openrouter/x", raw)


# --- broker: write_anon_fixtures -------------------------------------------


def test_write_anon_fixtures_metrics_and_files(tmp_path):
    metrics = broker.write_anon_fixtures(tmp_path, DEV_ANON)
    assert metrics["attachments"] == ["dev1"]
    assert metrics["extractors"] == ["pdfplumber", "docling"]
    assert metrics["table_counts"] == {"pdfplumber/dev1": 1, "docling/dev1": 1}
    assert (tmp_path / "fixtures" / "anon_pdfplumber_dev1.json").exists()
    assert (tmp_path / "fixtures" / "anon_docling_dev1.json").exists()
    assert json.loads((tmp_path / "fixtures" / "anon_index.json").read_text()) == {
        "extractors": ["pdfplumber", "docling"],
        "attachments": ["dev1"],
    }


def test_write_anon_fixtures_subset(tmp_path):
    dev = {"a": {"pdfplumber": ANON_TABLES}, "b": {"pdfplumber": ANON_TABLES}}
    metrics = broker.write_anon_fixtures(tmp_path, dev, attachment_ids=["b"])
    assert metrics["attachments"] == ["b"]
    assert (tmp_path / "fixtures" / "anon_pdfplumber_b.json").exists()
    assert not (tmp_path / "fixtures" / "anon_pdfplumber_a.json").exists()


# --- broker: socket protocol (short /tmp run dir keeps path under the limit) -


def _broker_request(sock_path: str, payload: dict) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall((json.dumps(payload) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf)


@pytest.fixture
def _short_run_dir():
    d = Path(tempfile.mkdtemp(dir="/tmp"))
    (d / "fixtures").mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_broker_describe_and_regenerate(_short_run_dir):
    b = broker.ToolBroker(_short_run_dir, DEV_ANON)
    b.start()
    try:
        token = json.loads((_short_run_dir / broker.TOKEN_FILE).read_text())["token"]
        desc = _broker_request(b.socket_path, {"action": "describe_dev_set", "token": token})
        assert desc["ok"] is True
        assert desc["data"]["attachments"] == ["dev1"]
        assert desc["data"]["extractors"] == ["pdfplumber", "docling"]

        regen = _broker_request(
            b.socket_path,
            {"action": "regenerate_anonymized_tables", "token": token, "args": {}},
        )
        assert regen["ok"] is True
        assert (_short_run_dir / "fixtures" / "anon_pdfplumber_dev1.json").exists()
    finally:
        b.stop()
    assert not Path(b.socket_path).exists()  # cleaned up on stop


def test_broker_rejects_bad_token(_short_run_dir):
    b = broker.ToolBroker(_short_run_dir, DEV_ANON)
    b.start()
    try:
        resp = _broker_request(b.socket_path, {"action": "describe_dev_set", "token": "wrong"})
        assert resp["ok"] is False
        assert "unauthorized" in resp["error"]
    finally:
        b.stop()


def test_broker_response_contains_no_data_rows(_short_run_dir):
    # A raw secret placed in a DATA row must never appear in broker responses (headers + counts only).
    leaky = {
        "dev1": {
            "pdfplumber": [
                ExtractedTable(name=None, rows=[["Txn Date", "Amount"], ["x", RAW_SECRET]], page=1)
            ]
        }
    }
    b = broker.ToolBroker(_short_run_dir, leaky)
    b.start()
    try:
        token = json.loads((_short_run_dir / broker.TOKEN_FILE).read_text())["token"]
        desc = _broker_request(b.socket_path, {"action": "describe_dev_set", "token": token})
        regen = _broker_request(
            b.socket_path, {"action": "regenerate_anonymized_tables", "token": token, "args": {}}
        )
        assert RAW_SECRET not in json.dumps(desc)
        assert RAW_SECRET not in json.dumps(regen)
    finally:
        b.stop()


def test_broker_socket_path_too_long_raises(monkeypatch, tmp_path):
    # A deeply nested run dir would exceed the AF_UNIX sun_path limit — fail loudly, not silently.
    monkeypatch.setattr(broker, "_MAX_SOCKET_PATH", 10)
    b = broker.ToolBroker(tmp_path / "run", DEV_ANON)
    with pytest.raises(CodegenError, match="socket path too long"):
        b.start()


# --- real bwrap confinement (integration) ----------------------------------


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
def test_bwrap_confinement_blocks_repo_access(tmp_path):
    """A command run under the bwrap prefix cannot read a repo file outside the sandbox."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = str(Path("pyproject.toml").resolve())
    argv = oc._bwrap_prefix(run_dir) + ["cat", target]
    import subprocess

    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0, "sandbox should not be able to read repo files"
    assert "[tool" not in proc.stdout
