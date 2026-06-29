"""Sandboxed opencode headless codegen backend.

Runs `opencode run` inside a bubblewrap (`bwrap`) sandbox confined to a per-run directory
under `src/autogen/sandbox/`. The sandbox is populated with ONLY anonymized data, so the
autonomous agent can never read real PDFs, the password, or `.env`: those paths are simply
not bind-mounted into it, and `--clearenv` strips the parent environment. opencode iterates
on an in-sandbox harness (`test_logic.py`) that exercises the pure transform logic against
anonymized table fixtures — a strong, PII-free feedback signal.

Split contract the generated `pipeline.py` must satisfy:
    extract_tables(attachment, password) -> list[dict]   # real PDF parsing (tested outside)
    tables_to_dataframe(tables: list[dict]) -> pl.DataFrame  # pure logic (tested in-sandbox)
    transform(attachment, password) -> pl.DataFrame      # = tables_to_dataframe(extract_tables(...))
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from src import config as cfg
from src.autogen.codegen import pfc_tool_broker, pii_guard, writer
from src.autogen.codegen.pfc_tool_broker import ToolBroker  # re-export for generator + tests
from src.autogen.exceptions import CodegenError
from src.autogen.models import ExtractedTable
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

WORKDIR = "/work"  # mount point of the per-run sandbox dir, inside bwrap
SANDBOX_HOME = "/work/.home"
_VALID_WINNERS = ("docling", "camelot", "pymupdf", "pdfplumber")


def _extract_tables_stub_source(winner: str) -> str:
    """Source for a thin extract_tables that reuses the project's scored extractor.

    The generated pipeline delegates PDF parsing to ``extract_one(winner, ...)`` so it can
    never drift from the extractor chosen during scoring, and inherits its decryption and
    cell normalization. The agent reads this in-sandbox only to learn the input table shape;
    it is never executed inside the sandbox. Raises ValueError for an unknown winner.
    """
    if winner not in _VALID_WINNERS:
        raise ValueError(f"unknown extraction library: {winner!r}")
    return (
        "from src.autogen.extraction.runner import extract_one\n"
        "from src.autogen.models import table_to_dict\n"
        "\n"
        "\n"
        "def extract_tables(attachment, password):\n"
        '    """Parse the PDF with the scored extractor.\n'
        "\n"
        "    Returns a list of tables, each a dict shaped\n"
        '    ``{"name": str | None, "page": int | None, "rows": list[list[str]]}``\n'
        "    where rows[0] is the header. This is exactly the shape of the table dicts in\n"
        "    fixtures/anon_*.json that tables_to_dataframe receives.\n"
        '    """\n'
        f"    tables = extract_one({winner!r}, attachment.physical_file, password)\n"
        "    return [table_to_dict(t) for t in tables]\n"
    )


RUN_MESSAGE = (
    "Follow AGENTS.md exactly. CHECKPOINT 1: compare every extractor's anonymized fixtures "
    "against the target schema and write the chosen extractor's bare name to `winner.txt`. "
    "CHECKPOINT 2: implement tables_to_dataframe in pipeline.py for that extractor, then run "
    "`python test_logic.py` and iterate until it prints a line starting with OK. If "
    "`host_feedback.txt` exists, read it first — it reports how a previous attempt failed on the "
    "real held-out test set. Use the `pfc_samples` tool to inspect the anonymized fixtures. Do "
    "not access anything outside this directory."
)


def make_run_dir(bank: str, instrument: str) -> Path:
    """Create and return a fresh per-run sandbox directory under the sandbox root."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.opencode["sandbox_root"]) / f"{bank}_{instrument}_{ts}"
    (run_dir / "fixtures").mkdir(parents=True, exist_ok=True)
    (run_dir / ".home").mkdir(parents=True, exist_ok=True)
    return run_dir


def _test_logic_src() -> str:
    """In-sandbox harness: validate tables_to_dataframe over the CHOSEN extractor's fixtures.

    Reads the extractor name from winner.txt (CHECKPOINT 1) and runs tables_to_dataframe against
    that extractor's anonymized fixtures for every attachment in fixtures/anon_index.json. Each
    must yield a non-empty, schema-conforming DataFrame — a 0-row frame is a failure, closing the
    empty-frame false-pass. DTYPE_OK mirrors src.autogen.runner.DTYPE_OK; it is inlined because the
    harness runs standalone inside the sandbox.
    """
    return """import json
from pathlib import Path

import polars as pl

import pipeline

DTYPE_OK = {
    "datetime": (pl.Datetime, pl.Date),
    "float": (pl.Float32, pl.Float64),
    "str": (pl.String,),
}

schema = json.loads(Path("target_schema.json").read_text())
index = json.loads(Path("fixtures/anon_index.json").read_text())
selectable = json.loads(Path("selectable.json").read_text())
attachments = index["attachments"]

winner_path = Path("winner.txt")
if not winner_path.exists():
    print("FAIL winner.txt missing — complete CHECKPOINT 1 (choose an extractor) first")
    raise SystemExit(1)
winner = winner_path.read_text().strip()
if winner not in selectable:
    print(f"FAIL winner.txt is {winner!r}; choose one of {selectable}")
    raise SystemExit(1)

failures = []
for aid in attachments:
    tables = json.loads(Path(f"fixtures/anon_{winner}_{aid}.json").read_text())
    try:
        df = pipeline.tables_to_dataframe(tables)
    except Exception as exc:
        failures.append(f"{aid}: raised {type(exc).__name__}: {exc}")
        continue
    if not isinstance(df, pl.DataFrame):
        failures.append(f"{aid}: tables_to_dataframe did not return a polars DataFrame")
        continue
    missing = [c for c in schema if c not in df.columns]
    if missing:
        failures.append(f"{aid}: missing columns: {missing}")
        continue
    bad = [c for c, k in schema.items() if not isinstance(df[c].dtype, DTYPE_OK[k])]
    if bad:
        failures.append(f"{aid}: dtype mismatch on {bad}")
        continue
    if df.height == 0:
        failures.append(f"{aid}: produced an empty DataFrame (0 rows)")
        continue

if failures:
    for f in failures:
        print("FAIL", f)
    raise SystemExit(1)

print("OK", winner, len(attachments), "dev attachment(s) pass")
"""


def _agents_md(target_schema: dict[str, str]) -> str:
    """AGENTS.md task spec — two checkpoints: pick the winning extractor, then generate code."""
    spec = f"""# Codegen task: pick an extractor, then write tables_to_dataframe

The SAME bank statements were parsed by several different PDF table extractors. Your job has two
checkpoints: first choose the extractor whose output maps most cleanly onto the target schema,
then write the transform for it.

Anonymized fixtures live in `fixtures/`. The file name maps each input to the extractor that
produced it:

    fixtures/anon_<extractor>_<attachment_id>.json   ->   output of extractor "<extractor>"

`fixtures/anon_index.json` lists the available `extractors` and `attachments`. Only the cell
values are anonymized — table shapes, headers, row counts and quirks are real. `selectable.json`
lists the extractors that can drive the final pipeline; choose your winner from those. Any
extractor present but not in `selectable.json` is reference-only (do not pick it).

Target schema (column -> polars dtype): {json.dumps(target_schema)}

You have a custom tool `pfc_samples` (use it via the normal tool-call mechanism):
- `describe_dev_set` — for every extractor and attachment, each table's column headers and row counts.
- `regenerate_anonymized_tables` (optional `attachment_ids`) — rewrite the fixture files.
It returns anonymized/structural data only.

## CHECKPOINT 1 — select the winning extractor
For each selectable extractor, read its fixtures across the attachments and judge how cleanly its
tables map onto the target schema:
- columns line up with the schema (after discarding address/metadata/promo tables),
- the header is a real header row, NOT duplicated into the first data row (some extractors, e.g.
  camelot, push the header into row 0),
- rows are not mangled by merged or misaligned cells.
Write the chosen extractor's BARE name (e.g. `docling`) to `winner.txt` — one line, nothing else.
It MUST be one of the names in `selectable.json`.

## CHECKPOINT 2 — write tables_to_dataframe for that extractor
Write `pipeline.py` defining EXACTLY one function:

`tables_to_dataframe(tables: list[dict]) -> pl.DataFrame`

Pure function: given the chosen extractor's tables in the shape
`{{"name": str|None, "page": int|None, "rows": list[list[str]]}}` (rows[0] is the header), return a
polars DataFrame with EXACTLY the target schema above. `test_logic.py` reads `winner.txt` and runs
your function against that extractor's fixtures for every attachment; each must yield a non-empty,
schema-conforming DataFrame. The host then copies this same extractor's `extract_tables` function
into the final pipeline, so your transform must match the extractor you named in `winner.txt`.

Rules:
- READ-ONLY env, fixed deps: only stdlib and `polars` (as `pl`). Do NOT install packages.
- Discard irrelevant tables; detect the transaction table by the columns it contains, NOT by exact
  header equality (headers vary across statements).
- Cells may contain embedded newlines (wrapped text, or several records merged into one cell). Do
  NOT split every cell independently — that misaligns rows. Anchor the row count to the column with
  exactly one value per record (date, else amount): split it into N sub-rows, split other columns
  and zip positionally when they have N segments, and broadcast a single-segment cell across all N.
  Never emit more rows than the anchor column implies.
- No `if __name__ == "__main__"` block, no hardcoded file paths, no hardcoded data values, account
  numbers, names, or example records (in code, comments, or docstrings).

Workflow — follow these steps EXACTLY:
Step 1. Read `fixtures/anon_index.json`, `selectable.json`, `target_schema.json`, and a couple of
        `fixtures/anon_<extractor>_<id>.json` for different extractors. Call `pfc_samples`
        `describe_dev_set` for a structural overview across all extractors.
Step 2. CHECKPOINT 1: choose the best extractor and write its name to `winner.txt`.
Step 3. CHECKPOINT 2: implement `tables_to_dataframe` in `pipeline.py`.
Step 4. Run `python test_logic.py`.
Step 5. If it prints a line starting with `OK` — you are done.
Step 6. If it fails — read the FAIL line(s), fix `pipeline.py` (or reconsider `winner.txt`), go to
        Step 4. If `host_feedback.txt` exists, address it first: it reports how a previous attempt
        failed on the real held-out test set.
Step 7. You are NOT done until `test_logic.py` prints OK. If you have attempted Step 4-6 five (5)
        times without success, stop and exit.
"""
    return spec


def _pfc_samples_ts() -> str:
    """Custom opencode tool source: forwards actions to the host broker over the run-dir socket."""
    return '''import { tool } from "@opencode-ai/plugin"
import { readFileSync } from "node:fs"
import { connect } from "node:net"

function call(payload) {
  const cfg = JSON.parse(readFileSync("/work/.pfc_tool_broker.json", "utf8"))
  return new Promise((resolve, reject) => {
    const sock = connect(cfg.socket)
    let buf = ""
    sock.on("connect", () =>
      sock.write(JSON.stringify({ ...payload, token: cfg.token }) + "\\n"),
    )
    sock.on("data", (d) => {
      buf += d.toString()
    })
    sock.on("end", () => {
      try {
        resolve(JSON.parse(buf))
      } catch (e) {
        reject(e)
      }
    })
    sock.on("error", reject)
  })
}

export default tool({
  description:
    "Access anonymized development-set tables for this bank statement. " +
    "action 'describe_dev_set' lists dev attachment ids with each table's column headers and " +
    "row counts; action 'regenerate_anonymized_tables' rewrites fixtures/anon_<id>.json for the " +
    "given attachment_ids (or all dev when omitted). Returns anonymized/structural data only.",
  args: {
    action: tool.schema
      .enum(["describe_dev_set", "regenerate_anonymized_tables"])
      .describe("which broker action to run"),
    attachment_ids: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe("optional subset of dev attachment ids"),
  },
  async execute(args) {
    const payload = { action: args.action, args: {} }
    if (args.attachment_ids) payload.args.attachment_ids = args.attachment_ids
    const res = await call(payload)
    return JSON.stringify(res, null, 2)
  },
})
'''


def _opencode_json() -> str:
    """Project-level opencode config: allow edit/bash (already fs-confined), deny webfetch."""
    return json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
        },
        indent=2,
    )


def prepare_sandbox(
    run_dir: Path,
    dev_anon: dict[str, dict[str, list[ExtractedTable]]],
    target_schema: dict[str, str],
) -> None:
    """Populate run_dir with only PII-free assets opencode needs to select, generate, and self-test.

    ``dev_anon`` maps attachment_id -> extractor -> already-anonymized tables; all extractors are
    written as fixtures so the agent can compare them and pick a winner (CHECKPOINT 1). No
    extract_tables stub is written here — the host composes it from the agent's winner.txt choice.
    """
    pfc_tool_broker.write_anon_fixtures(run_dir, dev_anon)
    (run_dir / "target_schema.json").write_text(json.dumps(target_schema))
    (run_dir / "selectable.json").write_text(json.dumps(list(_VALID_WINNERS)))
    (run_dir / "test_logic.py").write_text(_test_logic_src())
    (run_dir / "AGENTS.md").write_text(_agents_md(target_schema))
    (run_dir / "opencode.json").write_text(_opencode_json())
    tools_dir = run_dir / ".opencode" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "pfc_samples.ts").write_text(_pfc_samples_ts())
    n_tables = sum(len(t) for per_ext in dev_anon.values() for t in per_ext.values())
    logger.info(
        "opencode sandbox prepared at %s (%d dev attachments, %d tables across extractors)",
        run_dir,
        len(dev_anon),
        n_tables,
    )


def _system_binds() -> list[str]:
    """Read-only binds for system libraries/certs, recreating usr-merge symlinks as needed."""
    args = ["--ro-bind", "/usr", "/usr", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for top in ("/lib", "/lib64", "/bin", "/sbin"):
        if os.path.islink(top):
            args += ["--symlink", os.readlink(top), top]
        elif os.path.isdir(top):
            args += ["--ro-bind", top, top]
    for etc in ("/etc/ssl", "/etc/ca-certificates", "/etc/pki", "/etc/resolv.conf", "/etc/hosts"):
        if os.path.exists(etc):
            args += ["--ro-bind", etc, etc]
    return args


def _bwrap_prefix(run_dir: Path) -> list[str]:
    """bwrap argv up to and including `--`, confining the child to run_dir.

    Binds only system libs, the opencode install, node, the project venv, and (virtually)
    the model credential file. The repo root, data/, and .env are never bound, and
    --clearenv prevents leaking the parent environment (which may hold the bank password).
    """
    oc = cfg.opencode
    venv = os.path.abspath(oc["venv_dir"])
    install_dir = oc["install_dir"]

    argv = [oc["bwrap_binary"], "--die-with-parent", "--unshare-pid", "--clearenv"]
    argv += _system_binds()
    argv += ["--ro-bind", install_dir, install_dir]
    if os.path.exists(venv):
        argv += ["--ro-bind", venv, venv]
        # Portability: if the venv interpreter resolves outside /usr (e.g. a uv-managed Python),
        # bind its install dir too so the symlink isn't dangling inside the sandbox.
        real_python = os.path.realpath(os.path.join(venv, "bin", "python"))
        if os.path.exists(real_python) and not real_python.startswith("/usr/"):
            base = os.path.dirname(os.path.dirname(real_python))
            argv += ["--ro-bind", base, base]
    # Fresh, writable HOME inside the run dir; bind only the credential files (not the global DB),
    # matching the TUI's credential set so the zen/other providers resolve headlessly.
    argv += ["--bind", str(run_dir.resolve()), WORKDIR, "--chdir", WORKDIR]
    for cred in ("auth.json", "account.json"):
        src = os.path.join(oc["auth_dir"], cred)
        if os.path.exists(src):
            argv += ["--ro-bind", src, f"{SANDBOX_HOME}/.local/share/opencode/{cred}"]
    argv += [
        "--setenv",
        "HOME",
        SANDBOX_HOME,
        "--setenv",
        "XDG_DATA_HOME",
        f"{SANDBOX_HOME}/.local/share",
        "--setenv",
        "XDG_CONFIG_HOME",
        f"{SANDBOX_HOME}/.config",
        "--setenv",
        "XDG_CACHE_HOME",
        f"{SANDBOX_HOME}/.cache",
        "--setenv",
        "PATH",
        f"{venv}/bin:/usr/bin:/bin",
        "--setenv",
        "VIRTUAL_ENV",
        venv,
        "--setenv",
        "TERM",
        "dumb",
    ]
    # Allowlist only model-provider credential vars from the parent env — never the bank password.
    for name in oc.get("env_passthrough", ()):
        val = os.environ.get(name)
        if val:
            argv += ["--setenv", name, val]
    argv.append("--")
    return argv


def build_bwrap_argv(run_dir: Path, model: str, message: str = RUN_MESSAGE) -> list[str]:
    """Full argv: bwrap confinement prefix + headless `opencode run`."""
    return _bwrap_prefix(run_dir) + [
        cfg.opencode["binary"],
        "run",
        "-m",
        model,
        "--pure",
        message,
    ]


def run_opencode(run_dir: Path, model: str, timeout: int) -> None:
    """Execute opencode headless inside the sandbox; raise CodegenError on failure/timeout."""
    argv = build_bwrap_argv(run_dir, model)
    logger.info("opencode run (model=%s) in %s", model, run_dir)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise CodegenError(f"opencode run timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise CodegenError(f"opencode run failed (exit {proc.returncode}): {proc.stderr[-2000:]}")


def _read_winner(run_dir: Path) -> str:
    """Read and validate the extractor the agent chose in CHECKPOINT 1 (winner.txt)."""
    winner_path = run_dir / "winner.txt"
    if not winner_path.exists():
        raise CodegenError(f"agent did not write winner.txt in {run_dir} (CHECKPOINT 1 incomplete)")
    winner = winner_path.read_text().strip()
    if winner not in _VALID_WINNERS:
        raise CodegenError(f"winner.txt names {winner!r}, not one of {_VALID_WINNERS}")
    return winner


def build_final_pipeline(run_dir: Path, bank: str, instrument: str, winner: str) -> str:
    """Compose the chosen extractor's extract_tables stub with the generated tables_to_dataframe.

    ``winner`` is the extractor named in winner.txt; its extract_tables stub is generated here (so
    the chosen extractor's parsing is copied verbatim into the final pipeline) and concatenated with
    the agent's pipeline.py + a transform() wrapper.
    """
    pipeline_path = run_dir / "pipeline.py"
    if not pipeline_path.exists():
        raise CodegenError(f"generated pipeline.py not found in {run_dir}")
    stub_src = _extract_tables_stub_source(winner)
    pipeline_src = pipeline_path.read_text()

    # `from __future__` must be the first statement, so hoist it above the concatenated parts.
    def _strip_future(code: str) -> str:
        return re.sub(r"^from __future__ import .*\n?", "", code, flags=re.M)

    composed = (
        "from __future__ import annotations\n\n"
        + _strip_future(stub_src)
        + "\n\n"
        + _strip_future(pipeline_src)
        + "\n\n\ndef transform(attachment, password):\n"
        + "    return tables_to_dataframe(extract_tables(attachment, password))\n"
    )

    out_dir = Path(writer.PIPELINE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bank}_{instrument}.py"
    out_path.write_text(composed)
    logger.info("composed pipeline written to %s", out_path)
    return str(out_path)


def run_round(
    run_dir: Path,
    bank: str,
    instrument: str,
    model: str,
    raw_by_extractor: dict[str, list[ExtractedTable]],
    password: str = "",
    feedback: str | None = None,
) -> tuple[str, str]:
    """Run one ralph-wiggum round of opencode in an already-prepared sandbox.

    Writes ``feedback`` (PII-safe structural notes from a prior failed host test) to
    host_feedback.txt for the agent, runs opencode (which internally does CHECKPOINT 1 + 2 and its
    own dev-fixture loop), then reads the chosen extractor from winner.txt, composes the final
    pipeline, and PII-scans it against the chosen extractor's raw tables. The broker must already be
    serving on run_dir. Returns ``(winner, pipeline_path)``. Raises CodegenError if opencode
    produced no usable pipeline or an invalid winner; PiiLeakError on PII leak.
    """
    if not model:
        raise CodegenError("a model (provider/model) is required for the opencode backend")

    fb = run_dir / "host_feedback.txt"
    if feedback:
        pii_guard.scan(feedback, password=password)  # regex backstop before it enters the prompt
        fb.write_text(feedback)
    elif fb.exists():
        fb.unlink()

    run_opencode(run_dir, model, cfg.opencode["timeout_seconds"])

    pipeline_src = run_dir / "pipeline.py"
    if not pipeline_src.exists():
        raise CodegenError(f"opencode did not produce pipeline.py in {run_dir}")
    if not writer.strip_code_fences(pipeline_src.read_text()).strip():
        raise CodegenError("opencode produced an empty pipeline.py")

    winner = _read_winner(run_dir)
    final_path = build_final_pipeline(run_dir, bank, instrument, winner)
    final_code = Path(final_path).read_text()
    pii_guard.scan(final_code, password=password, input_tables=raw_by_extractor.get(winner))
    return winner, final_path
