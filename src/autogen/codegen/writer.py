"""LLM code generator: local (Ollama) or cloud (OpenRouter), with PII scan and file write."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import httpx

from src import config as cfg
from src.autogen import ollama_client
from src.autogen.codegen import pii_guard
from src.autogen.exceptions import CodegenError
from src.autogen.models import ExtractedTable
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

LOCAL_MODEL = "gemma4:12b"
# CLOUD_MODEL = "google/gemma-4-31b-it:free"
CLOUD_MODEL = "nex-agi/nex-n2-pro:free"
LOCAL_NUM_CTX = 8192
PREFILL = "```python\n"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PIPELINE_DIR = "src/email/pipelines"
MAX_RETRIES = 2

_FENCE_RE = re.compile(r"^```(?:python)?\n(.*?)```\s*$", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Remove leading ```python / ``` fence and trailing ``` if present; else return text stripped."""
    m = _FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()


def _call_local(prompt: str) -> str:
    """Generate with LOCAL_MODEL via Ollama chat, prefilling a code fence to force code output."""
    raw = ollama_client.chat(
        LOCAL_MODEL,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": PREFILL},
        ],
        options={"temperature": 0.2, "num_ctx": LOCAL_NUM_CTX},
    )
    # Ollama does not echo the prefill in its response; restore it so strip_code_fences works.
    return raw if raw.lstrip().startswith("```") else PREFILL + raw


def _call_cloud(prompt: str, api_key: str, timeout: float = 300.0) -> str:
    """POST to OpenRouter with the nemotron-ultra model; return generated content."""
    slug = CLOUD_MODEL
    resp = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": slug,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": PREFILL},
            ],
            "temperature": 0.1,
        },
        timeout=timeout,
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        # status only: the response body can echo the api_key back.
        raise CodegenError(f"OpenRouter returned HTTP {resp.status_code}")
    return resp.json()["choices"][0]["message"]["content"]


def generate_and_write(
    bank: str,
    instrument: str,
    prompt: str,
    backend: str,
    openrouter_api_key: str | None = None,
    input_tables: list[ExtractedTable] | None = None,
    password: str = "",
) -> str:
    """Generate ETL code, PII-scan, format, write to PIPELINE_DIR, return the file path."""
    if backend == "cloud":
        if not openrouter_api_key:
            raise CodegenError("openrouter_api_key required for cloud backend")
        caller = lambda p: _call_cloud(p, openrouter_api_key)  # noqa
    else:
        caller = _call_local

    code = ""
    current_prompt = prompt
    for attempt in range(MAX_RETRIES + 1):
        raw = caller(current_prompt)
        code = strip_code_fences(raw)
        if not code.strip():
            if attempt == MAX_RETRIES:
                raise CodegenError(f"Model returned empty output after {MAX_RETRIES + 1} attempts")
            current_prompt = prompt
            logger.warning("Empty output on attempt %d, retrying", attempt + 1)
            continue
        try:
            ast.parse(code)
            break
        except SyntaxError as err:
            if attempt == MAX_RETRIES:
                raise CodegenError(f"Syntax error after {MAX_RETRIES + 1} attempts: {err}") from err
            current_prompt = (
                f"The following Python code has a syntax error. "
                f"Return only the corrected code, no explanation.\n"
                f"Error: {err}\nCode:\n{code}"
            )
            logger.warning("Syntax error on attempt %d, retrying: %s", attempt + 1, err)

    pii_guard.scan(code, password=password, input_tables=input_tables)

    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "-"],
        input=code,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        code = proc.stdout

    out_dir = Path(PIPELINE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bank}_{instrument}.py"
    out_path.write_text(code)
    logger.info("Pipeline written to %s", out_path)
    return str(out_path)


