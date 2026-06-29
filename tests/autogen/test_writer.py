"""Tests for src.autogen.codegen.writer (TDD)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.autogen.codegen import writer
from src.autogen.exceptions import CodegenError, PiiLeakError

VALID_CODE = "def transform(attachment, password):\n    pass\n"
FENCED_CODE = f"```python\n{VALID_CODE}```"
FENCED_NO_LANG = f"```\n{VALID_CODE}```"
INVALID_CODE = "def transform(\n    pass\n"


# --- strip_code_fences ---

def test_strip_code_fences_python_fence():
    assert writer.strip_code_fences(FENCED_CODE) == VALID_CODE.strip()


def test_strip_code_fences_no_lang_fence():
    assert writer.strip_code_fences(FENCED_NO_LANG) == VALID_CODE.strip()


def test_strip_code_fences_plain():
    assert writer.strip_code_fences(VALID_CODE) == VALID_CODE.strip()


def test_strip_code_fences_extra_whitespace():
    padded = f"  {VALID_CODE}  "
    assert writer.strip_code_fences(padded) == VALID_CODE.strip()


# --- _call_local ---

def _models_response(slugs: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": [{"id": s} for s in slugs]}
    return resp

def test_call_local_generates():
    mock_oc = MagicMock()
    # Mock ollama_client.chat to return code without fence prefix (normal Ollama behavior)
    mock_oc.chat.return_value = VALID_CODE

    with patch.object(writer, "ollama_client", mock_oc):
        result = writer._call_local("some prompt")

    # Verify chat was called with model, messages list (user + assistant prefill), and options
    mock_oc.chat.assert_called_once_with(
        writer.LOCAL_MODEL,
        messages=[
            {"role": "user", "content": "some prompt"},
            {"role": "assistant", "content": writer.PREFILL},
        ],
        options={"temperature": 0.2, "num_ctx": writer.LOCAL_NUM_CTX},
    )
    # Since VALID_CODE doesn't start with ```, _call_local prepends PREFILL
    assert result == writer.PREFILL + VALID_CODE


# --- generate_and_write: local happy path ---

def test_generate_and_write_local_writes_file(tmp_path):
    with patch.object(writer, "PIPELINE_DIR", str(tmp_path)):
        with patch.object(writer, "_call_local", return_value=VALID_CODE):
            path = writer.generate_and_write("bankx", "card", "prompt", backend="local")

    out = Path(path)
    assert out.exists()
    assert out.name == "bankx_card.py"
    assert "def transform" in out.read_text()


# --- fence stripping integration ---

def test_generate_and_write_strips_fence(tmp_path):
    with patch.object(writer, "PIPELINE_DIR", str(tmp_path)):
        with patch.object(writer, "_call_local", return_value=FENCED_CODE):
            path = writer.generate_and_write("bankx", "card", "prompt", backend="local")

    assert "def transform" in Path(path).read_text()


# --- retry: 1 bad then 1 good ---

def test_generate_and_write_retry_succeeds(tmp_path):
    responses = iter([INVALID_CODE, VALID_CODE])

    with patch.object(writer, "PIPELINE_DIR", str(tmp_path)):
        with patch.object(writer, "_call_local", side_effect=lambda p: next(responses)):
            path = writer.generate_and_write("bankx", "card", "prompt", backend="local")

    assert Path(path).exists()


# --- 3 invalid → CodegenError ---

def test_generate_and_write_all_invalid_raises(tmp_path):
    with patch.object(writer, "PIPELINE_DIR", str(tmp_path)):
        with patch.object(writer, "_call_local", return_value=INVALID_CODE):
            with pytest.raises(CodegenError):
                writer.generate_and_write("bankx", "card", "prompt", backend="local")


# --- PII leak ---

def test_generate_and_write_pii_leak_raises(tmp_path):
    secret = "secret123abc"
    pii_code = f"def transform(attachment, password):\n    x = '{secret}'\n"
    with patch.object(writer, "PIPELINE_DIR", str(tmp_path)):
        with patch.object(writer, "_call_local", return_value=pii_code):
            with pytest.raises(PiiLeakError):
                writer.generate_and_write("bankx", "card", "prompt", backend="local", password=secret)


# --- cloud happy path ---

def _make_httpx_mock(api_key: str, valid_code: str) -> MagicMock:
    """Return a mock for httpx with GET models and POST completions wired."""
    get_resp = _models_response(["nvidia/nemotron-ultra"])
    post_resp = MagicMock()
    post_resp.status_code = 200
    post_resp.json.return_value = {
        "choices": [{"message": {"content": valid_code}}]
    }

    def fake_get(url, **kw):
        return get_resp

    def fake_post(url, **kw):
        # record headers for assertion
        fake_post.captured_kwargs = kw
        return post_resp

    fake_post.captured_kwargs = {}

    mock_httpx = MagicMock()
    mock_httpx.get.side_effect = fake_get
    mock_httpx.post.side_effect = fake_post
    return mock_httpx


def test_generate_and_write_cloud_writes_file(tmp_path):
    api_key = "test-key-xyz"
    mock_httpx = _make_httpx_mock(api_key, VALID_CODE)

    with patch.object(writer, "PIPELINE_DIR", str(tmp_path)):
        with patch("src.autogen.codegen.writer.httpx", mock_httpx):
            path = writer.generate_and_write(
                "bankx", "card", "prompt", backend="cloud", openrouter_api_key=api_key
            )

    out = Path(path)
    assert out.exists()
    content = out.read_text()
    assert "def transform" in content
    # api_key must not leak into the written file
    assert api_key not in content

    # Authorization header must use Bearer
    headers = mock_httpx.post.call_args.kwargs.get("headers", {})
    assert headers.get("Authorization") == f"Bearer {api_key}"


# --- cloud without api_key → CodegenError ---

def test_generate_and_write_cloud_no_key_raises(tmp_path):
    with patch.object(writer, "PIPELINE_DIR", str(tmp_path)):
        with pytest.raises(CodegenError, match="api_key"):
            writer.generate_and_write("bankx", "card", "prompt", backend="cloud")
