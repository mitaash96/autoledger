"""Tests for src.autogen.ollama_client — all HTTP mocked, no real network calls."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.autogen import ollama_client as oc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, body: dict | str) -> MagicMock:
    """Build a MagicMock that behaves like an httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if isinstance(body, dict):
        resp.json.return_value = body
        resp.text = json.dumps(body)
    else:
        resp.json.side_effect = ValueError("not json")
        resp.text = body
    return resp


# ---------------------------------------------------------------------------
# is_reachable
# ---------------------------------------------------------------------------


class TestIsReachable:
    def test_returns_true_on_200(self):
        resp = _mock_response(200, {"models": []})
        with patch("httpx.get", return_value=resp) as mock_get:
            assert oc.is_reachable() is True
            mock_get.assert_called_once()
            url = mock_get.call_args[0][0]
            assert url.endswith("/api/tags")

    def test_returns_false_on_connection_error(self):
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert oc.is_reachable() is False

    def test_returns_false_on_timeout(self):
        import httpx

        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            assert oc.is_reachable() is False

    def test_returns_false_on_non_200(self):
        resp = _mock_response(503, "Service Unavailable")
        with patch("httpx.get", return_value=resp):
            assert oc.is_reachable() is False

    def test_passes_timeout(self):
        resp = _mock_response(200, {"models": []})
        with patch("httpx.get", return_value=resp) as mock_get:
            oc.is_reachable(timeout=2.5)
            _, kwargs = mock_get.call_args
            assert kwargs.get("timeout") == 2.5


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_parses_names(self):
        body = {
            "models": [
                {"name": "gemma3:12b", "size": 1},
                {"name": "glm-ocr:latest", "size": 2},
            ]
        }
        with patch("httpx.get", return_value=_mock_response(200, body)):
            names = oc.list_models()
        assert names == ["gemma3:12b", "glm-ocr:latest"]

    def test_empty_models_list(self):
        body = {"models": []}
        with patch("httpx.get", return_value=_mock_response(200, body)):
            names = oc.list_models()
        assert names == []

    def test_raises_runtime_error_on_5xx(self):
        with patch("httpx.get", return_value=_mock_response(500, "Server Error")):
            with pytest.raises(RuntimeError, match="500"):
                oc.list_models()

    def test_raises_runtime_error_on_4xx(self):
        with patch("httpx.get", return_value=_mock_response(403, "Forbidden")):
            with pytest.raises(RuntimeError, match="403"):
                oc.list_models()

    def test_raises_on_connection_error(self):
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(httpx.ConnectError):
                oc.list_models()


# ---------------------------------------------------------------------------
# ensure_models
# ---------------------------------------------------------------------------


class TestEnsureModels:
    def _patch_list(self, names: list[str]):
        return patch.object(oc, "list_models", return_value=names)

    def test_exact_match_passes(self):
        with self._patch_list(["gemma3:12b", "glm-ocr:latest"]):
            # should not raise
            oc.ensure_models(["gemma3:12b"])

    def test_prefix_match_passes(self):
        """'glm-ocr' should match 'glm-ocr:latest' (prefix before ':')."""
        with self._patch_list(["glm-ocr:latest"]):
            oc.ensure_models(["glm-ocr"])

    def test_prefix_match_multiple_tags(self):
        with self._patch_list(["gemma3:4b", "gemma3:12b"]):
            # "gemma3" is a prefix of both; as long as one matches, OK
            oc.ensure_models(["gemma3"])

    def test_missing_model_raises_runtime_error(self):
        with self._patch_list(["gemma3:12b"]):
            with pytest.raises(RuntimeError, match="glm-ocr"):
                oc.ensure_models(["glm-ocr"])

    def test_partial_missing_includes_all_missing_in_error(self):
        with self._patch_list(["gemma3:12b"]):
            with pytest.raises(RuntimeError) as exc_info:
                oc.ensure_models(["gemma3:12b", "llama3", "glm-ocr"])
            msg = str(exc_info.value)
            assert "llama3" in msg
            assert "glm-ocr" in msg

    def test_empty_required_passes(self):
        with self._patch_list(["gemma3:12b"]):
            oc.ensure_models([])


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_response_text(self):
        body = {"response": "Here is the code."}
        with patch("httpx.post", return_value=_mock_response(200, body)) as mock_post:
            result = oc.generate(model="gemma3:12b", prompt="Write code")
        assert result == "Here is the code."

    def test_builds_correct_body(self):
        body = {"response": "ok"}
        with patch("httpx.post", return_value=_mock_response(200, body)) as mock_post:
            oc.generate(model="gemma3:12b", prompt="Hello")
        _, kwargs = mock_post.call_args
        sent = kwargs.get("json") or mock_post.call_args[1].get("json")
        assert sent["model"] == "gemma3:12b"
        assert sent["prompt"] == "Hello"
        assert sent["stream"] is False

    def test_includes_images_when_provided(self):
        body = {"response": "result"}
        with patch("httpx.post", return_value=_mock_response(200, body)) as mock_post:
            oc.generate(model="glm-ocr:latest", prompt="Describe", images=["base64abc"])
        _, kwargs = mock_post.call_args
        sent = kwargs.get("json")
        assert "images" in sent
        assert sent["images"] == ["base64abc"]

    def test_includes_format_when_provided(self):
        body = {"response": "{}"}
        with patch("httpx.post", return_value=_mock_response(200, body)) as mock_post:
            oc.generate(model="gemma3:12b", prompt="JSON please", format="json")
        _, kwargs = mock_post.call_args
        sent = kwargs.get("json")
        assert sent.get("format") == "json"

    def test_includes_options_when_provided(self):
        body = {"response": "ok"}
        with patch("httpx.post", return_value=_mock_response(200, body)) as mock_post:
            oc.generate(model="gemma3:12b", prompt="p", options={"temperature": 0.2})
        _, kwargs = mock_post.call_args
        sent = kwargs.get("json")
        assert sent.get("options") == {"temperature": 0.2}

    def test_omits_optional_fields_when_none(self):
        body = {"response": "ok"}
        with patch("httpx.post", return_value=_mock_response(200, body)) as mock_post:
            oc.generate(model="gemma3:12b", prompt="p")
        _, kwargs = mock_post.call_args
        sent = kwargs.get("json")
        assert "images" not in sent
        assert "format" not in sent
        assert "options" not in sent

    def test_raises_on_5xx(self):
        err_resp = _mock_response(500, "Internal Server Error")
        with patch("httpx.post", return_value=err_resp):
            with pytest.raises(RuntimeError, match="500"):
                oc.generate(model="gemma3:12b", prompt="bad")

    def test_raises_on_4xx(self):
        err_resp = _mock_response(404, "Not Found")
        with patch("httpx.post", return_value=err_resp):
            with pytest.raises(RuntimeError, match="404"):
                oc.generate(model="gemma3:12b", prompt="bad")

    def test_url_points_to_generate_endpoint(self):
        body = {"response": "ok"}
        with patch("httpx.post", return_value=_mock_response(200, body)) as mock_post:
            oc.generate(model="gemma3:12b", prompt="p")
        url = mock_post.call_args[0][0]
        assert url.endswith("/api/generate")

    def test_passes_timeout(self):
        body = {"response": "ok"}
        with patch("httpx.post", return_value=_mock_response(200, body)) as mock_post:
            oc.generate(model="gemma3:12b", prompt="p", timeout=60.0)
        _, kwargs = mock_post.call_args
        assert kwargs.get("timeout") == 60.0


# ---------------------------------------------------------------------------
# offload
# ---------------------------------------------------------------------------


class TestOffload:
    def test_posts_keep_alive_zero(self):
        resp = _mock_response(200, {"response": ""})
        with patch("httpx.post", return_value=resp) as mock_post:
            oc.offload(model="gemma3:12b")
        _, kwargs = mock_post.call_args
        sent = kwargs.get("json")
        assert sent["model"] == "gemma3:12b"
        assert sent["keep_alive"] == 0
        assert sent["stream"] is False

    def test_does_not_raise_on_connection_error(self):
        import httpx

        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            # must NOT raise
            oc.offload(model="gemma3:12b")

    def test_does_not_raise_on_server_error(self):
        err_resp = _mock_response(500, "oops")
        with patch("httpx.post", return_value=err_resp):
            oc.offload(model="gemma3:12b")

    def test_does_not_raise_on_timeout(self):
        import httpx

        with patch("httpx.post", side_effect=httpx.TimeoutException("slow")):
            oc.offload(model="gemma3:12b")

    def test_prompt_is_empty_string(self):
        resp = _mock_response(200, {"response": ""})
        with patch("httpx.post", return_value=resp) as mock_post:
            oc.offload(model="gemma3:12b")
        _, kwargs = mock_post.call_args
        sent = kwargs.get("json")
        assert sent["prompt"] == ""
