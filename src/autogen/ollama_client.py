"""Thin HTTP client over the Ollama REST API.

Uses httpx for all HTTP calls. Does NOT use the `ollama` Python package.
"""

import httpx

from src import config as cfg
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

OLLAMA_BASE_URL = cfg.ollama.get("base_url", "http://localhost:11434")


def is_reachable(timeout: float = 5.0) -> bool:
    """Return True if the Ollama server responds 200 to GET /api/tags."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
        logger.debug("Ollama not reachable: %s", exc)
        return False


def list_models(timeout: float = 5.0) -> list[str]:
    """Return the list of model names available in Ollama.

    Names are returned as-is from the API (e.g. "gemma3:12b", "glm-ocr:latest").
    Raises httpx exceptions on connection / HTTP errors.
    """
    resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"Ollama list_models returned {resp.status_code}: {resp.text}"
        )
    return [m["name"] for m in resp.json()["models"]]


def ensure_models(required: list[str]) -> None:
    """Assert that all required models are available in Ollama.

    A required name matches a listed name when:
    - it is an exact match, OR
    - it is a prefix of a listed name up to the ":" separator
      (e.g. "glm-ocr" matches "glm-ocr:latest").

    Raises RuntimeError listing all missing models.
    """
    if not required:
        return

    available = list_models()

    def _matches(req: str) -> bool:
        for listed in available:
            if listed == req:
                return True
            tag_stripped = listed.split(":")[0]
            if tag_stripped == req:
                return True
        return False

    missing = [r for r in required if not _matches(r)]
    if missing:
        raise RuntimeError(
            f"Required Ollama models not available: {', '.join(missing)}. "
            f"Run `ollama pull <model>` to download them."
        )


def generate(
    model: str,
    prompt: str,
    images: list[str] | None = None,
    format: str | None = None,
    options: dict | None = None,
    timeout: float = 300.0,
) -> str:
    """Send a generation request to Ollama and return the response text.

    Args:
        model: Name of the model to use (e.g. "gemma3:12b").
        prompt: The text prompt.
        images: Optional list of base64-encoded images (for multimodal models).
        format: Optional response format hint (e.g. "json").
        options: Optional model parameters (e.g. {"temperature": 0.2}).
        timeout: HTTP timeout in seconds.

    Returns:
        The generated text from response["response"].

    Raises:
        RuntimeError: On non-2xx HTTP status.
    """
    body: dict = {"model": model, "prompt": prompt, "stream": False}
    if images is not None:
        body["images"] = images
    if format is not None:
        body["format"] = format
    if options is not None:
        body["options"] = options

    resp = httpx.post(f"{OLLAMA_BASE_URL}/api/generate", json=body, timeout=timeout)

    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"Ollama generate returned {resp.status_code}: {resp.text}"
        )

    return resp.json()["response"]


def chat(
    model: str,
    messages: list[dict],
    options: dict | None = None,
    timeout: float = 300.0,
) -> str:
    """Send a chat request to Ollama and return the assistant's text.

    Unlike `generate`, this uses the /api/chat endpoint and a message list,
    enabling assistant-message prefill (end `messages` with a partial assistant
    turn to force the model to continue from it).

    Args:
        model: Name of the model to use (e.g. "gemma4:12b").
        messages: List of {"role", "content"} dicts.
        options: Optional model parameters (e.g. {"temperature": 0.2, "num_ctx": 8192}).
        timeout: HTTP timeout in seconds.

    Returns:
        The generated text from response["message"]["content"].

    Raises:
        RuntimeError: On non-2xx HTTP status.
    """
    body: dict = {"model": model, "messages": messages, "stream": False}
    if options is not None:
        body["options"] = options

    resp = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=body, timeout=timeout)

    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"Ollama chat returned {resp.status_code}: {resp.text}"
        )

    return resp.json()["message"]["content"]


def offload(model: str, timeout: float = 30.0) -> None:
    """Evict a model from VRAM by sending a keep_alive=0 generate request.

    This is best-effort: any failure is logged as a warning and suppressed.
    Purpose: free GPU memory before loading a different model.
    """
    try:
        httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("Failed to offload model %r from VRAM: %s", model, exc)

if __name__ == "__main__":
    print(generate(
        model="codegemma:7b",
        prompt="write a hello world function in python",
    ))
