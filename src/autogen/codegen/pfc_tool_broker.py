"""Host-side broker serving anonymized dev-table data to the sandboxed opencode agent.

The broker listens on a Unix socket inside the sandbox run dir (which is bind-mounted at
``/work`` in the sandbox), so the in-sandbox custom tool can reach it. It returns ONLY
anonymized cells and structural metrics — never raw rows, PDF paths, or the password — and
it never executes agent-written code. This keeps the anonymization invariant intact while
giving the agent on-demand access to anonymized dev data for debugging.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import threading
from pathlib import Path

from src import config as cfg
from src.autogen.exceptions import CodegenError
from src.autogen.models import ExtractedTable, table_to_dict
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

SOCKET_NAME = ".pfc.sock"
TOKEN_FILE = ".pfc_tool_broker.json"
IN_SANDBOX_SOCKET = "/work/.pfc.sock"  # socket_path as seen inside the bwrap sandbox
_MAX_SOCKET_PATH = 100  # conservative bound under the AF_UNIX sun_path limit (~108)


def _fixtures_dir(run_dir: Path) -> Path:
    return run_dir / "fixtures"


def _extractor_union(dev_anon: dict[str, dict[str, list[ExtractedTable]]]) -> list[str]:
    """Ordered union of extractor names across all dev attachments."""
    seen: list[str] = []
    for per_ext in dev_anon.values():
        for ext in per_ext:
            if ext not in seen:
                seen.append(ext)
    return seen


def write_anon_fixtures(
    run_dir: Path,
    dev_anon: dict[str, dict[str, list[ExtractedTable]]],
    attachment_ids: list[str] | None = None,
) -> dict:
    """Write one anonymized fixture per (extractor, attachment) + an index, returning metrics only.

    ``dev_anon`` maps attachment_id -> extractor -> anonymized tables. Each is written to
    ``fixtures/anon_<extractor>_<attachment_id>.json`` so the agent can compare all extractors'
    output and pick a winner. The index records the extractor and attachment names. The returned
    metrics contain only counts — no raw data.
    """
    ids = (
        list(dev_anon)
        if not attachment_ids
        else [a for a in attachment_ids if a in dev_anon]
    )
    fixtures = _fixtures_dir(run_dir)
    fixtures.mkdir(parents=True, exist_ok=True)

    table_counts: dict[str, int] = {}
    for aid in ids:
        for ext, tables in dev_anon[aid].items():
            (fixtures / f"anon_{ext}_{aid}.json").write_text(
                json.dumps([table_to_dict(t) for t in tables], ensure_ascii=False)
            )
            table_counts[f"{ext}/{aid}"] = len(tables)

    extractors = _extractor_union(dev_anon)
    index_path = fixtures / "anon_index.json"
    index_path.write_text(
        json.dumps({"extractors": extractors, "attachments": list(dev_anon)})
    )

    return {
        "extractors": extractors,
        "attachments": ids,
        "table_counts": table_counts,
    }


class ToolBroker:
    """Threaded Unix-socket server feeding anonymized dev data to the in-sandbox tool."""

    def __init__(
        self, run_dir: Path, dev_anon: dict[str, dict[str, list[ExtractedTable]]]
    ) -> None:
        self.run_dir = Path(run_dir)
        self.dev_anon = dev_anon
        self.token = secrets.token_hex(16)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def socket_path(self) -> str:
        return str((self.run_dir / SOCKET_NAME).resolve())

    @property
    def token_path(self) -> Path:
        return self.run_dir / TOKEN_FILE

    def start(self) -> None:
        """Bind the socket (0600), write the token file, and serve in a daemon thread."""
        sp = self.socket_path
        if len(sp) >= _MAX_SOCKET_PATH:
            raise CodegenError(f"sandbox socket path too long ({len(sp)} chars): {sp}")
        if os.path.exists(sp):
            os.unlink(sp)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(sp)
        os.chmod(sp, 0o600)
        self._sock.listen(8)
        self._sock.settimeout(0.5)
        self.token_path.write_text(
            json.dumps({"token": self.token, "socket": IN_SANDBOX_SOCKET})
        )
        os.chmod(self.token_path, 0o600)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        logger.info("tool broker listening at %s", sp)

    def stop(self) -> None:
        """Stop serving and remove the socket + token file."""
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
        for p in (self.socket_path, str(self.token_path)):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass
        logger.info("tool broker stopped")

    def __enter__(self) -> ToolBroker:
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # -- internals ----------------------------------------------------------

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    self._handle(conn)
                except Exception as exc:  # noqa: BLE001 — never crash the broker thread
                    logger.warning("broker handler error: %s", exc)

    def _handle(self, conn: socket.socket) -> None:
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
        try:
            req = json.loads(data.decode() or "{}")
        except json.JSONDecodeError:
            return self._send(conn, {"ok": False, "error": "invalid JSON request"})

        if req.get("token") != self.token:
            return self._send(conn, {"ok": False, "error": "unauthorized"})

        action = req.get("action")
        args = req.get("args") or {}
        try:
            if action == "describe_dev_set":
                resp = {"ok": True, "data": self._describe()}
            elif action == "regenerate_anonymized_tables":
                metrics = write_anon_fixtures(
                    self.run_dir, self.dev_anon, args.get("attachment_ids")
                )
                resp = {"ok": True, "data": metrics}
            else:
                resp = {"ok": False, "error": f"unknown action: {action!r}"}
        except Exception as exc:  # noqa: BLE001 — return structured error, never raw data
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._send(conn, resp)

    def _describe(self) -> dict:
        return {
            "attachments": list(self.dev_anon),
            "extractors": _extractor_union(self.dev_anon),
            "tables": {
                aid: {
                    ext: [
                        {
                            "columns": (t.rows[0] if t.rows else []),
                            "n_rows": max(0, len(t.rows) - 1),
                        }
                        for t in tables
                    ]
                    for ext, tables in per_ext.items()
                }
                for aid, per_ext in self.dev_anon.items()
            },
        }

    def _send(self, conn: socket.socket, obj: dict) -> None:
        conn.sendall((json.dumps(obj) + "\n").encode())
