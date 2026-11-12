"""Loopback-only HTTP chat interface for the local OpenVINO model."""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from localmind import ConversationHistory, DEFAULT_MAX_NEW_TOKENS, LocalMindEngine


WEB_ROOT = Path(__file__).resolve().parent / "web"
MAX_REQUEST_BYTES = 16_384
MAX_ACTIVE_SESSIONS = 32
SESSION_ID_PATTERN = re.compile(r"^[0-9a-fA-F-]{36}$")


@dataclass
class SessionState:
    history: ConversationHistory
    lock: Lock = field(default_factory=Lock)
    active_requests: int = 0


def make_handler(engine: LocalMindEngine) -> type[BaseHTTPRequestHandler]:
    sessions: dict[str, SessionState] = {}
    sessions_lock = Lock()

    class ChatHandler(BaseHTTPRequestHandler):
        server_version = "LocalMind/1.0"

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/":
                self._send_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            elif route == "/api/status":
                self._send_json(200, {"ready": True, **engine.status()})
            else:
                self._send_json(404, {"error": "Not found."})

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            if route not in {"/api/chat", "/api/reset"}:
                self._send_json(404, {"error": "Not found."})
                return

            try:
                payload = self._read_json()
                session_id = payload.get("session_id")
                if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
                    raise ValueError("session_id must be a UUID.")
                session_id = str(uuid.UUID(session_id))

                with sessions_lock:
                    state = sessions.get(session_id)
                    if state is None:
                        if len(sessions) >= MAX_ACTIVE_SESSIONS:
                            self._send_json(
                                503,
                                {
                                    "error": (
                                        "LocalMind has reached its active conversation "
                                        "limit. Clear an older conversation and retry."
                                    )
                                },
                            )
                            return
                        state = SessionState(engine.new_history())
                        sessions[session_id] = state
                    state.active_requests += 1

                try:
                    with state.lock:
                        history = state.history
                        if route == "/api/reset":
                            history.clear()
                            self._send_json(200, {"cleared": True})
                            return

                        message = payload.get("message")
                        max_new_tokens = payload.get(
                            "max_new_tokens", DEFAULT_MAX_NEW_TOKENS
                        )
                        response, metrics = engine.respond(
                            history, message, max_new_tokens
                        )
                        self._send_json(
                            200,
                            {
                                "response": response,
                                "metrics": metrics,
                                "session_id": session_id,
                            },
                        )
                finally:
                    with sessions_lock:
                        state.active_requests -= 1
                        if (
                            route == "/api/reset"
                            and state.active_requests == 0
                            and sessions.get(session_id) is state
                        ):
                            del sessions[session_id]
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                print(f"Request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._send_json(500, {"error": "The local model could not complete the request."})

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header.") from exc
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError(f"Request body must be between 1 and {MAX_REQUEST_BYTES} bytes.")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Request body must be a JSON object.")
            return value

        def _send_json(self, status: int, value: dict[str, Any]) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            try:
                body = path.read_bytes()
            except OSError as exc:
                print(f"Could not read web interface {path}: {exc}", file=sys.stderr)
                self._send_json(500, {"error": "The web interface file could not be read."})
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"HTTP {self.address_string()} - {fmt % args}", file=sys.stderr)

    return ChatHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Start LocalMind on this computer.")
    parser.add_argument("--model", help="Path to an OpenVINO GenAI model directory.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")

    try:
        engine = LocalMindEngine(args.model, "CPU")
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    except Exception as exc:
        print(f"Could not start LocalMind web chat: {exc}", file=sys.stderr)
        return 1

    print(f"LocalMind is available at http://127.0.0.1:{args.port}")
    print("The server binds to loopback only; it is not configured for shared company access.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down LocalMind.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
