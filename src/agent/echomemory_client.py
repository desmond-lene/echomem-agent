"""HTTP client for EchoMemory public APIs."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from .config import EchoMemoryConfig


class EchoMemoryClientError(RuntimeError):
    """Raised when EchoMemory returns an error or cannot be reached."""


class EchoMemoryClient:
    def __init__(self, config: EchoMemoryConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/") + "/"

    def open_session(self, user_id: str, agent_id: str, session_id: str) -> dict[str, Any]:
        return self.post(
            "/api/sessions/open",
            {"user_id": user_id, "agent_id": agent_id, "session_id": session_id},
        )

    def add_message(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        return self.post(f"/api/sessions/{session_id}/messages", {"role": role, "content": content})

    def get_history(self, session_id: str, *, limit: int) -> dict[str, Any]:
        uri = f"echo://sessions/{session_id}/current/messages.jsonl"
        try:
            payload = self.get(f"/agent/inspect/fs/read?{urlencode({'uri': uri})}")
        except EchoMemoryClientError as exc:
            message = str(exc)
            if "echomemory_http_404" in message or "No such file or directory" in message:
                return {"messages": []}
            raise
        messages: list[dict[str, Any]] = []
        for line in str(payload.get("text") or "").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                messages.append(item)
        return {"messages": messages[-limit:]}

    def search(
        self,
        *,
        query: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        limit: int,
    ) -> dict[str, Any]:
        return self.post(
            "/api/retrieval/search",
            {
                "query": query,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "limit": limit,
                "include_explain": True,
            },
        )

    def commit(self, session_id: str) -> dict[str, Any]:
        return self.post(f"/api/sessions/{session_id}/commit", {})

    def get_commit_memories(self, session_id: str, commit_id: str) -> dict[str, Any]:
        return self.get(f"/api/sessions/{session_id}/commits/{commit_id}/memories")

    def get(self, path: str) -> dict[str, Any]:
        request = Request(urljoin(self.base_url, path.lstrip("/")), method="GET")
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EchoMemoryClientError(f"echomemory_http_{exc.code}: {detail}") from exc
        except URLError as exc:
            raise EchoMemoryClientError(f"echomemory_unreachable: {exc.reason}") from exc

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EchoMemoryClientError(f"echomemory_http_{exc.code}: {detail}") from exc
        except URLError as exc:
            raise EchoMemoryClientError(f"echomemory_unreachable: {exc.reason}") from exc
