"""OpenAI-compatible chat completion client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .config import ModelConfig


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    raw: dict[str, Any]


class ModelClientError(RuntimeError):
    """Raised when the model provider cannot return a completion."""


class OpenAICompatibleChatClient:
    """Minimal non-streaming client for /chat/completions."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/") + "/"

    def complete(self, messages: list[dict[str, str]]) -> ChatCompletionResult:
        if not self.config.api_key:
            raise ModelClientError("model api_key is not configured")

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            urljoin(self.base_url, "chat/completions"),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ModelClientError(f"model_http_{exc.code}: {detail}") from exc
        except URLError as exc:
            raise ModelClientError(f"model_unreachable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ModelClientError("model_timeout") from exc

        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelClientError("model response did not include choices[0].message.content") from exc
        return ChatCompletionResult(content=content or "", raw=raw)
