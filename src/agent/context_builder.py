"""Build Phase 1 chat-completion messages from session and retrieval context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import ChatConfig, ContextConfig


@dataclass(frozen=True)
class ContextBuildResult:
    messages: list[dict[str, str]]
    trace: dict[str, Any]


class ContextBuilder:
    def __init__(self, config: ChatConfig, context_config: ContextConfig | None = None) -> None:
        self.config = config
        self.context_config = context_config or ContextConfig()

    def build(
        self,
        *,
        user_id: str,
        agent_id: str,
        session_id: str,
        user_message: str,
        history: dict[str, Any],
        retrieval: dict[str, Any],
        include_history: bool = True,
    ) -> list[dict[str, str]]:
        return self.build_with_trace(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            user_message=user_message,
            history=history,
            retrieval=retrieval,
            include_history=include_history,
        ).messages

    def build_with_trace(
        self,
        *,
        user_id: str,
        agent_id: str,
        session_id: str,
        user_message: str,
        history: dict[str, Any],
        retrieval: dict[str, Any],
        include_history: bool = True,
    ) -> ContextBuildResult:
        del user_id, agent_id, session_id

        trace_layers: list[dict[str, Any]] = []
        messages = [
            {
                "role": "system",
                "content": (
                    "<agent_charter>\n"
                    f"{self.config.system_prompt}\n"
                    "\u4f60\u5177\u5907\u957f\u671f\u8bb0\u5fc6\u68c0\u7d22\u80fd\u529b\u3002\u8bf7\u4f18\u5148\u57fa\u4e8e\u68c0\u7d22\u5230\u7684\u4e8b\u5b9e\u4f5c\u7b54\u3002\n"
                    "\u56de\u7b54\u65f6\u5148\u7ed9\u7ed3\u8bba\uff0c\u518d\u7ed9\u8bc1\u636e\u4e0e\u63a8\u7406\uff1b\u5c3d\u91cf\u660e\u786e\u65f6\u95f4\u3001\u4eba\u7269\u3001\u516c\u53f8\u4e0e\u56e0\u679c\u5173\u7cfb\u3002\n"
                    "</agent_charter>"
                ),
            },
            {
                "role": "system",
                "content": (
                    "<behavior_policy>\n"
                    "\u4e0d\u8981\u7f16\u9020\u4e8b\u5b9e\uff1b\u8bc1\u636e\u4e0d\u8db3\u65f6\u8bf4\u660e\u4e0d\u786e\u5b9a\u70b9\uff0c\u4f46\u5148\u7ed9\u6700\u53ef\u80fd\u7ed3\u8bba\u3002\n"
                    "\u82e5\u5b58\u5728\u591a\u4e2a\u5019\u9009\u7b54\u6848\uff0c\u7ed9\u51fa\u6392\u5e8f\u548c\u4f60\u9009\u62e9\u7684\u4f9d\u636e\u3002\n"
                    "\u82e5\u68c0\u7d22\u4fe1\u606f\u51b2\u7a81\uff0c\u4f18\u5148\u66f4\u5177\u4f53\u3001\u65f6\u95f4\u66f4\u660e\u786e\u3001\u6765\u6e90\u66f4\u76f4\u63a5\u7684\u8bc1\u636e\u3002\n"
                    "\u9ed8\u8ba4\u4f7f\u7528\u4e2d\u6587\uff0c\u8868\u8fbe\u7b80\u6d01\u660e\u786e\uff0c\u907f\u514d\u7a7a\u6cdb\u514d\u8d23\u58f0\u660e\u3002\n"
                    "</behavior_policy>"
                ),
            },
        ]
        _add_trace_layer(trace_layers, "Agent \u7ae0\u7a0b", "\u9759\u6001\u89c4\u5219", [0], messages)
        _add_trace_layer(trace_layers, "\u884c\u4e3a\u89c4\u5219", "\u9759\u6001\u89c4\u5219", [1], messages)

        items = _items_from_retrieval(retrieval)
        messages.append({"role": "system", "content": _format_retrieved_memory(items)})
        _add_trace_layer(
            trace_layers,
            "Retrieved Memory",
            "EchoMemory",
            [len(messages) - 1],
            messages,
            item_count=len(items),
            highlight=True,
        )

        history_messages = _messages_from_history(history) if include_history else []
        if history_messages:
            start = len(messages)
            messages.extend(_format_conversation_tail(history_messages))
            _add_trace_layer(
                trace_layers,
                "近期对话",
                "EchoMemory 会话",
                list(range(start, len(messages))),
                messages,
                item_count=len(messages) - start,
            )
        else:
            trace_layers.append(
                {
                    "name": "近期对话",
                    "source": "EchoMemory 会话",
                    "enabled": False,
                    "message_indexes": [],
                    "char_count": 0,
                    "item_count": 0,
                }
            )

        messages.append({"role": "user", "content": f"<current_request>\n{user_message}\n</current_request>"})
        _add_trace_layer(trace_layers, "当前请求", "用户", [len(messages) - 1], messages)
        return ContextBuildResult(
            messages=messages,
            trace={
                "phase": self.context_config.phase,
                "stable_prefix_version": self.context_config.stable_prefix_version,
                "layers": trace_layers,
            },
        )


def _add_trace_layer(
    layers: list[dict[str, Any]],
    name: str,
    source: str,
    message_indexes: list[int],
    messages: list[dict[str, str]],
    *,
    item_count: int | None = None,
    highlight: bool = False,
) -> None:
    char_count = sum(len(messages[index].get("content", "")) for index in message_indexes)
    layer: dict[str, Any] = {
        "name": name,
        "source": source,
        "enabled": True,
        "message_indexes": message_indexes,
        "char_count": char_count,
        "highlight": highlight,
    }
    if item_count is not None:
        layer["item_count"] = item_count
    layers.append(layer)


def _messages_from_history(history: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(history.get("messages"), list):
        return history["messages"]
    data = history.get("history")
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data["messages"]
    return []


def _format_conversation_tail(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    tail: list[dict[str, str]] = []
    for item in messages:
        role = str(item.get("role") or "user")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            tail.append({"role": role, "content": content})
    return tail


def _items_from_retrieval(retrieval: dict[str, Any]) -> list[Any]:
    if isinstance(retrieval.get("items"), list):
        return retrieval["items"]
    if isinstance(retrieval.get("results"), list):
        return retrieval["results"]
    data = retrieval.get("data")
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    result = retrieval.get("result")
    if isinstance(result, dict) and isinstance(result.get("items"), list):
        return result["items"]
    return []


def _format_retrieved_memory(items: list[Any]) -> str:
    lines = ['<retrieved_memory source="EchoMemory">', "## Retrieved Memory"]
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            text = item.get("content") or item.get("text") or item.get("summary") or json.dumps(
                item, ensure_ascii=False
            )
            kind = item.get("kind") or item.get("type") or "memory"
            score = item.get("score")
            source = item.get("source") or item.get("uri") or item.get("id")
            prefix = f"{index}. [{kind}]"
            if score is not None:
                prefix += f" score={score}"
            if source:
                prefix += f" source={source}"
            lines.append(f"{prefix} {text}")
        else:
            lines.append(f"{index}. {item}")
    lines.append("</retrieved_memory>")
    return "\n".join(lines)
