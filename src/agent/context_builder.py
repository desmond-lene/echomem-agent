"""Build chat-completion messages from session and retrieval context."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .config import ChatConfig


class ContextBuilder:
    def __init__(self, config: ChatConfig) -> None:
        self.config = config

    def build(
        self,
        *,
        user_id: str,
        agent_id: str,
        session_id: str,
        user_message: str,
        history: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "system",
                "content": (
                    "<agent_context>\n"
                    "身份：你是 EchoMemory Agent，一个接入长期记忆、会话归档和检索增强能力的任务型 Agent。\n"
                    "工作方式：先理解用户目标，再结合可用记忆与当前会话事实，给出可执行、可验证的回答。\n"
                    "行为边界：不要编造未给出的事实；证据不足时说明不确定性；不要暴露 API key、内部配置或隐藏系统细节。\n"
                    "输出偏好：中文优先，结构清楚，避免空泛套话；复杂任务先给短计划，再给结果。\n"
                    "</agent_context>"
                ),
            },
            {
                "role": "system",
                "content": (
                    "<runtime_context>\n"
                    f"user_id: {user_id}\n"
                    f"agent_id: {agent_id}\n"
                    f"session_id: {session_id}\n"
                    f"timestamp_utc: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
                    f"retrieval_enabled: {self.config.retrieval_enabled}\n"
                    f"retrieval_limit: {self.config.retrieval_limit}\n"
                    "</runtime_context>"
                ),
            },
        ]
        history_messages = _messages_from_history(history)
        if history_messages:
            messages.append({"role": "system", "content": _format_history(history_messages)})
        items = _items_from_retrieval(retrieval)
        if items:
            messages.append({"role": "system", "content": _format_memory(items)})
        else:
            messages.append(
                {
                    "role": "system",
                    "content": "<memory>\n本轮 EchoMemory 检索没有返回可用记忆。\n</memory>",
                }
            )
        messages.append({"role": "user", "content": f"<session>\n{user_message}\n</session>"})
        return messages


def _messages_from_history(history: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(history.get("messages"), list):
        return history["messages"]
    data = history.get("history")
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data["messages"]
    return []


def _format_history(messages: list[dict[str, Any]]) -> str:
    lines = ["<session_history>"]
    for item in messages:
        role = item.get("role") or "unknown"
        content = item.get("content") or ""
        if not content:
            continue
        lines.append(f"{role}: {content}")
    lines.append("</session_history>")
    return "\n".join(lines)


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


def _format_memory(items: list[Any]) -> str:
    lines = ["<memory>"]
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            text = item.get("content") or item.get("text") or item.get("summary") or json.dumps(
                item, ensure_ascii=False
            )
            kind = item.get("kind") or item.get("type") or "memory"
            score = item.get("score")
            prefix = f"{index}. [{kind}]"
            if score is not None:
                prefix += f" score={score}"
            lines.append(f"{prefix} {text}")
        else:
            lines.append(f"{index}. {item}")
    lines.append("</memory>")
    return "\n".join(lines)
