"""Single-turn Agent chat orchestration."""

from __future__ import annotations

import re
from typing import Any

from .config import AgentConfig
from .context_builder import ContextBuilder
from .echomemory_client import EchoMemoryClient, EchoMemoryClientError
from .model_client import OpenAICompatibleChatClient


class AgentChatService:
    def __init__(
        self,
        config: AgentConfig,
        *,
        memory: EchoMemoryClient | None = None,
        model: OpenAICompatibleChatClient | None = None,
    ) -> None:
        self.config = config
        self.memory = memory or EchoMemoryClient(config.echomemory)
        self.model = model or OpenAICompatibleChatClient(config.model)
        self.context_builder = ContextBuilder(config.chat, config.context)

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id, agent_id, session_id, message = self._parse_chat_payload(payload)
        include_history = self._include_history(payload)

        scope = self.memory.open_session(user_id, agent_id, session_id)
        history = (
            self.memory.get_history(session_id, limit=self.config.chat.history_turns)
            if include_history
            else {"messages": []}
        )
        user_write = self.memory.add_message(session_id, "user", message)
        retrieval = self._search_retrieval(
            query=message,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

        context_result = self._build_context(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            message=message,
            history=history,
            retrieval=retrieval,
            include_history=include_history,
        )
        messages = context_result.messages
        completion = self.model.complete(messages)
        final_answer = self._ground_answer(message, retrieval, completion.content)
        assistant_write = self.memory.add_message(session_id, "assistant", final_answer)

        commit_result = None
        if self.config.chat.auto_commit == "each_turn":
            commit_result = self.memory.commit(session_id)

        return {
            "session_id": session_id,
            "assistant": {"role": "assistant", "content": final_answer},
            "history": history,
            "retrieval": retrieval,
            "model": {"provider": self.config.model.provider, "model": self.config.model.model},
            "messages": messages,
            "context_trace": context_result.trace if self.config.context.debug_trace_enabled else None,
            "writes": {"session": scope, "user": user_write, "assistant": assistant_write},
            "commit": commit_result,
        }

    def preview_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id, agent_id, session_id, message = self._parse_chat_payload(payload)
        include_history = self._include_history(payload)

        scope = self.memory.open_session(user_id, agent_id, session_id)
        history = (
            self.memory.get_history(session_id, limit=self.config.chat.history_turns)
            if include_history
            else {"messages": []}
        )
        retrieval = self._search_retrieval(
            query=message,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        context_result = self._build_context(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            message=message,
            history=history,
            retrieval=retrieval,
            include_history=include_history,
        )

        return {
            "session_id": session_id,
            "history": history,
            "retrieval": retrieval,
            "messages": context_result.messages,
            "context_trace": context_result.trace if self.config.context.debug_trace_enabled else None,
            "writes": {"session": scope},
        }

    def _parse_chat_payload(self, payload: dict[str, Any]) -> tuple[str, str, str, str]:
        user_id = str(payload.get("user_id") or "").strip()
        agent_id = str(payload.get("agent_id") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        message = str(payload.get("message") or "").strip()
        if not user_id or not agent_id or not session_id or not message:
            raise ValueError("user_id, agent_id, session_id and message are required")
        return user_id, agent_id, session_id, message

    def _include_history(self, payload: dict[str, Any]) -> bool:
        return payload.get("include_history", True) is not False

    def _search_retrieval(
        self,
        *,
        query: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        if not self.config.chat.retrieval_enabled:
            return {"items": []}
        queries = self._expand_queries(query)
        merged_items: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            for item_query in queries:
                result = self.memory.search(
                    query=item_query,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    limit=self.config.chat.retrieval_limit,
                )
                merged_items.extend(self._extract_items(result))
        except EchoMemoryClientError as exc:
            errors.append(str(exc))

        if not merged_items:
            if errors:
                return {"items": [], "error": errors[0], "degraded": True}
            return {"items": []}

        ranked_items = self._rank_items(query, merged_items)
        return {
            "items": ranked_items[: self.config.chat.retrieval_limit],
            "query_plan": queries,
            "degraded": bool(errors),
            "errors": errors,
        }

    def _expand_queries(self, query: str) -> list[str]:
        base = query.strip()
        if not base:
            return [query]
        parts = [base]
        splitters = r"[，,。；;：:、\n]|(?:和|与|及|以及|并且|同时|然后)|(?:and|then|also)"
        for segment in re.split(splitters, base):
            cleaned = segment.strip()
            if len(cleaned) >= 4:
                parts.append(cleaned)
        quoted = re.findall(r"[\"“”'‘’]([^\"“”'‘’]{2,64})[\"“”'‘’]", base)
        parts.extend(item.strip() for item in quoted if item.strip())
        deduped: list[str] = []
        for item in parts:
            if item and item not in deduped:
                deduped.append(item)
        return deduped[:5]

    def _extract_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(payload.get("items"), list):
            return [item for item in payload["items"] if isinstance(item, dict)]
        result = payload.get("result")
        if isinstance(result, dict) and isinstance(result.get("items"), list):
            return [item for item in result["items"] if isinstance(item, dict)]
        return []

    def _rank_items(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keywords = [token for token in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", query.lower()) if len(token) >= 2]
        unique: dict[str, dict[str, Any]] = {}
        for item in items:
            text = str(item.get("text") or item.get("content") or item.get("summary") or "")
            key = f"{item.get('kind','')}|{text}"
            if key not in unique:
                unique[key] = item
        def score(item: dict[str, Any]) -> tuple[float, int, int]:
            text = str(item.get("text") or item.get("content") or item.get("summary") or "").lower()
            overlap = sum(1 for token in keywords if token in text)
            base = float(item.get("score") or 0.0)
            kind = str(item.get("kind") or "")
            kind_bonus = 1 if kind in {"relation", "causal", "episode", "task"} else 0
            return (base, overlap, kind_bonus)
        return sorted(unique.values(), key=score, reverse=True)

    def _build_context(
        self,
        *,
        user_id: str,
        agent_id: str,
        session_id: str,
        message: str,
        history: dict[str, Any],
        retrieval: dict[str, Any],
        include_history: bool,
    ):
        return self.context_builder.build_with_trace(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            user_message=message,
            history=history,
            retrieval=retrieval,
            include_history=include_history,
        )

    def _ground_answer(self, query: str, retrieval: dict[str, Any], draft: str) -> str:
        items = self._extract_items(retrieval)[:12]
        if not items:
            return draft
        evidence_lines = []
        for index, item in enumerate(items, start=1):
            kind = str(item.get("kind") or "memory")
            text = str(item.get("text") or item.get("content") or item.get("summary") or "").strip()
            if not text:
                continue
            evidence_lines.append(f"{index}. [{kind}] {text}")
        if not evidence_lines:
            return draft
        prompt = (
            "你是回答校正器。请把“回答初稿”改写为严格基于“检索证据”的最终回答。\n"
            "规则：\n"
            "1) 不得引入证据中没有的新人物、新公司、新时间、新事件。\n"
            "2) 若证据不足，明确写“证据不足”，但保留已能确定的部分结论。\n"
            "3) 尽量直接回答问题，保持简洁。\n\n"
            f"问题：{query}\n\n"
            f"检索证据：\n{chr(10).join(evidence_lines)}\n\n"
            f"回答初稿：\n{draft}\n"
        )
        try:
            revised = self.model.complete([{"role": "user", "content": prompt}]).content.strip()
            return revised or draft
        except Exception:
            return draft
