"""Single-turn Agent chat orchestration."""

from __future__ import annotations

from typing import Any

from .config import AgentConfig
from .context_builder import ContextBuilder
from .echomemory_client import EchoMemoryClient
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
        user_id = str(payload.get("user_id") or "").strip()
        agent_id = str(payload.get("agent_id") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        message = str(payload.get("message") or "").strip()
        if not user_id or not agent_id or not session_id or not message:
            raise ValueError("user_id, agent_id, session_id and message are required")

        scope = self.memory.open_session(user_id, agent_id, session_id)
        history = self.memory.get_history(session_id, limit=self.config.chat.history_turns)
        user_write = self.memory.add_message(session_id, "user", message)
        retrieval: dict[str, Any] = {"items": []}
        if self.config.chat.retrieval_enabled:
            retrieval = self.memory.search(
                query=message,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                limit=self.config.chat.retrieval_limit,
            )

        context_result = self.context_builder.build_with_trace(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            user_message=message,
            history=history,
            retrieval=retrieval,
        )
        messages = context_result.messages
        completion = self.model.complete(messages)
        assistant_write = self.memory.add_message(session_id, "assistant", completion.content)

        commit_result = None
        if self.config.chat.auto_commit == "each_turn":
            commit_result = self.memory.commit(session_id)

        return {
            "session_id": session_id,
            "assistant": {"role": "assistant", "content": completion.content},
            "history": history,
            "retrieval": retrieval,
            "model": {"provider": self.config.model.provider, "model": self.config.model.model},
            "messages": messages,
            "context_trace": context_result.trace if self.config.context.debug_trace_enabled else None,
            "writes": {"session": scope, "user": user_write, "assistant": assistant_write},
            "commit": commit_result,
        }
