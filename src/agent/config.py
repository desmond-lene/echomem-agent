"""Runtime configuration for the standalone agent server."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_ECHOMEM_URL = "http://127.0.0.1:8000"
DEFAULT_OBSERVER_CONFIG = r"E:\KVCache\observer\observer-config.json"


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass(frozen=True)
class EchoMemoryConfig:
    base_url: str = DEFAULT_ECHOMEM_URL
    timeout_seconds: float = 15
    auth_key: str = ""


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout_seconds: float = 60


@dataclass(frozen=True)
class ChatConfig:
    system_prompt: str = "你是一个接入 EchoMemory 的任务型 Agent。回答要清晰、可靠，并说明关键依据。"
    history_turns: int = 8
    retrieval_enabled: bool = True
    retrieval_limit: int = 6
    auto_commit: str = "manual"
    context_budget_tokens: int = 12000


@dataclass(frozen=True)
class ContextConfig:
    phase: str = "dialogue"
    stable_prefix_version: str = "context-v1"
    debug_trace_enabled: bool = True
    tool_context_enabled: bool = False
    workspace_state_enabled: bool = False
    summary_enabled: bool = False


@dataclass(frozen=True)
class AgentConfig:
    server: ServerConfig = ServerConfig()
    echomemory: EchoMemoryConfig = EchoMemoryConfig()
    model: ModelConfig = ModelConfig()
    chat: ChatConfig = ChatConfig()
    context: ContextConfig = ContextConfig()

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["model"].get("api_key"):
            data["model"]["api_key"] = "***"
        if data["echomemory"].get("auth_key"):
            data["echomemory"]["auth_key"] = "***"
        return data


def load_config(config_path: str | None = None, *, echomem_url: str | None = None) -> AgentConfig:
    """Load config from JSON, then apply environment and explicit overrides."""

    path = config_path or os.environ.get("AGENT_CONFIG")
    data: dict[str, Any] = {}
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))

    config = AgentConfig(
        server=_build(ServerConfig, data.get("server", {})),
        echomemory=_build(EchoMemoryConfig, data.get("echomemory", {})),
        model=_build(ModelConfig, data.get("model", {})),
        chat=_build(ChatConfig, data.get("chat", {})),
        context=_build(ContextConfig, data.get("context", {})),
    )

    config = replace(
        config,
        echomemory=replace(
            config.echomemory,
            base_url=echomem_url
            or os.environ.get("ECHOMEM_URL")
            or config.echomemory.base_url,
            auth_key=os.environ.get("ECHOMEM_AUTH_KEY", config.echomemory.auth_key),
        ),
        model=replace(
            config.model,
            base_url=os.environ.get("OPENAI_BASE_URL", config.model.base_url),
            api_key=os.environ.get("OPENAI_API_KEY", config.model.api_key),
            model=os.environ.get("OPENAI_MODEL", config.model.model),
        ),
    )
    config = _apply_observer_alibaba_config(config)
    return config


def _build(config_type: type[Any], values: dict[str, Any]) -> Any:
    fields = set(config_type.__dataclass_fields__)
    return config_type(**{key: value for key, value in values.items() if key in fields})


def _apply_observer_alibaba_config(config: AgentConfig) -> AgentConfig:
    """Use observer's Alibaba provider unless an explicit base URL override is present."""

    if os.environ.get("OPENAI_BASE_URL"):
        return config
    observer_path = os.environ.get("OBSERVER_CONFIG") or DEFAULT_OBSERVER_CONFIG
    path = Path(observer_path)
    if not path.exists():
        return config

    data = json.loads(path.read_text(encoding="utf-8"))
    providers = data.get("providers", {})
    alibaba = providers.get("alibaba", {})
    base_url = alibaba.get("baseUrl")
    api_key = alibaba.get("apiKey")
    model = _first_model_for_provider(data.get("model_map", {}), "alibaba")
    if not base_url or not api_key or not model:
        return config

    return replace(
        config,
        model=replace(
            config.model,
            provider="alibaba",
            base_url=base_url,
            api_key=api_key,
            model=model,
        ),
    )

def _first_model_for_provider(model_map: dict[str, Any], provider: str) -> str | None:
    for model, mapped_provider in model_map.items():
        if mapped_provider == provider:
            return model
    return None
