"""Configuration loading tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config import load_config


class ConfigTests(unittest.TestCase):
    def test_observer_alibaba_config_wins_over_openai_api_key_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            observer_config = Path(temp_dir) / "observer-config.json"
            observer_config.write_text(
                json.dumps(
                    {
                        "providers": {
                            "alibaba": {
                                "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                                "apiKey": "alibaba-key",
                            }
                        },
                        "model_map": {"qwen3-max-2026-01-23": "alibaba"},
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "OBSERVER_CONFIG": str(observer_config),
                    "OPENAI_API_KEY": "openai-key",
                },
                clear=True,
            ):
                config = load_config()

        self.assertEqual(config.model.provider, "alibaba")
        self.assertEqual(config.model.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(config.model.api_key, "alibaba-key")
        self.assertEqual(config.model.model, "qwen3-max-2026-01-23")

    def test_explicit_base_url_override_skips_observer_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            observer_config = Path(temp_dir) / "observer-config.json"
            observer_config.write_text(
                json.dumps(
                    {
                        "providers": {
                            "alibaba": {
                                "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                                "apiKey": "alibaba-key",
                            }
                        },
                        "model_map": {"qwen3-max-2026-01-23": "alibaba"},
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "OBSERVER_CONFIG": str(observer_config),
                    "OPENAI_BASE_URL": "http://127.0.0.1:9999/v1",
                    "OPENAI_API_KEY": "test-key",
                    "OPENAI_MODEL": "fake-chat",
                },
                clear=True,
            ):
                config = load_config()

        self.assertEqual(config.model.provider, "openai_compatible")
        self.assertEqual(config.model.base_url, "http://127.0.0.1:9999/v1")
        self.assertEqual(config.model.model, "fake-chat")

    def test_echomemory_auth_key_loads_from_env(self) -> None:
        with patch.dict("os.environ", {"ECHOMEM_AUTH_KEY": "ek_test"}, clear=True):
            config = load_config()

        self.assertEqual(config.echomemory.auth_key, "ek_test")

    def test_context_config_loads_from_agent_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "agent.local.json"
            config_path.write_text(
                json.dumps(
                    {
                        "context": {
                            "phase": "dialogue",
                            "stable_prefix_version": "context-v2",
                            "debug_trace_enabled": False,
                            "tool_context_enabled": False,
                            "workspace_state_enabled": False,
                            "summary_enabled": False,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"AGENT_CONFIG": str(config_path)}, clear=True):
                config = load_config()

        self.assertEqual(config.context.phase, "dialogue")
        self.assertEqual(config.context.stable_prefix_version, "context-v2")
        self.assertFalse(config.context.debug_trace_enabled)
        self.assertFalse(config.public_dict()["context"]["debug_trace_enabled"])


if __name__ == "__main__":
    unittest.main()
