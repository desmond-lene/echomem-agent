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


if __name__ == "__main__":
    unittest.main()
