"""Agent playground server tests."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.request import Request, urlopen

from agent.server import create_server as create_agent_server


class FakeEchoMemoryHandler(BaseHTTPRequestHandler):
    """Tiny EchoMemory-compatible HTTP double."""

    calls: list[dict[str, object]] = []

    def do_GET(self) -> None:
        self.calls.append({"method": "GET", "path": self.path, "payload": {}, "headers": dict(self.headers)})
        if self.path == "/agent/inspect/runtime":
            self._send_json(
                HTTPStatus.OK,
                {
                    "service": "echomem",
                    "version": "0.1.0",
                    "features": {"session_service": "ready:D03"},
                },
            )
            return
        if self.path.startswith("/agent/inspect/fs/tree"):
            self._send_json(
                HTTPStatus.OK,
                {
                    "uri": "echo://sessions/chat-001",
                    "entries": [
                        {
                            "uri": "echo://sessions/chat-001/current/session.json",
                            "name": "session.json",
                            "kind": "file",
                            "size": 2,
                        }
                    ],
                },
            )
            return
        if self.path.startswith("/agent/inspect/fs/read"):
            text = "{}"
            if "messages.jsonl" in self.path:
                text = "\n".join(
                    [
                        json.dumps({"id": "msg_prev_001", "role": "user", "content": "你好"}, ensure_ascii=False),
                        json.dumps(
                            {"id": "msg_prev_002", "role": "assistant", "content": "你好，我可以帮你整理方案。"},
                            ensure_ascii=False,
                        ),
                    ]
                )
            self._send_json(HTTPStatus.OK, {"uri": "echo://sessions/chat-001/current/messages.jsonl", "text": text})
            return
        if self.path == "/agent/inspect/events":
            self._send_json(HTTPStatus.OK, {"events": []})
            return
        if self.path == "/api/sessions/chat-001/history?limit=8":
            self._send_json(
                HTTPStatus.OK,
                {
                    "history": {
                        "scope": {"user_id": "alice", "agent_id": "demo-agent", "session_id": "chat-001"},
                        "messages": [
                            {"id": "msg_prev_001", "role": "user", "content": "你好"},
                            {"id": "msg_prev_002", "role": "assistant", "content": "你好，我可以帮你整理方案。"},
                        ],
                    }
                },
            )
            return
        if self.path == "/api/sessions/chat-001/commits/archive_001/memories":
            self._send_json(
                HTTPStatus.OK,
                {
                    "summary": {
                        "session_id": "chat-001",
                        "commit_id": "archive_001",
                        "memory_kinds": ["preference"],
                        "memories": [{"engine_id": "simple", "kind": "preference", "title": "user preference"}],
                    }
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
        payload = json.loads(body.decode("utf-8")) if body else {}
        self.calls.append({"method": "POST", "path": self.path, "payload": payload, "headers": dict(self.headers)})
        if self.path == "/api/auth/tenants":
            self._send_json(HTTPStatus.OK, {"tenant": {"tenant_id": "tenant_test"}})
            return
        if self.path == "/api/auth/tenants/tenant_test/users":
            self._send_json(HTTPStatus.OK, {"user": {"user_id": "user_test"}})
            return
        if self.path == "/api/auth/tenants/tenant_test/users/user_test/key":
            self._send_json(HTTPStatus.OK, {"auth_key": "ek_test"})
            return
        if self.path == "/api/sessions/open":
            self._send_json(
                HTTPStatus.OK,
                {"scope": {"session_id": "chat-001"}},
            )
            return
        if self.path == "/api/sessions/chat-001/messages":
            self._send_json(
                HTTPStatus.OK,
                {"message": {"id": "msg_001", "role": payload.get("role"), "content": payload.get("content")}},
            )
            return
        if self.path == "/api/retrieval/search":
            self._send_json(
                HTTPStatus.OK,
                {
                    "items": [
                        {
                            "kind": "preference",
                            "content": "用户希望方案简洁，优先列步骤。",
                            "score": 0.91,
                        }
                    ],
                    "explain": {"strategy": "fake"},
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, format: str, *args: object) -> None:
        """Silence test logs."""

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeModelHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible model double."""

    calls: list[dict[str, object]] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
        payload = json.loads(body.decode("utf-8")) if body else {}
        self.calls.append({"path": self.path, "payload": payload})
        if self.path == "/v1/chat/completions":
            self._send_json(
                HTTPStatus.OK,
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "这是模型生成的 D03 提交方案。",
                            }
                        }
                    ]
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, format: str, *args: object) -> None:
        """Silence test logs."""

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class AgentServerTests(unittest.TestCase):
    def test_agent_page_serves_chat_ui(self) -> None:
        server = create_agent_server(port=0, echomem_url="http://127.0.0.1:1")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=2) as response:
                html = response.read().decode("utf-8")
            self.assertEqual(response.status, HTTPStatus.OK)
            self.assertIn("EchoMemory 智能体对话", html)
            self.assertIn("sessionList", html)
            self.assertIn("提交归档", html)
            self.assertNotIn('id="commit"', html)
            self.assertIn("组装后的模型上下文", html)
            self.assertIn("memory-highlight", html)
            self.assertIn("EchoMemory 检索上下文", html)
            self.assertIn("近期对话", html)
            self.assertIn("文件系统目标", html)
            self.assertIn("平铺展开", html)
            self.assertIn("本次 commit 抽取了", html)
            self.assertIn("accountSelect", html)
            self.assertIn("createAccount", html)
            self.assertIn("currentAccountLabel", html)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_agent_proxy_uses_echomem_http_api(self) -> None:
        FakeEchoMemoryHandler.calls = []
        echomem = ThreadingHTTPServer(("127.0.0.1", 0), FakeEchoMemoryHandler)
        echomem_thread = threading.Thread(target=echomem.serve_forever, daemon=True)
        echomem_thread.start()
        agent = create_agent_server(port=0, echomem_url=f"http://127.0.0.1:{echomem.server_port}")
        agent_thread = threading.Thread(target=agent.serve_forever, daemon=True)
        agent_thread.start()

        try:
            base_url = f"http://127.0.0.1:{agent.server_port}"
            opened = self._post_json(
                f"{base_url}/api/sessions/open",
                {"user_id": "alice", "agent_id": "demo-agent", "session_id": "chat-001"},
                headers={"X-Auth-Key": "ek_proxy"},
            )
            self.assertEqual(opened["scope"]["session_id"], "chat-001")
            self.assertEqual(FakeEchoMemoryHandler.calls[0]["headers"].get("X-Auth-Key"), "ek_proxy")

            with urlopen(f"{base_url}/agent/inspect/runtime", timeout=2) as response:
                runtime_payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(runtime_payload["features"]["session_service"], "ready:D03")

            created = self._post_json(f"{base_url}/api/auth/tenants", {})
            self.assertEqual(created["tenant"]["tenant_id"], "tenant_test")

            with urlopen(f"{base_url}/api/sessions/chat-001/commits/archive_001/memories", timeout=2) as response:
                summary = json.loads(response.read().decode("utf-8"))
            self.assertEqual(summary["summary"]["memory_kinds"], ["preference"])
        finally:
            agent.shutdown()
            agent.server_close()
            agent_thread.join(timeout=2)

    def test_agent_chat_uses_memory_retrieval_model_and_assistant_write(self) -> None:
        FakeEchoMemoryHandler.calls = []
        FakeModelHandler.calls = []
        echomem = ThreadingHTTPServer(("127.0.0.1", 0), FakeEchoMemoryHandler)
        model = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelHandler)
        echomem_thread = threading.Thread(target=echomem.serve_forever, daemon=True)
        model_thread = threading.Thread(target=model.serve_forever, daemon=True)
        echomem_thread.start()
        model_thread.start()

        env = {
            "OPENAI_BASE_URL": f"http://127.0.0.1:{model.server_port}/v1",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "fake-chat",
        }
        with patch.dict("os.environ", env, clear=False):
            agent = create_agent_server(port=0, echomem_url=f"http://127.0.0.1:{echomem.server_port}")
        agent_thread = threading.Thread(target=agent.serve_forever, daemon=True)
        agent_thread.start()

        try:
            response = self._post_json(
                f"http://127.0.0.1:{agent.server_port}/agent/chat",
                {
                    "user_id": "alice",
                    "agent_id": "demo-agent",
                    "session_id": "chat-001",
                    "message": "帮我整理 D03 的提交方案",
                },
                headers={"X-Auth-Key": "ek_chat"},
            )
            self.assertEqual(response["assistant"]["content"], "这是模型生成的 D03 提交方案。")
            trace = response["context_trace"]
            self.assertEqual(trace["phase"], "dialogue")
            memory_layers = [layer for layer in trace["layers"] if layer["name"] == "检索记忆"]
            self.assertEqual(memory_layers[0]["source"], "EchoMemory")
            self.assertTrue(memory_layers[0]["highlight"])
            self.assertEqual(memory_layers[0]["item_count"], 1)
            self.assertEqual([call["path"] for call in FakeEchoMemoryHandler.calls], [
                "/api/sessions/open",
                "/agent/inspect/fs/read?uri=echo%3A%2F%2Fsessions%2Fchat-001%2Fcurrent%2Fmessages.jsonl",
                "/api/sessions/chat-001/messages",
                "/api/retrieval/search",
                "/api/sessions/chat-001/messages",
            ])
            self.assertNotIn("user_id", FakeEchoMemoryHandler.calls[0]["payload"])
            self.assertNotIn("user_id", FakeEchoMemoryHandler.calls[3]["payload"])
            self.assertEqual(FakeEchoMemoryHandler.calls[0]["headers"].get("X-Auth-Key"), "ek_chat")
            self.assertEqual(FakeEchoMemoryHandler.calls[3]["headers"].get("X-Auth-Key"), "ek_chat")
            self.assertEqual(FakeModelHandler.calls[0]["path"], "/v1/chat/completions")
            model_payload = FakeModelHandler.calls[0]["payload"]
            self.assertEqual(model_payload["model"], "fake-chat")
            roles = [message["role"] for message in model_payload["messages"]]
            self.assertEqual(roles, ["system", "system", "system", "user", "assistant", "user"])
            joined_context = "\n".join(message["content"] for message in model_payload["messages"])
            self.assertIn("你是 EchoMemory Agent", joined_context)
            self.assertNotIn("<memory_contract>", joined_context)
            self.assertNotIn("session_id: chat-001", joined_context)
            self.assertNotIn("timestamp_utc", joined_context)
            self.assertNotIn("<session_history>", joined_context)
            self.assertIn("<current_request>\n帮我整理 D03 的提交方案\n</current_request>", joined_context)
            self.assertIn('<retrieved_memory source="EchoMemory">', joined_context)
            self.assertIn("用户希望方案简洁", joined_context)
        finally:
            agent.shutdown()
            agent.server_close()
            agent_thread.join(timeout=2)
            echomem.shutdown()
            echomem.server_close()
            echomem_thread.join(timeout=2)
            model.shutdown()
            model.server_close()
            model_thread.join(timeout=2)
            echomem.shutdown()
            echomem.server_close()
            echomem_thread.join(timeout=2)

    def test_agent_context_previews_messages_without_model_or_message_writes(self) -> None:
        FakeEchoMemoryHandler.calls = []
        FakeModelHandler.calls = []
        echomem = ThreadingHTTPServer(("127.0.0.1", 0), FakeEchoMemoryHandler)
        model = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelHandler)
        echomem_thread = threading.Thread(target=echomem.serve_forever, daemon=True)
        model_thread = threading.Thread(target=model.serve_forever, daemon=True)
        echomem_thread.start()
        model_thread.start()

        env = {
            "OPENAI_BASE_URL": f"http://127.0.0.1:{model.server_port}/v1",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "fake-chat",
        }
        with patch.dict("os.environ", env, clear=False):
            agent = create_agent_server(port=0, echomem_url=f"http://127.0.0.1:{echomem.server_port}")
        agent_thread = threading.Thread(target=agent.serve_forever, daemon=True)
        agent_thread.start()

        try:
            response = self._post_json(
                f"http://127.0.0.1:{agent.server_port}/agent/context",
                {
                    "user_id": "alice",
                    "agent_id": "demo-agent",
                    "session_id": "chat-001",
                    "message": "帮我整理 D03 的提交方案",
                },
                headers={"X-Auth-Key": "ek_context"},
            )
            self.assertEqual(response["session_id"], "chat-001")
            self.assertIn("messages", response)
            self.assertIn("context_trace", response)
            self.assertNotIn("assistant", response)
            self.assertEqual(FakeModelHandler.calls, [])
            self.assertEqual([call["path"] for call in FakeEchoMemoryHandler.calls], [
                "/api/sessions/open",
                "/agent/inspect/fs/read?uri=echo%3A%2F%2Fsessions%2Fchat-001%2Fcurrent%2Fmessages.jsonl",
                "/api/retrieval/search",
            ])
            self.assertEqual(FakeEchoMemoryHandler.calls[0]["headers"].get("X-Auth-Key"), "ek_context")
            self.assertEqual(FakeEchoMemoryHandler.calls[2]["headers"].get("X-Auth-Key"), "ek_context")
            joined_context = "\n".join(message["content"] for message in response["messages"])
            self.assertIn("<current_request>\n帮我整理 D03 的提交方案\n</current_request>", joined_context)
            self.assertIn("用户希望方案简洁", joined_context)
        finally:
            agent.shutdown()
            agent.server_close()
            agent_thread.join(timeout=2)
            echomem.shutdown()
            echomem.server_close()
            echomem_thread.join(timeout=2)
            model.shutdown()
            model.server_close()
            model_thread.join(timeout=2)

    def test_agent_chat_tolerates_missing_session_history_file(self) -> None:
        FakeEchoMemoryHandler.calls = []
        FakeModelHandler.calls = []

        class MissingHistoryEchoMemoryHandler(FakeEchoMemoryHandler):
            def do_GET(self) -> None:
                if self.path.startswith("/agent/inspect/fs/read"):
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {
                            "error": "FileSystemError",
                            "message": "Cannot read messages.jsonl: No such file or directory",
                        },
                    )
                    return
                super().do_GET()

        echomem = ThreadingHTTPServer(("127.0.0.1", 0), MissingHistoryEchoMemoryHandler)
        model = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelHandler)
        echomem_thread = threading.Thread(target=echomem.serve_forever, daemon=True)
        model_thread = threading.Thread(target=model.serve_forever, daemon=True)
        echomem_thread.start()
        model_thread.start()
        env = {
            "OPENAI_BASE_URL": f"http://127.0.0.1:{model.server_port}/v1",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "fake-chat",
        }
        with patch.dict("os.environ", env, clear=False):
            agent = create_agent_server(port=0, echomem_url=f"http://127.0.0.1:{echomem.server_port}")
        agent_thread = threading.Thread(target=agent.serve_forever, daemon=True)
        agent_thread.start()

        try:
            response = self._post_json(
                f"http://127.0.0.1:{agent.server_port}/agent/chat",
                {
                    "user_id": "alice",
                    "agent_id": "demo-agent",
                    "session_id": "chat-001",
                    "message": "你好",
                },
            )
            self.assertEqual(response["history"], {"messages": []})
            self.assertEqual(response["assistant"]["content"], "这是模型生成的 D03 提交方案。")
        finally:
            agent.shutdown()
            agent.server_close()
            agent_thread.join(timeout=2)
            echomem.shutdown()
            echomem.server_close()
            echomem_thread.join(timeout=2)
            model.shutdown()
            model.server_close()
            model_thread.join(timeout=2)

    def test_locomo_page_dataset_and_run_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = f"{tmp}/locomo"
            config_path = f"{tmp}/agent.json"
            self._write_locomo_fixture(dataset_dir)
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "locomo": {
                            "data_dir": dataset_dir,
                            "auto_download": False,
                            "run_worker_enabled": False,
                        }
                    },
                    handle,
                )

            agent = create_agent_server(port=0, echomem_url="http://127.0.0.1:1", config_path=config_path)
            agent_thread = threading.Thread(target=agent.serve_forever, daemon=True)
            agent_thread.start()

            try:
                base_url = f"http://127.0.0.1:{agent.server_port}"
                with urlopen(f"{base_url}/agent/locomo", timeout=2) as response:
                    html = response.read().decode("utf-8")
                self.assertEqual(response.status, HTTPStatus.OK)
                self.assertIn("LoCoMo 评测台", html)

                with urlopen(f"{base_url}/agent/locomo/dataset", timeout=2) as response:
                    index = json.loads(response.read().decode("utf-8"))
                self.assertEqual(index["samples"][0]["sample_id"], "sample_1")
                self.assertEqual(index["samples"][0]["qa_count"], 1)

                with urlopen(f"{base_url}/agent/locomo/dataset/sample_1", timeout=2) as response:
                    sample = json.loads(response.read().decode("utf-8"))
                self.assertEqual(sample["sessions"][0]["turns"][0]["dia_id"], "D1:1")
                self.assertEqual(sample["qa"][0]["id"], "qa_001")

                run = self._post_json(
                    f"{base_url}/agent/locomo/runs",
                    {"sample_ids": ["sample_1"], "qa_ids": {"sample_1": ["qa_001"]}},
                )
                self.assertEqual(run["status"], "queued")
                self.assertEqual(run["progress"]["total"], 1)

                with urlopen(f"{base_url}/agent/locomo/runs", timeout=2) as response:
                    runs = json.loads(response.read().decode("utf-8"))
                self.assertEqual(runs["runs"][0]["run_id"], run["run_id"])
            finally:
                agent.shutdown()
                agent.server_close()
                agent_thread.join(timeout=2)

    def _write_locomo_fixture(self, dataset_dir: str) -> None:
        from pathlib import Path

        path = Path(dataset_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "sample_id": "sample_1",
                "conversation": {
                    "speaker_a": "Caroline",
                    "speaker_b": "Melanie",
                    "session_1_date_time": "2023-01-01",
                    "session_1": [
                        {"speaker": "Caroline", "dia_id": "D1:1", "text": "I booked the pottery class."},
                        {"speaker": "Melanie", "dia_id": "D1:2", "text": "That sounds useful."},
                    ],
                },
                "observation": {"session_1_observation": "Caroline booked a pottery class."},
                "session_summary": {"session_1_summary": "They discussed pottery."},
                "event_summary": {"events_session_1": ["Caroline booked a class."]},
                "qa": [
                    {
                        "question": "What class did Caroline book?",
                        "answer": "pottery class",
                        "category": "single-hop",
                        "evidence": ["D1:1"],
                    }
                ],
            }
        ]
        (path / "locomo10.json").write_text(json.dumps(payload), encoding="utf-8")

    def _post_json(
        self,
        url: str,
        payload: dict[str, object],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, HTTPStatus.OK)
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
