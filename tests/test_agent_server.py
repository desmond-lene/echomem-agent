"""Agent playground server tests."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from urllib.parse import urlparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

from agent.config import ChatConfig
from agent.context_builder import ContextBuilder
from agent.server import create_server as create_agent_server


class FakeEchoMemoryHandler(BaseHTTPRequestHandler):
    """Tiny EchoMemory-compatible HTTP double."""

    calls: list[dict[str, object]] = []

    def do_GET(self) -> None:
        self.calls.append({"method": "GET", "path": self.path, "payload": {}, "headers": dict(self.headers)})
        parsed = urlparse(self.path)
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
        if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/commits/archive_001"):
            self._send_json(HTTPStatus.OK, {"status": {"status": "completed", "commit_id": "archive_001"}})
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
                {"scope": {"session_id": payload.get("session_id") or "chat-001"}},
            )
            return
        if self.path == "/api/sessions/chat-001/messages" or (
            self.path.startswith("/api/sessions/") and self.path.endswith("/messages")
        ):
            self._send_json(
                HTTPStatus.OK,
                {"message": {"id": "msg_001", "role": payload.get("role"), "content": payload.get("content")}},
            )
            return
        if self.path.startswith("/api/sessions/") and self.path.endswith("/commit"):
            self._send_json(
                HTTPStatus.OK,
                {"result": {"commit_id": "archive_001", "archive_id": "archive_001", "status": "pending"}},
            )
            return
        if self.path == "/api/retrieval/search":
            query = str(payload.get("query") or "")
            task_id = payload.get("task_id")
            target_time = payload.get("target_time")
            if task_id in {"Task_02", "五一北京旅游规划"}:
                items = [
                    {
                        "kind": "relation",
                        "text": "Relation 小明 -[旅游计划]-> 北京。小明计划五一去北京旅游并查看故宫的门票，后来订好了酒店。",
                        "score": 0.94,
                    },
                    {
                        "kind": "episode",
                        "text": "小明订好了北京酒店，并发送确认单。",
                        "score": 0.91,
                    },
                ]
            elif target_time == 1714953600:
                items = [
                    {
                        "kind": "relation",
                        "text": "Relation 张三 -[导师]-> 小明。张三是小明的导师。",
                        "score": 0.96,
                    }
                ]
            elif "导师" in query:
                items = [
                    {
                        "kind": "relation",
                        "text": "Relation 李四 -[导师]-> 小明。李四是小明的导师。",
                        "score": 0.98,
                    }
                ]
            else:
                items = [
                    {
                        "kind": "preference",
                        "content": "用户希望方案简洁，优先列步骤。",
                        "score": 0.91,
                    }
                ]
            self._send_json(
                HTTPStatus.OK,
                {
                    "items": items,
                    "result": {"items": items, "explain": {"strategy": "fake"}},
                    "explain": {"strategy": "fake"},
                },
            )
            return
        if self.path == "/api/v1/memory/reset":
            self._send_json(HTTPStatus.OK, {"status": "reset", "tenant_id": self.headers.get("X-Tenant-ID")})
            return
        if self.path == "/api/v1/memory":
            self._send_json(HTTPStatus.OK, {"status": "ok", "result": {"projection": {"node_count": 1}}})
            return
        if self.path == "/api/v1/memory/search":
            query = str(payload.get("query") or "")
            task_id = payload.get("task_id")
            target_time = payload.get("target_time")
            if self.headers.get("X-Tenant-ID") == "tenant_other_001":
                context = "涉及实体: []。实体状态变更: []"
            elif task_id == "Task_02":
                context = "涉及实体: []。实体状态变更: []\nEpisode: 故宫的门票\nEpisode: 小明订好了北京酒店，并发送确认单。"
            elif target_time == 1714953600:
                context = "涉及实体: ['张三']。实体状态变更: ['导师']\nRelation 张三 -[导师]-> 小明。张三是小明的导师"
            elif "导师" in query:
                context = "涉及实体: ['李四']。实体状态变更: ['导师']\nRelation 李四 -[导师]-> 小明。李四是小明的导师"
            else:
                context = "涉及实体: []。实体状态变更: []"
            self._send_json(HTTPStatus.OK, {"status": "ok", "result": {"items": [], "context_prompt_assembly": context}})
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
    def test_context_keeps_empty_retrieved_memory_section(self) -> None:
        result = ContextBuilder(ChatConfig(system_prompt="你是测试 Agent")).build_with_trace(
            user_id="alice",
            agent_id="demo-agent",
            session_id="chat-001",
            user_message="张三的导师是谁",
            history={"messages": []},
            retrieval={"items": []},
        )

        joined_context = "\n".join(message["content"] for message in result.messages)
        self.assertIn('<retrieved_memory source="EchoMemory">', joined_context)
        self.assertIn("## Retrieved Memory", joined_context)
        memory_layers = [layer for layer in result.trace["layers"] if layer["name"] == "Retrieved Memory"]
        self.assertEqual(memory_layers[0]["item_count"], 0)
        self.assertTrue(memory_layers[0]["highlight"])

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
            self.assertIn("includeHistory", html)
            self.assertIn("Conversation Tail", html)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_graph_eval_page_and_run_api(self) -> None:
        FakeEchoMemoryHandler.calls = []
        echomem = ThreadingHTTPServer(("127.0.0.1", 0), FakeEchoMemoryHandler)
        echomem_thread = threading.Thread(target=echomem.serve_forever, daemon=True)
        echomem_thread.start()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = f"{tmp}/agent.json"
            Path(config_path).write_text(
                json.dumps({"locomo": {"data_dir": f"{tmp}/locomo", "auto_download": False}}),
                encoding="utf-8",
            )
            agent = create_agent_server(
                port=0,
                echomem_url=f"http://127.0.0.1:{echomem.server_port}",
                config_path=config_path,
            )
            agent_thread = threading.Thread(target=agent.serve_forever, daemon=True)
            agent_thread.start()

            try:
                base_url = f"http://127.0.0.1:{agent.server_port}"
                with urlopen(f"{base_url}/agent/graph-eval", timeout=2) as response:
                    html = response.read().decode("utf-8")
                self.assertEqual(response.status, HTTPStatus.OK)
                self.assertIn("Graph 评测", html)
                self.assertIn("accountSelect", html)
                self.assertIn("对话导入", html)
                self.assertIn("历史对话预览", html)
                self.assertIn("导入历史会话", html)
                self.assertIn("本次评测返回", html)
                self.assertIn("QA 对选择", html)
                self.assertNotIn("ensureAccount", html)

                with urlopen(f"{base_url}/agent/graph-eval/cases", timeout=2) as response:
                    cases = json.loads(response.read().decode("utf-8"))
                self.assertEqual(cases["cases"][0]["id"], "mentor-travel-temporal-task-tenant")
                self.assertEqual(cases["cases"][0]["queries"][0]["id"], "current_mentor")
                self.assertEqual(
                    cases["cases"][0]["dialogue"][0]["content"],
                    "2024-05-01，我新入职了 A 公司，带我的导师是张三。另外，我想制定一个五一去北京玩的计划，帮我看看故宫的门票。",
                )
                self.assertEqual(
                    cases["cases"][0]["dialogue"][1]["content"],
                    "2024-05-11，计划有变，张三今天突然离职了，新带我的导师换成了李四。对了，北京那边我订好了酒店，发你确认单。",
                )
                self.assertEqual(
                    cases["cases"][0]["dialogue"][2]["content"],
                    "2024-05-18， 李四今天带我做的新项目上线了，特别顺利！",
                )
                self.assertNotIn("task_id", cases["cases"][0]["queries"][2])
                self.assertNotIn("target_time", cases["cases"][0]["queries"][1])

                imported = self._post_json(
                    f"{base_url}/agent/graph-eval/imports",
                    {
                        "case_id": "mentor-travel-temporal-task-tenant",
                        "account_id": "tenant_test",
                        "tenant_id": "tenant_test",
                        "user_id": "user_test",
                    },
                )
                self.assertEqual(imported["status"], "imported")
                self.assertEqual(imported["imported_count"], 3)
                self.assertEqual(imported["tenant_id"], "tenant_test")

                duplicate = self._post_json(
                    f"{base_url}/agent/graph-eval/imports",
                    {
                        "case_id": "mentor-travel-temporal-task-tenant",
                        "account_id": "tenant_test",
                        "tenant_id": "tenant_test",
                        "user_id": "user_test",
                    },
                )
                self.assertTrue(duplicate["already_imported"])
                self.assertEqual(duplicate["import_id"], imported["import_id"])

                run = self._post_json(
                    f"{base_url}/agent/graph-eval/runs",
                    {
                        "case_id": "mentor-travel-temporal-task-tenant",
                        "import_id": imported["import_id"],
                        "qa_ids": ["current_mentor", "historical_mentor", "travel_task"],
                    },
                )
                self.assertEqual(run["status"], "passed")
                self.assertEqual(len(run["results"]), 3)
                self.assertEqual(run["tenant_id"], "tenant_test")
                self.assertIn("李四是小明的导师", run["results"][0]["context"])

                paths = [str(call["path"]) for call in FakeEchoMemoryHandler.calls if call["method"] == "POST"]
                self.assertIn("/api/sessions/open", paths)
                self.assertTrue(any(path.startswith("/api/sessions/") and path.endswith("/messages") for path in paths))
                self.assertTrue(any(path.startswith("/api/sessions/") and path.endswith("/commit") for path in paths))
                self.assertIn("/api/retrieval/search", paths)
                self.assertFalse(any(path.startswith("/api/v1/memory") for path in paths))
            finally:
                agent.shutdown()
                agent.server_close()
                agent_thread.join(timeout=2)
                echomem.shutdown()
                echomem.server_close()
                echomem_thread.join(timeout=2)

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

    def test_agent_account_ensure_refreshes_stale_auth_key(self) -> None:
        class StaleKeyEchoMemoryHandler(FakeEchoMemoryHandler):
            def do_GET(self) -> None:
                if self.path == "/agent/inspect/events" and self.headers.get("X-Auth-Key") == "ek_stale":
                    self.calls.append({"method": "GET", "path": self.path, "payload": {}, "headers": dict(self.headers)})
                    self._send_json(
                        HTTPStatus.UNAUTHORIZED,
                        {"error": "AuthError", "message": "invalid X-Auth-Key"},
                    )
                    return
                super().do_GET()

        FakeEchoMemoryHandler.calls = []
        echomem = ThreadingHTTPServer(("127.0.0.1", 0), StaleKeyEchoMemoryHandler)
        echomem_thread = threading.Thread(target=echomem.serve_forever, daemon=True)
        echomem_thread.start()

        with tempfile.TemporaryDirectory() as tmp:
            config_path = f"{tmp}/agent.json"
            accounts_path = f"{tmp}/locomo/accounts.json"
            Path(accounts_path).parent.mkdir(parents=True, exist_ok=True)
            Path(accounts_path).write_text(
                json.dumps(
                    {
                        "stale": {
                            "id": "tenant_old",
                            "label": "stale",
                            "tenantId": "tenant_old",
                            "userId": "user_old",
                            "authKey": "ek_stale",
                        }
                    }
                ),
                encoding="utf-8",
            )
            Path(config_path).write_text(json.dumps({"locomo": {"data_dir": f"{tmp}/locomo"}}), encoding="utf-8")
            agent = create_agent_server(
                port=0,
                echomem_url=f"http://127.0.0.1:{echomem.server_port}",
                config_path=config_path,
            )
            agent_thread = threading.Thread(target=agent.serve_forever, daemon=True)
            agent_thread.start()
            try:
                data = self._post_json(f"http://127.0.0.1:{agent.server_port}/agent/accounts/ensure", {"label": "stale"})
                self.assertTrue(data["created"])
                self.assertEqual(data["account"]["authKey"], "ek_test")
                registry = json.loads(Path(accounts_path).read_text(encoding="utf-8"))
                self.assertEqual(registry["stale"]["authKey"], "ek_test")
            finally:
                agent.shutdown()
                agent.server_close()
                agent_thread.join(timeout=2)
                echomem.shutdown()
                echomem.server_close()
                echomem_thread.join(timeout=2)

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
            memory_layers = [layer for layer in trace["layers"] if layer["name"] == "Retrieved Memory"]
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
            self.assertIn("## Retrieved Memory", joined_context)
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

    def test_agent_context_can_exclude_conversation_tail(self) -> None:
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
                    "include_history": False,
                },
                headers={"X-Auth-Key": "ek_context"},
            )
            self.assertEqual(response["history"], {"messages": []})
            self.assertEqual([call["path"] for call in FakeEchoMemoryHandler.calls], [
                "/api/sessions/open",
                "/api/retrieval/search",
            ])
            roles = [message["role"] for message in response["messages"]]
            self.assertEqual(roles, ["system", "system", "system", "user"])
            joined_context = "\n".join(message["content"] for message in response["messages"])
            self.assertIn("## Retrieved Memory", joined_context)
            self.assertNotIn("你好，我可以帮你整理方案。", joined_context)
            history_layers = [
                layer for layer in response["context_trace"]["layers"] if layer["name"] == "近期对话"
            ]
            self.assertEqual(history_layers[0]["enabled"], False)
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

                imported = self._post_json(
                    f"{base_url}/agent/locomo/imports",
                    {"sample_ids": ["sample_1"], "dry_run": True, "commit_wait_seconds": 1, "commit_poll_seconds": 0.1},
                )
                self.assertEqual(imported["imports"][0]["status"], "imported")

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
