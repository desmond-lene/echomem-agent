"""Graph and agent evaluation cases for the standalone agent playground."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .chat_service import AgentChatService
from .config import AgentConfig


class GraphEvalService:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.root_dir = Path(config.locomo.data_dir).parent / "graph_eval"
        self.runs_dir = self.root_dir / "runs"
        self.imports_dir = self.root_dir / "imports"
        self._chenmo_module: ModuleType | None = None
        self._cases_cache: tuple[dict[str, Any], ...] | None = None

    def cases(self) -> dict[str, Any]:
        return {"cases": [self._public_case(case) for case in self._cases()]}

    def import_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        case_id = str(payload.get("case_id") or self._cases()[0]["id"])
        case = self._case(case_id)
        account = self._account_scope(payload, case)
        existing = self._find_import(case_id=case_id, account_id=account["account_id"], tenant_id=account["tenant_id"])
        if existing:
            return {**existing, "already_imported": True}

        import_id = "graph_import_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        session_id = str(payload.get("session_id") or f"{case['session_id']}-{import_id}")
        record = {
            "import_id": import_id,
            "case_id": case_id,
            "case_version": case.get("version", "v1"),
            "account_id": account["account_id"],
            "tenant_id": account["tenant_id"],
            "user_id": account["user_id"],
            "session_id": session_id,
            "status": "running",
            "imported_count": 0,
            "total_count": len(case["dialogue"]),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "dialogue": case["dialogue"],
            "commit": None,
            "error": None,
        }
        self._write_import(record)
        try:
            self._post_memory("/api/sessions/open", {"agent_id": case["agent_id"], "session_id": session_id})
            for turn in case["dialogue"]:
                self._post_memory(
                    f"/api/sessions/{session_id}/messages",
                    {
                        "role": turn["role"],
                        "content": turn["content"],
                        "metadata": {
                            "graph_eval": True,
                            "case_id": case_id,
                            "turn_id": turn["id"],
                            "event_time": turn["timestamp"],
                            "event_date": turn["date"],
                            "eval_mode": case["eval_mode"],
                        },
                    },
                )
                record["imported_count"] += 1
                record["updated_at"] = datetime.now(UTC).isoformat()
                self._write_import(record)
            commit = self._post_memory(
                f"/api/sessions/{session_id}/commit",
                {"metadata": {"graph_eval": True, "case_id": case_id, "import_id": import_id}},
            )
            commit_id = str((commit.get("result") or {}).get("commit_id") or (commit.get("result") or {}).get("archive_id") or "")
            status = self._wait_commit(session_id, commit_id)
            record["status"] = "imported"
            record["commit"] = {"commit_id": commit_id, "status": status}
            record["updated_at"] = datetime.now(UTC).isoformat()
            self._write_import(record)
            return record
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            record["updated_at"] = datetime.now(UTC).isoformat()
            self._write_import(record)
            raise

    def list_imports(self) -> dict[str, Any]:
        self.imports_dir.mkdir(parents=True, exist_ok=True)
        imports = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.imports_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        ]
        return {"imports": imports[:50]}

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        case_id = str(payload.get("case_id") or self._cases()[0]["id"])
        case = self._case(case_id)
        import_id = str(payload.get("import_id") or "")
        if not import_id:
            raise ValueError("please import the historical session before running graph evaluation")
        import_record = self._load_import(import_id)
        if import_record.get("status") != "imported":
            raise ValueError(f"case must be imported successfully before evaluation: {import_record.get('status')}")
        qa_ids = {str(item) for item in payload.get("qa_ids") or () if str(item)}
        queries = [query for query in case["queries"] if not qa_ids or query["id"] in qa_ids]
        if not queries:
            raise ValueError("at least one QA pair must be selected")

        run_id = "graph_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        results = self._execute_queries(case, import_record, queries)
        passed = all(item["passed"] for item in results)
        run = {
            "run_id": run_id,
            "case_id": case_id,
            "case_title": case["title"],
            "import_id": import_record["import_id"],
            "account_id": import_record.get("account_id"),
            "tenant_id": import_record["tenant_id"],
            "session_id": import_record["session_id"],
            "status": "passed" if passed else "failed",
            "summary": {
                "total": len(results),
                "passed": sum(1 for item in results if item["passed"]),
                "failed": sum(1 for item in results if not item["passed"]),
            },
            "results": results,
        }
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        (self.runs_dir / f"{run_id}.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        return run

    def list_runs(self) -> dict[str, Any]:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        runs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.runs_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        ]
        return {"runs": runs[:50]}

    def _cases(self) -> tuple[dict[str, Any], ...]:
        if self._cases_cache is None:
            self._cases_cache = (self._simple_case(), self._chenmo_case())
        return self._cases_cache

    def _case(self, case_id: str) -> dict[str, Any]:
        for case in self._cases():
            if case["id"] == case_id:
                return case
        raise ValueError(f"unknown graph eval case: {case_id}")

    def _public_case(self, case: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": case["id"],
            "title": case["title"],
            "description": case.get("description", ""),
            "tenant_id": case["tenant_id"],
            "query_count": len(case["queries"]),
            "ingest_count": len(case["dialogue"]),
            "eval_mode": case["eval_mode"],
            "dialogue": [
                {
                    "id": turn["id"],
                    "role": turn["role"],
                    "date": turn["date"],
                    "timestamp": turn["timestamp"],
                    "content": turn["content"],
                }
                for turn in case["dialogue"]
            ],
            "queries": [
                {
                    "id": query["id"],
                    "name": query["name"],
                    "query": query["query"],
                    "number": query["number"],
                    "question_id": query.get("question_id", query["id"]),
                    "section": query.get("section", ""),
                    "expected": query.get("expected", ""),
                }
                for query in case["queries"]
            ],
        }

    def _simple_case(self) -> dict[str, Any]:
        return {
            "id": "mentor-travel-temporal-task-tenant",
            "version": "simple-scenario-v2",
            "title": "Simple Scenario: Mentor and Travel",
            "description": "A short regression case covering current state, historical state, task memory, and tenant isolation.",
            "tenant_id": "tenant_user_001",
            "user_id": "xiaoming",
            "agent_id": "graph-eval-agent",
            "session_id": "graph-eval-session",
            "eval_mode": "retrieval",
            "dialogue": [
                {
                    "id": "turn_20240501",
                    "timestamp": 1714521600,
                    "date": "2024-05-01",
                    "role": "user",
                    "content": "\u6211\u65b0\u5165\u804c\u4e86 A \u516c\u53f8\uff0c\u5e26\u6211\u7684\u5bfc\u5e08\u662f\u5f20\u4e09\u3002\u53e6\u5916\uff0c\u6211\u60f3\u5236\u5b9a\u4e00\u4e2a\u4e94\u4e00\u53bb\u5317\u4eac\u73a9\u7684\u8ba1\u5212\uff0c\u5e2e\u6211\u770b\u770b\u6545\u5bab\u7684\u95e8\u7968\u3002",
                },
                {
                    "id": "turn_20240511",
                    "timestamp": 1715385600,
                    "date": "2024-05-11",
                    "role": "user",
                    "content": "\u8ba1\u5212\u6709\u53d8\uff0c\u5f20\u4e09\u4eca\u5929\u7a81\u7136\u79bb\u804c\u4e86\uff0c\u65b0\u5e26\u6211\u7684\u5bfc\u5e08\u6362\u6210\u4e86\u674e\u56db\u3002\u5bf9\u4e86\uff0c\u5317\u4eac\u90a3\u8fb9\u6211\u8ba2\u597d\u4e86\u9152\u5e97\uff0c\u53d1\u4f60\u786e\u8ba4\u5355\u3002",
                },
                {
                    "id": "turn_20240518",
                    "timestamp": 1715990400,
                    "date": "2024-05-18",
                    "role": "user",
                    "content": "\u674e\u56db\u4eca\u5929\u5e26\u6211\u505a\u7684\u65b0\u9879\u76ee\u4e0a\u7ebf\u4e86\uff0c\u7279\u522b\u987a\u5229\uff01",
                },
            ],
            "queries": [
                {
                    "number": 1,
                    "id": "current_mentor",
                    "name": "Current Mentor",
                    "query": "\u6211\u73b0\u5728\u7684\u5bfc\u5e08\u662f\u8c01\uff1f",
                    "must": ["\u674e\u56db", "\u5bfc\u5e08"],
                    "must_not": ["\u5f20\u4e09\u662f\u6211\u73b0\u5728\u7684\u5bfc\u5e08"],
                },
                {
                    "number": 2,
                    "id": "historical_mentor",
                    "name": "Historical Mentor",
                    "query": "\u5e2e\u6211\u67e5\u4e00\u4e0b\uff0c\u6211\u5728 5 \u6708 6 \u53f7\u90a3\u5929\u7684\u5bfc\u5e08\u662f\u8c01\uff1f",
                    "target_time": 1714953600,
                    "must": ["\u5f20\u4e09", "\u5bfc\u5e08"],
                    "must_not": ["\u674e\u56db", "\u65b0\u9879\u76ee\u4e0a\u7ebf"],
                },
                {
                    "number": 3,
                    "id": "travel_task",
                    "name": "Travel Task Memory",
                    "query": "\u5173\u4e8e\u6211\u8fd9\u6b21\u65c5\u6e38\uff0c\u6211\u4eec\u90fd\u804a\u4e86\u4ec0\u4e48\uff1f",
                    "task_id": "\u4e94\u4e00\u5317\u4eac\u65c5\u6e38\u89c4\u5212",
                    "must": ["\u5317\u4eac", "\u6545\u5bab", "\u9152\u5e97"],
                    "must_not": ["\u65b0\u9879\u76ee\u4e0a\u7ebf", "\u674e\u56db"],
                },
                {
                    "number": 4,
                    "id": "tenant_isolation",
                    "name": "Tenant Isolation",
                    "query": "\u6211\u73b0\u5728\u7684\u5bfc\u5e08\u662f\u8c01\uff1f",
                    "tenant_id": "tenant_other_001",
                    "must_not": ["\u674e\u56db", "\u5f20\u4e09", "\u5bfc\u5e08"],
                },
            ],
        }

    def _chenmo_case(self) -> dict[str, Any]:
        module = self._chenmo_eval_module()
        root = Path(__file__).resolve().parents[2]
        scenario_path = root / "docs" / "design" / "agent" / "evaluation_scenario.md"
        scenario = module.parse_scenario(scenario_path)
        questions = module.parse_questions(scenario_path)
        turns = module.build_turns(scenario)
        dialogue = [
            {
                "id": f"turn_{index + 1:02d}",
                "timestamp": turn.timestamp,
                "date": datetime.fromtimestamp(turn.timestamp, UTC).date().isoformat(),
                "role": turn.role,
                "content": turn.content,
            }
            for index, turn in enumerate(turns)
        ]
        queries = [
            {
                "number": index + 1,
                "id": question.question_id,
                "question_id": question.question_id,
                "name": f"{question.question_id} {question.section}",
                "query": question.prompt,
                "expected": question.expected,
                "reasoning": question.reasoning,
                "section": question.section,
            }
            for index, question in enumerate(questions)
        ]
        return {
            "id": "chenmo-long-horizon",
            "version": "chenmo-v1",
            "title": "Chenmo Long-Horizon Evaluation",
            "description": "Long-horizon memory evaluation with temporal, multi-hop, causal, complex-task, and comprehensive questions.",
            "tenant_id": "tenant_chenmo_eval",
            "user_id": "chenmo_user",
            "agent_id": "chenmo-eval-agent",
            "session_id": "chenmo-eval-session",
            "eval_mode": "agent_answer",
            "dialogue": dialogue,
            "queries": queries,
        }

    def _execute_queries(self, case: dict[str, Any], import_record: dict[str, Any], queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if case["eval_mode"] == "agent_answer":
            return self._execute_agent_answer_queries(case, import_record, queries)
        return self._execute_retrieval_queries(case, import_record, queries)

    def _execute_retrieval_queries(self, case: dict[str, Any], import_record: dict[str, Any], queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for query in queries:
            if query.get("tenant_id") and query.get("tenant_id") != import_record.get("tenant_id"):
                context = ""
                items: list[dict[str, Any]] = []
            else:
                response = self._post_memory(
                    "/api/retrieval/search",
                    {
                        "query": query["query"],
                        "agent_id": case["agent_id"],
                        "session_id": import_record["session_id"],
                        "task_id": query.get("task_id"),
                        "target_time": query.get("target_time"),
                        "limit": 8,
                        "include_explain": True,
                    },
                )
                result = response.get("result") if isinstance(response.get("result"), dict) else response
                raw_items = result.get("items") if isinstance(result, dict) else []
                items = [dict(item) for item in raw_items if isinstance(item, dict)]
                relation_items = [item for item in items if str(item.get("kind") or "") == "relation"]
                display_items = items if query.get("task_id") else relation_items or items
                context = "\n".join(str(item.get("text") or item.get("content") or "") for item in display_items)
            failures = [f"missing:{text}" for text in query.get("must", ()) if text not in context]
            failures.extend(f"forbidden:{text}" for text in query.get("must_not", ()) if text in context)
            results.append(
                {
                    "number": query["number"],
                    "id": query["id"],
                    "name": query["name"],
                    "question": query["query"],
                    "passed": not failures,
                    "failures": failures,
                    "context": context,
                    "items": items,
                }
            )
        return results

    def _execute_agent_answer_queries(self, case: dict[str, Any], import_record: dict[str, Any], queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        module = self._chenmo_eval_module()
        chat_service = AgentChatService(self.config)
        results = []
        for query in queries:
            qa_session_id = f"{import_record['session_id']}-qa-{query['number']:02d}-{query['id'].lower()}"
            retrieval = self._post_memory(
                "/api/retrieval/search",
                {
                    "query": query["query"],
                    "agent_id": case["agent_id"],
                    "session_id": qa_session_id,
                    "limit": 8,
                    "include_explain": True,
                },
            )
            response = chat_service.chat(
                {
                    "user_id": import_record["user_id"],
                    "agent_id": case["agent_id"],
                    "session_id": qa_session_id,
                    "message": query["query"],
                    "include_history": False,
                }
            )
            answer = str((response.get("assistant") or {}).get("content") or "")
            question_obj = SimpleNamespace(
                section=query["section"],
                question_id=query["question_id"],
                prompt=query["query"],
                expected=query["expected"],
                reasoning=query["reasoning"],
            )
            judgment = module.judge_answer(question_obj, answer)
            retrieval_failure = module.classify_retrieval(retrieval)
            failures = list(judgment.get("missing_points") or [])
            failures.extend(f"contradiction:{item}" for item in judgment.get("contradictions") or [])
            results.append(
                {
                    "number": query["number"],
                    "id": query["id"],
                    "name": query["name"],
                    "section": query["section"],
                    "question": query["query"],
                    "expected": query["expected"],
                    "reasoning": query["reasoning"],
                    "passed": bool(judgment.get("passed")),
                    "score": float(judgment.get("score") or 0.0),
                    "failures": failures,
                    "answer": answer,
                    "retrieval_failure": retrieval_failure,
                    "context": "\n".join(self._item_texts(retrieval)[:8]),
                    "items": self._extract_items(retrieval),
                }
            )
        return results

    def _find_import(self, *, case_id: str, account_id: str, tenant_id: str) -> dict[str, Any] | None:
        self.imports_dir.mkdir(parents=True, exist_ok=True)
        case = self._case(case_id)
        for path in sorted(self.imports_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            data = json.loads(path.read_text(encoding="utf-8"))
            if (
                data.get("case_id") == case_id
                and data.get("case_version") == case.get("version", "v1")
                and data.get("account_id") == account_id
                and data.get("tenant_id") == tenant_id
                and data.get("status") == "imported"
            ):
                return data
        return None

    def _load_import(self, import_id: str) -> dict[str, Any]:
        path = self.imports_dir / f"{import_id}.json"
        if not import_id or not path.exists():
            raise ValueError(f"unknown graph eval import: {import_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid graph eval import: {import_id}")
        return data

    def _write_import(self, record: dict[str, Any]) -> None:
        self.imports_dir.mkdir(parents=True, exist_ok=True)
        (self.imports_dir / f"{record['import_id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _wait_commit(self, session_id: str, commit_id: str) -> dict[str, Any]:
        if not commit_id:
            raise ValueError("commit response did not include commit_id")
        deadline = time.monotonic() + max(1800.0, float(self.config.locomo.commit_wait_seconds))
        last_status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            response = self._get_memory(f"/api/sessions/{session_id}/commits/{commit_id}")
            status = response.get("status") if isinstance(response.get("status"), dict) else {}
            last_status = dict(status)
            value = str(status.get("status") or "")
            if value == "completed":
                return last_status
            if value == "failed":
                raise RuntimeError(str(status.get("error") or "commit failed"))
            time.sleep(max(0.2, float(self.config.locomo.commit_poll_seconds)))
        raise TimeoutError(f"commit timeout: {commit_id}; last_status={last_status}")

    def _account_scope(self, payload: dict[str, Any], case: dict[str, Any]) -> dict[str, str]:
        tenant_id = str(payload.get("tenant_id") or case["tenant_id"]).strip() or case["tenant_id"]
        user_id = str(payload.get("user_id") or case["user_id"]).strip() or case["user_id"]
        account_id = str(payload.get("account_id") or tenant_id).strip() or tenant_id
        return {"tenant_id": tenant_id, "user_id": user_id, "account_id": account_id}

    def _extract_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
        raw_items = result.get("items") if isinstance(result, dict) else []
        return [dict(item) for item in raw_items if isinstance(item, dict)]

    def _item_texts(self, payload: dict[str, Any]) -> list[str]:
        return [str(item.get("text") or item.get("content") or "") for item in self._extract_items(payload) if str(item.get("text") or item.get("content") or "").strip()]

    def _chenmo_eval_module(self) -> ModuleType:
        if self._chenmo_module is not None:
            return self._chenmo_module
        root = Path(__file__).resolve().parents[2]
        path = root / "scripts" / "run_chenmo_eval.py"
        spec = importlib.util.spec_from_file_location("run_chenmo_eval_module", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load chenmo evaluation module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._chenmo_module = module
        return module

    def _get_memory(self, path: str) -> dict[str, Any]:
        request = Request(
            self.config.echomemory.base_url.rstrip("/") + path,
            method="GET",
            headers=self._headers(),
        )
        return self._read_json_response(request)

    def _post_memory(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.config.echomemory.base_url.rstrip("/") + path,
            data=body,
            method="POST",
            headers=self._headers({"Content-Type": "application/json; charset=utf-8"}),
        )
        return self._read_json_response(request)

    def _headers(self, base: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(base or {})
        if self.config.echomemory.auth_key:
            headers["X-Auth-Key"] = self.config.echomemory.auth_key
        return headers

    def _read_json_response(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=max(self.config.echomemory.timeout_seconds, 60)) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            exc.close()
            raise RuntimeError(f"echomemory_http_{exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"echomemory_unreachable: {exc.reason}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("memory service response must be a JSON object")
        return data
