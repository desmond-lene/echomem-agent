"""LoCoMo evaluation runs for the standalone agent playground."""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .chat_service import AgentChatService
from .config import AgentConfig
from .echomemory_client import EchoMemoryClientError
from .locomo_dataset import LocomoDatasetError, LocomoDatasetService
from .model_client import ModelClientError


class LocomoEvalError(RuntimeError):
    """Raised when a LoCoMo evaluation run cannot be created or read."""


class LocomoEvalService:
    _lock = threading.Lock()
    _workers: dict[str, threading.Thread] = {}

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.dataset = LocomoDatasetService(config.locomo)
        self.runs_dir = Path(config.locomo.data_dir) / "runs"
        self.imports_dir = Path(config.locomo.data_dir) / "imports"

    def import_status(self) -> dict[str, Any]:
        self.imports_dir.mkdir(parents=True, exist_ok=True)
        imports = []
        for path in sorted(self.imports_dir.glob("*.json")):
            try:
                imports.append(self._read_json(path))
            except LocomoEvalError:
                continue
        return {"imports": imports, "by_sample": {str(item.get("sample_id")): item for item in imports if item.get("sample_id")}}

    def import_samples(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_ids = _string_list(payload.get("sample_ids"))
        if not sample_ids:
            raise LocomoEvalError("sample_ids is required")
        samples = self.dataset.samples_for(sample_ids)
        requested_sessions = payload.get("session_ids") if isinstance(payload.get("session_ids"), dict) else {}
        results = []
        for sample in samples:
            session_ids = _string_list(requested_sessions.get(sample["sample_id"])) if requested_sessions else []
            results.append(self._import_sample(sample, session_ids, payload))
        return {"imports": results}

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_ids = _string_list(payload.get("sample_ids"))
        if not sample_ids:
            raise LocomoEvalError("sample_ids is required")
        selected_samples = self.dataset.samples_for(sample_ids)
        missing = [sample["sample_id"] for sample in selected_samples if not self._import_record(sample["sample_id"]).get("sessions")]
        if missing:
            raise LocomoEvalError(f"samples must be imported before evaluation: {', '.join(missing)}")
        selected = self._select_items(selected_samples, payload.get("qa_ids"))
        if not selected:
            raise LocomoEvalError("selected run has no QA items")

        run_id = _new_run_id()
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run = {
            "run_id": run_id,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "config": {
                "sample_ids": sample_ids,
                "qa_ids": payload.get("qa_ids", "all"),
                "scorers": _string_list(payload.get("scorers")) or ["token_f1", "contains"],
                "dry_run": bool(payload.get("dry_run", False)),
                "session_prefix": str(payload.get("session_prefix") or "locomo-eval"),
                "commit_after_replay": bool(payload.get("commit_after_replay", True)),
                "commit_wait_seconds": float(
                    payload.get("commit_wait_seconds") or self.config.locomo.commit_wait_seconds
                ),
                "commit_poll_seconds": float(
                    payload.get("commit_poll_seconds") or self.config.locomo.commit_poll_seconds
                ),
                "user_id": str(payload.get("user_id") or "locomo-user"),
                "agent_id": str(payload.get("agent_id") or "locomo-agent"),
            },
            "progress": {"total": len(selected), "completed": 0, "current": None},
            "selected": selected,
            "results": [],
            "summary": {},
        }
        self._write_json(run_dir / "run.json", run)
        self._write_json(run_dir / "results.json", {"run_id": run_id, "results": [], "summary": {}})
        self._event(run_dir, "queued", {"selected_count": len(selected)})

        if self.config.locomo.run_worker_enabled:
            worker = threading.Thread(target=self._run_worker, args=(run_id,), daemon=True)
            with self._lock:
                self._workers[run_id] = worker
            worker.start()
        return self.get_run(run_id)

    def list_runs(self) -> dict[str, Any]:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        runs = []
        for path in sorted(self.runs_dir.glob("*/run.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                run = self._read_json(path)
            except LocomoEvalError:
                continue
            runs.append(
                {
                    "run_id": run.get("run_id"),
                    "status": run.get("status"),
                    "created_at": run.get("created_at"),
                    "finished_at": run.get("finished_at"),
                    "progress": run.get("progress"),
                    "summary": run.get("summary") or {},
                    "config": run.get("config") or {},
                }
            )
        return {"runs": runs}

    def get_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        run = self._read_json(run_dir / "run.json")
        results = self._read_json(run_dir / "results.json") if (run_dir / "results.json").exists() else {}
        return {**run, "results": results.get("results", run.get("results", []))}

    def get_events(self, run_id: str) -> dict[str, Any]:
        events_path = self._run_dir(run_id) / "events.jsonl"
        events = []
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    events.append(item)
        return {"run_id": run_id, "events": events}

    def _run_worker(self, run_id: str) -> None:
        run_dir = self._run_dir(run_id)
        try:
            run = self._read_json(run_dir / "run.json")
            self._update_run(run_dir, run, status="running", started_at=_now())
            service = AgentChatService(self.config)

            sample_cache = {sample["sample_id"]: sample for sample in self.dataset.samples_for(run["config"]["sample_ids"])}
            replayed: set[str] = set()
            results = []
            for selected in run["selected"]:
                sample = sample_cache[selected["sample_id"]]
                session_id = self._import_session_id(selected["sample_id"])
                if selected["sample_id"] not in replayed:
                    self._event(run_dir, "sample_import_reused", {"sample_id": sample["sample_id"], "session_id": session_id})
                    replayed.add(selected["sample_id"])
                result = self._answer_qa(run_dir, service, run, sample, selected, session_id)
                results.append(result)
                run["results"] = results
                run["progress"] = {
                    "total": len(run["selected"]),
                    "completed": len(results),
                    "current": selected,
                }
                run["summary"] = _summarize(results)
                self._write_results(run_dir, run)
                self._write_json(run_dir / "run.json", run)

            self._update_run(
                run_dir,
                run,
                status="completed",
                finished_at=_now(),
                progress={"total": len(run["selected"]), "completed": len(results), "current": None},
                summary=_summarize(results),
            )
            self._event(run_dir, "completed", {"summary": _summarize(results)})
            with self._lock:
                self._workers.pop(run_id, None)
        except Exception as exc:  # noqa: BLE001 - background worker must persist failures.
            try:
                run = self._read_json(run_dir / "run.json")
                self._update_run(run_dir, run, status="failed", finished_at=_now(), error=str(exc))
                self._event(run_dir, "failed", {"error": str(exc)})
            finally:
                with self._lock:
                    self._workers.pop(run_id, None)

    def _import_sample(self, sample: dict[str, Any], session_ids: list[str], payload: dict[str, Any]) -> dict[str, Any]:
        sample_id = sample["sample_id"]
        record = self._import_record(sample_id)
        imported_sessions = set(record.get("sessions") or [])
        available = [session["id"] for session in sample["sessions"]]
        selected_sessions = session_ids or available
        selected_sessions = [item for item in selected_sessions if item in set(available)]
        pending_sessions = [item for item in selected_sessions if item not in imported_sessions]
        session_id = self._import_session_id(sample_id)
        if not pending_sessions and record.get("imported"):
            return record

        import_id = f"import_{sample_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"
        import_dir = self.imports_dir / import_id
        import_dir.mkdir(parents=True, exist_ok=True)
        import_run = {
            "run_id": import_id,
            "config": {
                "user_id": str(payload.get("user_id") or "locomo-user"),
                "agent_id": str(payload.get("agent_id") or "locomo-agent"),
                "commit_after_replay": bool(payload.get("commit_after_replay", True)),
                "commit_wait_seconds": float(payload.get("commit_wait_seconds") or self.config.locomo.commit_wait_seconds),
                "commit_poll_seconds": float(payload.get("commit_poll_seconds") or self.config.locomo.commit_poll_seconds),
            },
        }
        self._event(import_dir, "import_started", {"sample_id": sample_id, "session_id": session_id, "sessions": pending_sessions})
        service = AgentChatService(self.config)
        self._replay_sample(
            import_dir,
            service,
            import_run,
            sample,
            session_id,
            allowed_sessions=set(pending_sessions),
        )
        merged = sorted(imported_sessions | set(pending_sessions), key=_session_sort_key)
        result = {
            "sample_id": sample_id,
            "status": "imported" if set(available).issubset(set(merged)) else "partial",
            "imported": True,
            "session_id": session_id,
            "sessions": merged,
            "session_count": len(merged),
            "total_sessions": len(available),
            "updated_at": _now(),
        }
        self._write_json(self._import_path(sample_id), result)
        self._event(import_dir, "import_completed", result)
        return result

    def _replay_sample(
        self,
        run_dir: Path,
        service: AgentChatService,
        run: dict[str, Any],
        sample: dict[str, Any],
        session_id: str,
        allowed_sessions: set[str] | None = None,
    ) -> None:
        self._event(run_dir, "sample_replay_started", {"sample_id": sample["sample_id"], "session_id": session_id})
        if run["config"].get("dry_run"):
            self._event(run_dir, "sample_replay_skipped", {"sample_id": sample["sample_id"], "reason": "dry_run"})
            return
        service.memory.open_session(run["config"]["user_id"], run["config"]["agent_id"], session_id)
        count = 0
        speaker_a = str(sample.get("speaker_a") or "")
        for session in sample["sessions"]:
            if allowed_sessions is not None and session["id"] not in allowed_sessions:
                continue
            self._event(run_dir, "session_replay_started", {"sample_id": sample["sample_id"], "session": session["id"]})
            for turn in session.get("turns", []):
                speaker = str(turn.get("speaker") or "speaker")
                text = str(turn.get("text") or "").strip()
                if not text:
                    continue
                content = f"[{sample['sample_id']} {session['id']} {turn.get('dia_id') or ''} {speaker}]\n{text}"
                role = "user" if not speaker_a or speaker == speaker_a else "assistant"
                service.memory.add_message(session_id, role, content)
                count += 1
                self._event(
                    run_dir,
                    "turn_replayed",
                    {
                        "sample_id": sample["sample_id"],
                        "session": session["id"],
                        "dia_id": turn.get("dia_id"),
                        "speaker": speaker,
                        "role": role,
                        "text": text,
                    },
                )
        commit_result = None
        if run["config"]["commit_after_replay"]:
            commit_result = service.memory.commit(session_id)
            self._event(
                run_dir,
                "commit_submitted",
                {"session_id": session_id, "commit_id": _commit_id_from(commit_result), "commit": commit_result},
            )
            commit_result = self._wait_for_commit(run_dir, service, run, session_id, commit_result)
        self._event(
            run_dir,
            "sample_replay_completed",
            {"sample_id": sample["sample_id"], "turn_count": count, "commit": commit_result},
        )

    def _import_session_id(self, sample_id: str) -> str:
        return f"locomo-import-{sample_id}"

    def _import_path(self, sample_id: str) -> Path:
        return self.imports_dir / f"{sample_id}.json"

    def _import_record(self, sample_id: str) -> dict[str, Any]:
        path = self._import_path(sample_id)
        if not path.exists():
            return {"sample_id": sample_id, "imported": False, "sessions": []}
        return self._read_json(path)

    def _wait_for_commit(
        self,
        run_dir: Path,
        service: AgentChatService,
        run: dict[str, Any],
        session_id: str,
        commit_result: dict[str, Any],
    ) -> dict[str, Any]:
        commit_id = _commit_id_from(commit_result)
        if not commit_id:
            raise LocomoEvalError(f"commit response did not include commit_id: {commit_result}")

        wait_seconds = max(0.0, float(run["config"].get("commit_wait_seconds") or 0))
        poll_seconds = max(0.1, float(run["config"].get("commit_poll_seconds") or 2))
        deadline = time.monotonic() + wait_seconds
        attempt = 0
        self._event(
            run_dir,
            "commit_waiting",
            {"session_id": session_id, "commit_id": commit_id, "wait_seconds": wait_seconds},
        )
        while True:
            attempt += 1
            status = service.memory.get_commit_status(session_id, commit_id)
            status_value = _commit_status_value(status)
            self._event(
                run_dir,
                "commit_status",
                {"commit_id": commit_id, "attempt": attempt, "status_value": status_value, "status": status},
            )
            if status_value == "completed":
                return {"commit_id": commit_id, "status": status}
            if status_value == "failed":
                raise LocomoEvalError(f"commit failed before QA: {status}")
            if time.monotonic() >= deadline:
                raise LocomoEvalError(f"commit not ready before QA after {wait_seconds:g}s: {commit_id}")
            time.sleep(min(poll_seconds, max(0.1, deadline - time.monotonic())))

    def _answer_qa(
        self,
        run_dir: Path,
        service: AgentChatService,
        run: dict[str, Any],
        sample: dict[str, Any],
        selected: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        qa = selected["qa"]
        self._event(run_dir, "qa_started", {"sample_id": sample["sample_id"], "qa_id": qa["id"]})
        answer = ""
        context_trace = None
        messages = []
        error = None
        if run["config"]["dry_run"]:
            answer = "[dry_run] skipped model call"
        else:
            try:
                response = service.chat(
                    {
                        "user_id": run["config"]["user_id"],
                        "agent_id": run["config"]["agent_id"],
                        "session_id": session_id,
                        "message": qa["question"],
                    }
                )
                answer = response["assistant"]["content"]
                messages = response.get("messages") or []
                context_trace = response.get("context_trace")
            except (EchoMemoryClientError, ModelClientError, ValueError) as exc:
                error = str(exc)
        scores = _score_answer(str(qa.get("answer") or ""), answer)
        result = {
            "sample_id": sample["sample_id"],
            "qa_id": qa["id"],
            "question": qa["question"],
            "gold_answer": qa["answer"],
            "agent_answer": answer,
            "category": qa["category"],
            "evidence": qa["evidence"],
            "scores": scores,
            "error": error,
            "completed_at": _now(),
        }
        safe_id = f"{sample['sample_id']}_{qa['id']}".replace("/", "_")
        answers_dir = run_dir / "answers"
        contexts_dir = run_dir / "contexts"
        answers_dir.mkdir(exist_ok=True)
        contexts_dir.mkdir(exist_ok=True)
        self._write_json(answers_dir / f"{safe_id}.json", result)
        self._write_json(contexts_dir / f"{safe_id}.json", {"messages": messages, "context_trace": context_trace})
        self._event(run_dir, "qa_completed", {"sample_id": sample["sample_id"], "qa_id": qa["id"], "scores": scores, "error": error})
        return result

    def _select_items(self, samples: list[dict[str, Any]], qa_ids: Any) -> list[dict[str, Any]]:
        selected = []
        qa_map = qa_ids if isinstance(qa_ids, dict) else {}
        for sample in samples:
            allowed = qa_map.get(sample["sample_id"], "all") if qa_map else "all"
            allowed_ids = set(_string_list(allowed)) if allowed != "all" else None
            for qa in sample["qa"]:
                if allowed_ids is not None and qa["id"] not in allowed_ids and str(qa["index"]) not in allowed_ids:
                    continue
                selected.append({"sample_id": sample["sample_id"], "qa": qa})
        return selected

    def _run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise LocomoEvalError("invalid run_id")
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise LocomoEvalError(f"LoCoMo run not found: {run_id}")
        return run_dir

    def _write_results(self, run_dir: Path, run: dict[str, Any]) -> None:
        self._write_json(run_dir / "results.json", {"run_id": run["run_id"], "results": run["results"], "summary": run["summary"]})

    def _update_run(self, run_dir: Path, run: dict[str, Any], **updates: Any) -> None:
        run.update(updates)
        self._write_json(run_dir / "run.json", run)

    def _event(self, run_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
        event = {"ts": _now(), "type": event_type, **payload}
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocomoEvalError(f"failed to read {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise LocomoEvalError(f"{path} must contain a JSON object")
        return data

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _new_run_id() -> str:
    return "locomo_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value and value != "all" else ([] if value != "all" else ["all"])
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _score_answer(gold: str, answer: str) -> dict[str, Any]:
    gold_norm = _normalize_text(gold)
    answer_norm = _normalize_text(answer)
    contains = 1.0 if gold_norm and gold_norm in answer_norm else 0.0
    gold_tokens = gold_norm.split()
    answer_tokens = answer_norm.split()
    if not gold_tokens or not answer_tokens:
        f1 = 0.0
    else:
        remaining = answer_tokens.copy()
        overlap = 0
        for token in gold_tokens:
            if token in remaining:
                overlap += 1
                remaining.remove(token)
        precision = overlap / len(answer_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"contains": contains, "token_f1": round(f1, 4)}


def _normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.lower(), flags=re.UNICODE).split())


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"count": 0, "token_f1": 0.0, "contains": 0.0, "errors": 0}
    return {
        "count": len(results),
        "token_f1": round(sum(result["scores"]["token_f1"] for result in results) / len(results), 4),
        "contains": round(sum(result["scores"]["contains"] for result in results) / len(results), 4),
        "errors": sum(1 for result in results if result.get("error")),
    }


def _commit_id_from(commit_result: dict[str, Any]) -> str | None:
    candidates = [commit_result]
    if isinstance(commit_result.get("result"), dict):
        candidates.append(commit_result["result"])
    for item in candidates:
        for key in ("commit_id", "archive_id"):
            value = item.get(key)
            if value:
                return str(value)
    return None


def _commit_status_value(status: dict[str, Any]) -> str | None:
    if isinstance(status.get("status"), dict):
        value = status["status"].get("status")
        return str(value) if value else None
    value = status.get("status")
    return str(value) if value else None


def _session_sort_key(value: str) -> int:
    try:
        return int(str(value).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0
