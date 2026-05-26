"""LoCoMo dataset download, cache, and indexing helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .config import LocomoConfig


class LocomoDatasetError(RuntimeError):
    """Raised when the local or remote LoCoMo dataset is unavailable."""


class LocomoDatasetService:
    def __init__(self, config: LocomoConfig) -> None:
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.dataset_path = self.data_dir / "locomo10.json"
        self.manifest_path = self.data_dir / "manifest.json"

    def ensure_dataset(self, *, force: bool = False) -> dict[str, Any]:
        if self.dataset_path.exists() and not force:
            return self._manifest(downloaded=False)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.dataset_path.exists():
            archive_dir = self.data_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            shutil.copy2(self.dataset_path, archive_dir / f"locomo10.{stamp}.json")

        try:
            with urlopen(self.config.dataset_url, timeout=30) as response:
                body = response.read()
        except URLError as exc:
            raise LocomoDatasetError(f"failed to download LoCoMo dataset: {exc.reason}") from exc

        data = self._loads_dataset(body)
        sha256 = hashlib.sha256(body).hexdigest()
        self.dataset_path.write_bytes(body)
        manifest = {
            "source_url": self.config.dataset_url,
            "sha256": sha256,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "sample_count": len(data),
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**manifest, "downloaded": True, "path": str(self.dataset_path)}

    def index(self) -> dict[str, Any]:
        self._auto_download_if_needed()
        data = self._read_dataset()
        samples = [self._sample_summary(sample) for sample in data]
        categories = sorted(
            {
                str(qa.get("category"))
                for sample in data
                for qa in _qa_items(sample)
                if qa.get("category")
            }
        )
        return {"manifest": self._manifest(downloaded=False), "samples": samples, "categories": categories}

    def sample(self, sample_id: str) -> dict[str, Any]:
        self._auto_download_if_needed()
        for sample in self._read_dataset():
            if sample.get("sample_id") == sample_id:
                return self._normalize_sample(sample)
        raise LocomoDatasetError(f"LoCoMo sample not found: {sample_id}")

    def samples_for(self, sample_ids: list[str]) -> list[dict[str, Any]]:
        self._auto_download_if_needed()
        wanted = set(sample_ids)
        samples = [self._normalize_sample(item) for item in self._read_dataset() if item.get("sample_id") in wanted]
        found = {sample["sample_id"] for sample in samples}
        missing = sorted(wanted - found)
        if missing:
            raise LocomoDatasetError(f"LoCoMo samples not found: {', '.join(missing)}")
        return samples

    def _auto_download_if_needed(self) -> None:
        if self.dataset_path.exists():
            return
        if not self.config.auto_download:
            raise LocomoDatasetError("LoCoMo dataset is missing and auto_download is disabled")
        self.ensure_dataset()

    def _read_dataset(self) -> list[dict[str, Any]]:
        try:
            body = self.dataset_path.read_bytes()
        except OSError as exc:
            raise LocomoDatasetError(f"failed to read LoCoMo dataset: {exc}") from exc
        return self._loads_dataset(body)

    def _loads_dataset(self, body: bytes) -> list[dict[str, Any]]:
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LocomoDatasetError("LoCoMo dataset must be valid UTF-8 JSON") from exc
        if not isinstance(data, list):
            raise LocomoDatasetError("LoCoMo dataset root must be a list")
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise LocomoDatasetError(f"LoCoMo sample at index {index} must be an object")
            if not item.get("sample_id") or not isinstance(item.get("conversation"), dict):
                raise LocomoDatasetError(f"LoCoMo sample at index {index} is missing sample_id or conversation")
        return data

    def _manifest(self, *, downloaded: bool) -> dict[str, Any]:
        manifest: dict[str, Any] = {}
        if self.manifest_path.exists():
            try:
                loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
            except json.JSONDecodeError:
                manifest = {}
        if self.dataset_path.exists():
            body = self.dataset_path.read_bytes()
            manifest.setdefault("sha256", hashlib.sha256(body).hexdigest())
            manifest.setdefault("sample_count", len(self._loads_dataset(body)))
        manifest.setdefault("source_url", self.config.dataset_url)
        manifest["downloaded"] = downloaded
        manifest["path"] = str(self.dataset_path)
        manifest["exists"] = self.dataset_path.exists()
        return manifest

    def _sample_summary(self, sample: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_sample(sample, include_turns=False)
        qa = normalized["qa"]
        categories: dict[str, int] = {}
        for item in qa:
            category = str(item.get("category") or "unknown")
            categories[category] = categories.get(category, 0) + 1
        return {
            "sample_id": normalized["sample_id"],
            "speaker_a": normalized["speaker_a"],
            "speaker_b": normalized["speaker_b"],
            "session_count": len(normalized["sessions"]),
            "turn_count": sum(session["turn_count"] for session in normalized["sessions"]),
            "qa_count": len(qa),
            "categories": categories,
        }

    def _normalize_sample(self, sample: dict[str, Any], *, include_turns: bool = True) -> dict[str, Any]:
        conversation = sample.get("conversation") if isinstance(sample.get("conversation"), dict) else {}
        sessions = []
        for key in sorted(
            [key for key, value in conversation.items() if key.startswith("session_") and isinstance(value, list)],
            key=_session_sort_key,
        ):
            turns = conversation.get(key) or []
            session = {
                "id": key,
                "date_time": conversation.get(f"{key}_date_time"),
                "turn_count": len(turns),
                "observation": _section_value(sample.get("observation"), f"{key}_observation"),
                "summary": _section_value(sample.get("session_summary"), f"{key}_summary"),
                "events": _section_value(sample.get("event_summary"), f"events_{key}"),
            }
            if include_turns:
                session["turns"] = turns
            sessions.append(session)

        return {
            "sample_id": sample.get("sample_id"),
            "speaker_a": conversation.get("speaker_a"),
            "speaker_b": conversation.get("speaker_b"),
            "sessions": sessions,
            "qa": [_normalize_qa(index, item) for index, item in enumerate(_qa_items(sample), start=1)],
        }


def _qa_items(sample: dict[str, Any]) -> list[dict[str, Any]]:
    qa = sample.get("qa")
    if isinstance(qa, list):
        return [item for item in qa if isinstance(item, dict)]
    if isinstance(qa, dict):
        for key in ("items", "questions", "qa"):
            value = qa.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _normalize_qa(index: int, item: dict[str, Any]) -> dict[str, Any]:
    qa_id = str(item.get("id") or item.get("qa_id") or f"qa_{index:03d}")
    evidence = item.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [evidence]
    return {
        "id": qa_id,
        "index": index,
        "question": item.get("question") or "",
        "answer": item.get("answer") or "",
        "category": item.get("category") or "unknown",
        "evidence": [str(value) for value in evidence],
    }


def _section_value(section: Any, key: str) -> Any:
    if isinstance(section, dict):
        return section.get(key)
    return None


def _session_sort_key(key: str) -> int:
    try:
        return int(key.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0
