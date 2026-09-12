"""Shared validation and checksum helpers for demonstration records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_schedule(path: Path) -> list[dict[str, Any]]:
    trials = []
    seen_ids = set()
    seen_indices = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            trial = json.loads(line)
            trial_id = trial.get("trial_id")
            schedule_index = trial.get("schedule_index")
            if not isinstance(trial_id, str) or not trial_id:
                raise ValueError(f"missing trial_id at {path}:{line_number}")
            if not isinstance(schedule_index, int) or schedule_index < 1:
                raise ValueError(f"invalid schedule_index at {path}:{line_number}")
            if trial_id in seen_ids:
                raise ValueError(f"duplicate trial_id={trial_id} in {path}")
            if schedule_index in seen_indices:
                raise ValueError(f"duplicate schedule_index={schedule_index} in {path}")
            if not isinstance(trial.get("messages"), list) or not trial["messages"]:
                raise ValueError(f"missing messages at {path}:{line_number}")
            seen_ids.add(trial_id)
            seen_indices.add(schedule_index)
            trials.append(trial)
    if [trial["schedule_index"] for trial in trials] != list(range(1, len(trials) + 1)):
        raise ValueError("schedule must be stored in contiguous schedule_index order")
    return trials


def completed_trial_ids(raw_dir: Path) -> set[str]:
    completed = set()
    for path in sorted(raw_dir.glob("*.json")):
        record = load_json(path)
        trial_id = record.get("trial_id")
        if not isinstance(trial_id, str):
            raise ValueError(f"raw record lacks trial_id: {path}")
        if trial_id in completed:
            raise ValueError(f"duplicate completed trial_id={trial_id}")
        payload = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != canonical_sha256(payload):
            raise ValueError(f"raw record checksum failed: {path}")
        completed.add(trial_id)
    return completed
