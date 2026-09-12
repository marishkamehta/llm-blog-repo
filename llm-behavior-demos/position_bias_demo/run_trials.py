"""Collect position-bias judgments with immutable per-trial raw records."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from silico import ChatModelConfig, ExecutionConfig, make_llm
from silico.contracts import LLMBase

from demo_records import (
    canonical_sha256,
    completed_trial_ids,
    load_json,
    load_schedule,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state(repo: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": command("rev-parse", "HEAD"),
            "worktree_dirty": bool(command("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None}


def write_new_json_exclusive(path: Path, value: Any) -> None:
    """Create path exclusively and fail rather than overwrite an existing file."""
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())


def replace_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as destination:
            json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def raw_filename(trial: dict[str, Any]) -> str:
    return f"{trial['schedule_index']:04d}-{trial['trial_id']}.json"


def append_error(path: Path, error: dict[str, Any]) -> None:
    line = json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as destination:
        destination.write(line)
        destination.flush()
        os.fsync(destination.fileno())


def initial_manifest(
    *,
    schedule_path: Path,
    build_manifest_path: Path,
    model_name: str,
    config: ChatModelConfig,
    execution: ExecutionConfig,
    repo: Path,
    trial_count: int,
) -> dict[str, Any]:
    build_manifest = load_json(build_manifest_path)
    schedule_hash = sha256_file(schedule_path)
    if schedule_hash != build_manifest.get("schedule_sha256"):
        raise ValueError(
            "schedule checksum does not match the supplied build manifest: "
            f"observed {schedule_hash}"
        )
    return {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "schedule_file": schedule_path.name,
        "schedule_sha256": schedule_hash,
        "scheduled_trial_count": trial_count,
        "build_manifest_file": build_manifest_path.name,
        "build_manifest_sha256": sha256_file(build_manifest_path),
        "model_registry_name": model_name,
        "generation": asdict(config),
        "execution": asdict(execution),
        "code": git_state(repo),
    }


def prepare_run_dir(run_dir: Path, proposed_manifest: dict[str, Any]) -> None:
    manifest_path = run_dir / "run-manifest.json"
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        (run_dir / "raw-responses").mkdir()
        write_new_json_exclusive(manifest_path, proposed_manifest)
        return
    if not run_dir.is_dir():
        raise ValueError(f"run path is not a directory: {run_dir}")
    if not manifest_path.is_file() or not (run_dir / "raw-responses").is_dir():
        raise ValueError(f"existing run directory is incomplete: {run_dir}")
    existing = load_json(manifest_path)
    immutable_fields = (
        "schedule_sha256",
        "build_manifest_sha256",
        "model_registry_name",
        "generation",
        "execution",
    )
    mismatches = [
        field for field in immutable_fields if existing.get(field) != proposed_manifest.get(field)
    ]
    if mismatches:
        raise ValueError(
            "resume configuration differs from the initial run: " + ", ".join(mismatches)
        )


def acquire_collector_lock(run_dir: Path) -> FileLock:
    """Hold an exclusive process lock so two collectors cannot issue duplicates."""
    lock = FileLock(run_dir / ".collector.lock")
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise RuntimeError(f"another collector is active for {run_dir}") from None
    return lock


def collect(
    *,
    schedule_path: Path,
    build_manifest_path: Path,
    run_dir: Path,
    model_name: str,
    llm: LLMBase,
    config: ChatModelConfig,
    execution: ExecutionConfig,
    repo: Path,
    max_new_trials: int | None = None,
) -> dict[str, Any]:
    trials = load_schedule(schedule_path)
    manifest = initial_manifest(
        schedule_path=schedule_path,
        build_manifest_path=build_manifest_path,
        model_name=model_name,
        config=config,
        execution=execution,
        repo=repo,
        trial_count=len(trials),
    )
    prepare_run_dir(run_dir, manifest)
    collector_lock = acquire_collector_lock(run_dir)
    try:
        raw_dir = run_dir / "raw-responses"
        completed = completed_trial_ids(raw_dir)
        scheduled_ids = {trial["trial_id"] for trial in trials}
        unexpected = completed - scheduled_ids
        if unexpected:
            raise ValueError(f"run contains trial IDs absent from schedule: {unexpected}")

        attempted_this_invocation = 0
        succeeded_this_invocation = 0
        failed_this_invocation = 0
        invocation_started = utc_now()
        for trial in trials:
            if trial["trial_id"] in completed:
                continue
            if max_new_trials is not None and attempted_this_invocation >= max_new_trials:
                break
            attempted_this_invocation += 1
            started_at = utc_now()
            try:
                response = llm.respond_with_metadata(
                    trial["messages"], config=config, execution=execution
                )
                record_without_hash = {
                    "schema_version": 1,
                    "trial_id": trial["trial_id"],
                    "schedule_index": trial["schedule_index"],
                    "schedule_trial_sha256": canonical_sha256(trial),
                    "model_registry_name": model_name,
                    "started_at_utc": started_at,
                    "saved_at_utc": utc_now(),
                    "raw_text": response.text,
                    "request": response.request,
                    "response": response.response,
                }
                record = {
                    **record_without_hash,
                    "record_sha256": canonical_sha256(record_without_hash),
                }
                write_new_json_exclusive(raw_dir / raw_filename(trial), record)
                completed.add(trial["trial_id"])
                succeeded_this_invocation += 1
            except Exception as exc:  # Preserve failure and continue collection.
                failed_this_invocation += 1
                append_error(
                    run_dir / "errors.jsonl",
                    {
                        "trial_id": trial["trial_id"],
                        "schedule_index": trial["schedule_index"],
                        "model_registry_name": model_name,
                        "started_at_utc": started_at,
                        "failed_at_utc": utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )

        status = {
            "schema_version": 1,
            "updated_at_utc": utc_now(),
            "invocation_started_at_utc": invocation_started,
            "scheduled": len(trials),
            "completed": len(completed),
            "remaining": len(trials) - len(completed),
            "attempted_this_invocation": attempted_this_invocation,
            "succeeded_this_invocation": succeeded_this_invocation,
            "failed_this_invocation": failed_this_invocation,
            "complete": len(completed) == len(trials),
        }
        replace_json_atomic(run_dir / "run-status.json", status)
        return status
    finally:
        collector_lock.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--max-new-trials", type=int)
    args = parser.parse_args()

    config = ChatModelConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    execution = ExecutionConfig(timeout=args.timeout, max_retries=args.max_retries)
    status = collect(
        schedule_path=args.schedule,
        build_manifest_path=args.build_manifest,
        run_dir=args.run_dir,
        model_name=args.model,
        llm=make_llm(args.model),
        config=config,
        execution=execution,
        repo=Path(__file__).resolve().parents[1],
        max_new_trials=args.max_new_trials,
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
