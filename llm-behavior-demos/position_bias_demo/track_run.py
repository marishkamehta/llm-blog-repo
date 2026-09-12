"""Track a position-bias collection without displaying response content."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from filelock import FileLock, Timeout

from demo_records import completed_trial_ids


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def parse_time(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def collector_active(run_dir: Path) -> bool:
    lock_path = run_dir / ".collector.lock"
    if not lock_path.exists():
        return False
    lock = FileLock(lock_path)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return True
    lock.release()
    return False


def error_count(run_dir: Path) -> int:
    path = run_dir / "errors.jsonl"
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as source:
        return sum(bool(line.strip()) for line in source)


def snapshot(run_dir: Path, verify: bool = False) -> dict:
    manifest_path = run_dir / "run-manifest.json"
    raw_dir = run_dir / "raw-responses"
    if not manifest_path.is_file() or not raw_dir.is_dir():
        raise ValueError(f"not a position-bias run directory: {run_dir}")

    manifest = load_json(manifest_path)
    scheduled = manifest.get("scheduled_trial_count")
    if scheduled is None:  # Compatibility with runs created before schema version 2.
        schedule_path = Path(manifest["schedule_path"])
        if not schedule_path.is_file():
            raise FileNotFoundError(
                f"schedule is unavailable at its recorded path: {schedule_path}"
            )
        with schedule_path.open(encoding="utf-8") as source:
            scheduled = sum(bool(line.strip()) for line in source)

    raw_files = list(raw_dir.glob("*.json"))
    if verify:
        completed = len(completed_trial_ids(raw_dir))
    else:
        completed = len(raw_files)
    remaining = scheduled - completed
    now = time.time()
    created = parse_time(manifest["created_at_utc"])
    elapsed = max(now - created, 0.0)
    overall_rate = completed / elapsed if elapsed and completed else 0.0
    recent_window = 300.0
    recent = sum(path.stat().st_mtime >= now - recent_window for path in raw_files)
    recent_rate = recent / recent_window
    rate = recent_rate if recent >= 2 else overall_rate
    eta_seconds = remaining / rate if rate > 0 and remaining else 0.0
    return {
        "model": manifest["model_registry_name"],
        "scheduled": scheduled,
        "completed": completed,
        "remaining": remaining,
        "percent": completed / scheduled * 100 if scheduled else 0.0,
        "errors": error_count(run_dir),
        "collector_active": collector_active(run_dir),
        "elapsed_seconds": elapsed,
        "recent_trials_per_minute": recent_rate * 60,
        "eta_seconds": eta_seconds,
        "verified_checksums": verify,
    }


def duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def display(status: dict) -> None:
    state = "running" if status["collector_active"] else "not running"
    verification = "yes" if status["verified_checksums"] else "no"
    print(f"Model:       {status['model']}")
    print(f"Collector:   {state}")
    print(f"Progress:    {status['completed']} / {status['scheduled']} ({status['percent']:.1f}%)")
    print(f"Remaining:   {status['remaining']}")
    print(f"Errors:      {status['errors']}")
    print(f"Recent rate: {status['recent_trials_per_minute']:.1f} trials/min")
    print(f"Elapsed:     {duration(status['elapsed_seconds'])}")
    print(f"ETA:         {duration(status['eta_seconds'])}")
    print(f"Checksums:   {verification}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="refresh repeatedly at this interval",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="validate every completed raw-record checksum before reporting",
    )
    args = parser.parse_args()
    if args.watch is not None and args.watch < 1:
        parser.error("--watch must be at least 1 second")

    while True:
        if args.watch is not None:
            print("\033[2J\033[H", end="")
        status = snapshot(args.run_dir, verify=args.verify)
        display(status)
        if args.watch is None or status["remaining"] == 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
