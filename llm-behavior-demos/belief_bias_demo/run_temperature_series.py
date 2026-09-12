"""Run the frozen full NeuBAROCO schedule at 0.1 temperature intervals."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Callable

TEMPERATURES = tuple(Decimal(index) / Decimal(10) for index in range(11))


def temperature_label(value: Decimal) -> str:
    return f"{value:.1f}".replace(".", "p")


def run_series(
    *,
    schedule: Path,
    build_manifest: Path,
    runs_root: Path,
    model: str,
    max_tokens: int,
    runner: Callable[..., object] = subprocess.run,
) -> None:
    with build_manifest.open(encoding="utf-8") as source:
        build = json.load(source)
    if build.get("selection_mode") != "full" or build.get("trial_count") not in {
        1098,
        1125,
    }:
        raise ValueError("temperature series requires a frozen full-mode schedule")
    trials_per_temperature = build["trial_count"]

    runs_root.mkdir(parents=True, exist_ok=True)
    series = {
        "schema_version": 1,
        "temperatures": [float(value) for value in TEMPERATURES],
        "trials_per_temperature": trials_per_temperature,
        "total_scheduled_calls": trials_per_temperature * len(TEMPERATURES),
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "model": model,
        "schedule_sha256": build["schedule_sha256"],
    }
    series_path = runs_root / "temperature-series-manifest.json"
    if not series_path.exists():
        with series_path.open("x", encoding="utf-8") as destination:
            json.dump(series, destination, indent=2, sort_keys=True)
            destination.write("\n")
    elif json.loads(series_path.read_text(encoding="utf-8")) != series:
        raise ValueError("existing temperature-series manifest differs")

    for temperature in TEMPERATURES:
        run_dir = runs_root / f"temp-{temperature_label(temperature)}"
        command = [
            sys.executable,
            "-m",
            "position_bias_demo.run_trials",
            "--schedule",
            str(schedule),
            "--build-manifest",
            str(build_manifest),
            "--run-dir",
            str(run_dir),
            "--model",
            model,
            "--temperature",
            str(temperature),
            "--top-p",
            "1.0",
            "--max-tokens",
            str(max_tokens),
            "--timeout",
            "120",
            "--max-retries",
            "1",
        ]
        print(f"Starting temperature {temperature:.1f}: {run_dir}", flush=True)
        runner(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--model", default="qwen-3b")
    parser.add_argument("--max-tokens", type=int, default=16)
    run_series(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
