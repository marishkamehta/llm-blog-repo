from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from belief_bias_demo.analyze_temperature_series import build_summaries
from belief_bias_demo.run_temperature_series import TEMPERATURES, run_series


class TemperatureRunnerTests(unittest.TestCase):
    def test_runs_all_temperatures_with_fixed_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "build.json"
            manifest.write_text(
                json.dumps(
                    {"selection_mode": "full", "trial_count": 1098, "schedule_sha256": "abc"}
                ),
                encoding="utf-8",
            )
            runner = Mock()
            with contextlib.redirect_stdout(io.StringIO()):
                run_series(
                    schedule=root / "trials.jsonl",
                    build_manifest=manifest,
                    runs_root=root / "runs",
                    model="test-model",
                    max_tokens=16,
                    runner=runner,
                )

            commands = [call.args[0] for call in runner.call_args_list]
            self.assertEqual(len(commands), 11)
            self.assertEqual([command[0] for command in commands], [sys.executable] * 11)
            self.assertEqual(
                [command[command.index("--temperature") + 1] for command in commands],
                [str(value) for value in TEMPERATURES],
            )
            self.assertTrue(
                all(command[command.index("--top-p") + 1] == "1.0" for command in commands)
            )


class CombinedAnalysisTests(unittest.TestCase):
    @staticmethod
    def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_excludes_selected_rows_from_combined_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "temp-0p0"
            folder.mkdir()
            trials = [
                {
                    "source_row": 1,
                    "content_type": "consistent",
                    "published_valid": True,
                    "published_correct": True,
                    "strict_valid": True,
                    "strict_correct": True,
                },
                {
                    "source_row": 234,
                    "content_type": "consistent",
                    "published_valid": True,
                    "published_correct": False,
                    "strict_valid": True,
                    "strict_correct": False,
                },
            ]
            items = [
                {
                    "source_row": 1,
                    "content_type": "consistent",
                    "modal_correct": True,
                    "repetition_stability": 1.0,
                },
                {
                    "source_row": 234,
                    "content_type": "consistent",
                    "modal_correct": False,
                    "repetition_stability": 1.0,
                },
            ]
            self.write_rows(folder / "parsed-trials.csv", trials)
            for parser in ("published", "strict"):
                self.write_rows(folder / f"{parser}-item-summaries.csv", items)

            with patch("belief_bias_demo.analyze_temperature_series.TEMPERATURES", [0.0]):
                overall, content = build_summaries(Path(directory), {234})

            self.assertEqual(len(overall), 2)
            self.assertTrue(all(row["scheduled_trials"] == 1 for row in overall))
            self.assertTrue(all(row["trial_accuracy"] == 1.0 for row in content))


if __name__ == "__main__":
    unittest.main()
