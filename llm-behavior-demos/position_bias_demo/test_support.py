from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from position_bias_demo.audit_materials import audit
from position_bias_demo.track_run import snapshot


def answer(first: str, second: str) -> dict:
    return {"choices": [{"turns": [first, second]}]}


class TrackerTests(unittest.TestCase):
    def test_snapshot_reports_progress_without_reading_response_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            raw_dir = run_dir / "raw-responses"
            raw_dir.mkdir(parents=True)
            (raw_dir / "one.json").write_text("{}", encoding="utf-8")
            (run_dir / "run-manifest.json").write_text(
                json.dumps(
                    {
                        "scheduled_trial_count": 2,
                        "model_registry_name": "mock",
                        "created_at_utc": "2026-01-01T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            with patch("position_bias_demo.track_run.collector_active", return_value=True):
                status = snapshot(run_dir)

            self.assertEqual(
                (status["scheduled"], status["completed"], status["remaining"]), (2, 1, 1)
            )
            self.assertTrue(status["collector_active"])


class MaterialAuditTests(unittest.TestCase):
    def test_audit_writes_one_complete_pair_record(self) -> None:
        question = {1: {"category": "writing", "turns": ["q1", "q2"]}}
        answers = {1: answer("first", "second")}
        scores = [
            {"question_id": 1, "model": "base", "score": 5},
            {"question_id": 1, "model": "base", "score": 5},
            {"question_id": 1, "model": "comp", "score": 8},
            {"question_id": 1, "model": "comp", "score": 8},
        ]
        trials = [{"question_id": 1, "comparison_file": "comp.jsonl", "messages": []}]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single, pair = root / "single.jsonl", root / "pair.jsonl"
            single.write_text("x", encoding="utf-8")
            pair.write_text("x", encoding="utf-8")
            with (
                patch("position_bias_demo.audit_materials.QUESTION_IDS", (1,)),
                patch("position_bias_demo.audit_materials.BASELINE_FILE", "base.jsonl"),
                patch("position_bias_demo.audit_materials.COMPARISON_FILES", ("comp.jsonl",)),
                patch("position_bias_demo.audit_materials.verify_sources", return_value={}),
                patch(
                    "position_bias_demo.audit_materials.load_jsonl",
                    side_effect=[question, answers, answers],
                ),
                patch(
                    "position_bias_demo.audit_materials.load_jsonl_rows", side_effect=[scores, []]
                ),
                patch("position_bias_demo.audit_materials.build_trials", return_value=trials),
                patch("position_bias_demo.audit_materials.tokenize_messages", return_value=100),
            ):
                manifest = audit(
                    input_dir=root,
                    single_judgments=single,
                    pair_judgments=pair,
                    output_dir=root / "audit",
                    tokenizer_endpoint="http://localhost:8000",
                    tokenizer_model="test",
                    seed=1,
                )

            self.assertEqual(manifest["row_count"], 1)
            row = (root / "audit/item-pair-audit.csv").read_text(encoding="utf-8")
            self.assertIn("clear", row)
            self.assertIn("comp", row)


if __name__ == "__main__":
    unittest.main()
