from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from silico import ChatModelConfig, ExecutionConfig, MockLLM

from demo_records import sha256_file
from position_bias_demo.run_trials import (
    acquire_collector_lock,
    collect,
)


def write_schedule(root: Path, count: int = 3) -> tuple[Path, Path]:
    schedule = root / "trials.jsonl"
    with schedule.open("w", encoding="utf-8") as destination:
        for index in range(1, count + 1):
            destination.write(
                json.dumps(
                    {
                        "trial_id": f"trial-{index}",
                        "schedule_index": index,
                        "messages": [
                            {"role": "system", "content": "Judge."},
                            {"role": "user", "content": f"Question {index}"},
                        ],
                    }
                )
                + "\n"
            )
    build_manifest = root / "build-manifest.json"
    build_manifest.write_text(
        json.dumps({"schedule_sha256": sha256_file(schedule)}) + "\n",
        encoding="utf-8",
    )
    return schedule, build_manifest


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.schedule, self.build_manifest = write_schedule(self.root)
        self.run_dir = self.root / "run"
        self.config = ChatModelConfig(temperature=1.0, max_tokens=20)
        self.execution = ExecutionConfig(max_retries=0)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def collect_run(self, *, max_new_trials=None, model_name="mock", llm=None):
        return collect(
            schedule_path=self.schedule,
            build_manifest_path=self.build_manifest,
            run_dir=self.run_dir,
            model_name=model_name,
            llm=llm or MockLLM(reply=lambda _: "[[A]]"),
            config=self.config,
            execution=self.execution,
            repo=Path(__file__).resolve().parents[1],
            max_new_trials=max_new_trials,
        )

    def test_interrupted_run_resumes_without_repeating_completed_trials(self) -> None:
        calls = []
        llm = MockLLM(reply=lambda prompt: calls.append(prompt) or "[[A]]")
        first = self.collect_run(max_new_trials=1, llm=llm)
        self.assertEqual(first["completed"], 1)
        manifest = json.loads((self.run_dir / "run-manifest.json").read_text())
        self.assertNotIn("schedule_path", manifest)
        self.assertNotIn("build_manifest_path", manifest)
        self.assertEqual(manifest["schedule_file"], self.schedule.name)
        self.assertEqual(manifest["scheduled_trial_count"], 3)
        second = self.collect_run(llm=llm)
        self.assertTrue(second["complete"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(list((self.run_dir / "raw-responses").glob("*.json"))), 3)

    def test_completed_run_makes_no_additional_calls(self) -> None:
        self.collect_run()
        calls = []
        status = self.collect_run(llm=MockLLM(reply=lambda value: calls.append(value) or "B"))
        self.assertEqual(calls, [])
        self.assertEqual(status["attempted_this_invocation"], 0)

    def test_resume_rejects_model_or_generation_changes(self) -> None:
        self.collect_run(max_new_trials=1)
        with self.assertRaisesRegex(ValueError, "model_registry_name"):
            self.collect_run(model_name="different")
        self.config = ChatModelConfig(temperature=0.0, max_tokens=20)
        with self.assertRaisesRegex(ValueError, "generation"):
            self.collect_run()

    def test_schedule_mutation_is_detected(self) -> None:
        self.collect_run(max_new_trials=1)
        with self.schedule.open("a", encoding="utf-8") as destination:
            destination.write("\n")
        with self.assertRaisesRegex(ValueError, "schedule checksum"):
            self.collect_run()

    def test_failures_are_logged_and_not_marked_complete(self) -> None:
        def fail(_):
            raise RuntimeError("deliberate failure")

        status = self.collect_run(max_new_trials=1, llm=MockLLM(reply=fail))
        self.assertEqual(status["completed"], 0)
        self.assertEqual(status["failed_this_invocation"], 1)
        self.assertIn("deliberate failure", (self.run_dir / "errors.jsonl").read_text())

    def test_second_collector_is_rejected_before_duplicate_calls(self) -> None:
        self.run_dir.mkdir()
        first_lock = acquire_collector_lock(self.run_dir)
        try:
            with self.assertRaisesRegex(RuntimeError, "another collector is active"):
                acquire_collector_lock(self.run_dir)
        finally:
            first_lock.release()

    def test_collector_lock_is_released_after_unexpected_failure(self) -> None:
        with patch(
            "position_bias_demo.run_trials.completed_trial_ids",
            side_effect=RuntimeError("unexpected"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                self.collect_run()

        lock = acquire_collector_lock(self.run_dir)
        lock.release()


if __name__ == "__main__":
    unittest.main()
