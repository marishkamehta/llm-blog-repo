from __future__ import annotations

import unittest

from position_bias_demo.analyze import (
    aggregate,
    parse_verdict,
    parse_verdict_fastchat,
    summarize_position,
    summarize_repetitions,
)


def rows_for_pair(first: list[str], second: list[str], condition="original"):
    rows = []
    for order, verdicts in (
        ("baseline_comparison", first),
        ("comparison_baseline", second),
    ):
        for repetition, verdict in enumerate(verdicts, start=1):
            rows.append(
                {
                    "trial_id": f"{order}-{repetition}",
                    "schedule_index": len(rows) + 1,
                    "question_id": 81,
                    "category": "writing",
                    "comparison_file": "comparison.jsonl",
                    "order": order,
                    "prompt_condition": condition,
                    "repetition": repetition,
                    "candidate_a": ("baseline" if order == "baseline_comparison" else "comparison"),
                    "candidate_b": ("comparison" if order == "baseline_comparison" else "baseline"),
                    "collection_status": "collected",
                    "raw_record_sha256": "x",
                    "valid": verdict in {"A", "B"},
                    "verdict": verdict if verdict in {"A", "B"} else None,
                    "selected_candidate": None,
                    "parse_reason": None if verdict in {"A", "B"} else "invalid",
                }
            )
    return rows


class ParserTests(unittest.TestCase):
    def test_primary_accepts_one_unique_token_including_markdown(self) -> None:
        for text in (
            "Answer A is more accurate.\n\n[[A]]",
            "**Final Verdict: [[A]]**",
            "[[B]] because it is better",
        ):
            with self.subTest(text=text):
                self.assertTrue(parse_verdict(text)["valid"])

    def test_primary_rejects_missing_multiple_and_tie_verdicts(self) -> None:
        cases = {
            "I choose A": "missing_verdict",
            "[[A]] then [[B]]": "multiple_verdicts",
            "[[C]]": "tie_not_allowed",
            "[[a]]": "missing_verdict",
        }
        for text, reason in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_verdict(text)["reason"], reason)

    def test_literal_fastchat_uses_a_then_b_then_c_containment(self) -> None:
        cases = {
            "**Final Verdict: [[A]]**": "A",
            "[[B]] trailing prose": "B",
            "tie: [[C]]": "C",
            "first [[B]], later [[A]]": "A",
        }
        for text, verdict in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_verdict_fastchat(text)["verdict"], verdict)
        self.assertFalse(parse_verdict_fastchat("no token")["valid"])


class MetricTests(unittest.TestCase):
    def test_same_underlying_candidate_is_position_consistent(self) -> None:
        parsed = rows_for_pair(["A", "A", "B"], ["B", "B", "A"])
        repetitions = summarize_repetitions(parsed)
        position = summarize_position(repetitions)[0]
        self.assertTrue(position["position_consistent"])
        self.assertEqual(position["baseline_first_selected_candidate"], "baseline")
        self.assertEqual(position["comparison_first_selected_candidate"], "baseline")

    def test_same_visible_a_choice_is_primacy_inconsistent(self) -> None:
        parsed = rows_for_pair(["A", "A", "B"], ["A", "A", "B"])
        repetitions = summarize_repetitions(parsed)
        position = summarize_position(repetitions)[0]
        self.assertFalse(position["position_consistent"])
        self.assertEqual(position["inconsistent_preference"], "primacy")

    def test_same_visible_b_choice_is_recency_inconsistent(self) -> None:
        parsed = rows_for_pair(["B", "B", "A"], ["B", "B", "A"])
        position = summarize_position(summarize_repetitions(parsed))[0]
        self.assertEqual(position["inconsistent_preference"], "recency")

    def test_invalids_remain_in_conservative_stability_denominator(self) -> None:
        parsed = rows_for_pair(["A", "A", "invalid"], ["B", "B", "invalid"])
        repetitions = summarize_repetitions(parsed)
        self.assertEqual(repetitions[0]["stability_valid"], 1.0)
        self.assertEqual(repetitions[0]["stability_all_scheduled"], 2 / 3)
        summary = aggregate(parsed, repetitions, summarize_position(repetitions))[0]
        self.assertEqual(summary["invalid_or_missing_trials"], 2)


if __name__ == "__main__":
    unittest.main()
