"""Tests for the NeuBAROCO schedule, parsing, and summaries."""

from __future__ import annotations

import unittest

from belief_bias_demo.analyze import aggregate, parse_published, parse_strict
from belief_bias_demo.build_schedule import (
    CONTENT_TYPES,
    GOLD_LABELS,
    build_trials,
    render_prompt,
    select_full_dataset,
    select_items,
)


def fixture_rows() -> list[dict[str, str]]:
    rows = []
    index = 0
    for content in CONTENT_TYPES:
        for gold in GOLD_LABELS:
            for _ in range(6):
                index += 1
                rows.append(
                    {
                        "source_row": str(index),
                        "premises_en": "All A are B. All B are C.",
                        "hypothesis_en": "All A are C.",
                        "gold": gold,
                        "figure": "AAA",
                        "conversion": "no",
                        "content-type": content,
                        "atmosphere": "no",
                        "syllogism-type": "syllogism",
                    }
                )
    return rows


class ScheduleTests(unittest.TestCase):
    def test_selection_is_balanced_reproducible_and_outcome_blind(self) -> None:
        first = select_items(fixture_rows(), 17)
        second = select_items(fixture_rows(), 17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 45)

    def test_schedule_has_three_repetitions_per_item(self) -> None:
        trials = build_trials(select_items(fixture_rows(), 17), 17)
        self.assertEqual(len(trials), 135)
        self.assertEqual(len({row["trial_id"] for row in trials}), 135)

    def test_full_mode_requires_and_retains_375_records(self) -> None:
        rows = [fixture_rows()[0] | {"source_row": str(index)} for index in range(1, 376)]
        selected = select_full_dataset(rows)
        self.assertEqual(len(selected), 375)
        self.assertEqual(selected[0]["source_row"], "1")
        self.assertEqual(selected[-1]["source_row"], "375")
        filtered = select_full_dataset(rows, exclude_three_premise_items=True)
        self.assertEqual(len(filtered), 366)
        self.assertNotIn("234", {row["source_row"] for row in filtered})
        with self.assertRaises(ValueError):
            select_full_dataset(rows[:-1])

    def test_prompt_matches_published_structure(self) -> None:
        prompt = render_prompt(fixture_rows()[0])
        self.assertIn("Premise 1: All A are B.", prompt)
        self.assertIn("Premise 2: All B are C.", prompt)
        self.assertTrue(prompt.endswith("The answer is: "))

    def test_three_premise_extended_item_retains_all_information(self) -> None:
        row = fixture_rows()[0] | {
            "premises_en": "P implies Q. Not Q. Therefore not P.",
            "hypothesis_en": "Not P.",
        }
        prompt = render_prompt(row)
        self.assertIn("Premise 3: Therefore not P.", prompt)


class ParserTests(unittest.TestCase):
    def test_published_parser_uses_final_line_first_token(self) -> None:
        self.assertEqual(
            parse_published("Explanation\nEntailment.")["published_response"],
            "entailment",
        )
        self.assertEqual(parse_published("neither extra")["published_response"], "neutral")
        self.assertFalse(parse_published("unknown")["published_valid"])

    def test_strict_parser_requires_only_one_label(self) -> None:
        self.assertEqual(parse_strict(" CONTRADICTION\n")["strict_response"], "contradiction")
        self.assertFalse(parse_strict("contradiction.")["strict_valid"])
        self.assertFalse(parse_strict("The answer is: entailment")["strict_valid"])


class MetricTests(unittest.TestCase):
    def test_content_accuracy_is_computed_from_gold(self) -> None:
        rows = [
            {
                "content_type": "consistent",
                "gold": "entailment",
                "published_valid": True,
                "published_correct": value,
            }
            for value in (True, False)
        ]
        _, contents = aggregate(rows, "published")
        self.assertEqual(contents[0]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
