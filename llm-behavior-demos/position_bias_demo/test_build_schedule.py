from __future__ import annotations

import unittest

from position_bias_demo.build_schedule import (
    THREE_OPTION_VERDICT,
    TWO_OPTION_VERDICT,
    WARNING,
    forced_choice_system_prompt,
    stable_trial_id,
    warning_ablation,
)


class PromptTransformationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = f"Judge fairly. {WARNING} Explain briefly. {THREE_OPTION_VERDICT}"

    def test_forced_choice_removes_tie_without_removing_warning(self) -> None:
        transformed = forced_choice_system_prompt(self.source)
        self.assertIn(WARNING, transformed)
        self.assertIn(TWO_OPTION_VERDICT, transformed)
        self.assertNotIn("[[C]]", transformed)

    def test_warning_ablation_changes_only_warning(self) -> None:
        baseline = forced_choice_system_prompt(self.source)
        ablated = warning_ablation(baseline)
        self.assertNotIn(WARNING, ablated)
        self.assertEqual(ablated, baseline.replace(f"{WARNING} ", "", 1))

    def test_transformations_fail_if_source_text_drifted(self) -> None:
        with self.assertRaises(ValueError):
            forced_choice_system_prompt("A different upstream prompt")
        with self.assertRaises(ValueError):
            warning_ablation("A prompt with no warning")


class TrialIdentifierTests(unittest.TestCase):
    def test_identifier_is_stable_across_dictionary_order(self) -> None:
        first = stable_trial_id({"question_id": 81, "order": "xy"})
        second = stable_trial_id({"order": "xy", "question_id": 81})
        self.assertEqual(first, second)

    def test_identifier_changes_with_condition(self) -> None:
        first = stable_trial_id({"condition": "original"})
        second = stable_trial_id({"condition": "warning_ablation"})
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
