"""Build the frozen position-bias trial schedule without calling a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

QUESTION_IDS = (81, 82, 83, 101, 102, 103, 141, 142, 143, 151, 152, 153)
BASELINE_FILE = "vicuna-13b-v1.3.jsonl"
COMPARISON_FILES = (
    "alpaca-13b.jsonl",
    "gpt-3.5-turbo.jsonl",
    "gpt-4.jsonl",
)
REPETITIONS = 3

EXPECTED_SHA256 = {
    "question.jsonl": "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7",
    "judge_prompts.jsonl": "fd283293406d024f44c174b094ef48031d0687a4682fd3a56b29b138f80281b6",
    "alpaca-13b.jsonl": "80f4a8bf4665839d33fa0ea7f4b58824abf88d5a0edc9e7cc68a362fca88ef98",
    "gpt-3.5-turbo.jsonl": "c3c097c21f91e4b44a554d82e5d47516daf7e8240e13b2eca79d211afc1fdfd6",
    "gpt-4.jsonl": "f65cdce00aa8567e9add118cabd34c2866291ed8f5c1d568f8da9a81d231d3e7",
    "vicuna-13b-v1.3.jsonl": "0a9bb7f7505d096e32b8a81f89b322cc09bb76d896484150844ce0787cc4caf5",
}

WARNING = (
    "Avoid any position biases and ensure that the order in which the "
    "responses were presented does not influence your decision."
)
THREE_OPTION_VERDICT = (
    "After providing your explanation, output your final verdict by strictly "
    'following this format: "[[A]]" if assistant A is better, "[[B]]" if '
    'assistant B is better, and "[[C]]" for a tie.'
)
TWO_OPTION_VERDICT = (
    "After providing your explanation, output your final verdict by strictly "
    'following this format: "[[A]]" if assistant A is better or "[[B]]" if '
    "assistant B is better."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(input_dir: Path) -> dict[str, str]:
    observed = {}
    for filename, expected in EXPECTED_SHA256.items():
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing required source: {path}")
        observed[filename] = sha256_file(path)
        if observed[filename] != expected:
            raise ValueError(
                f"checksum mismatch for {filename}: expected {expected}, "
                f"observed {observed[filename]}"
            )
    return observed


def load_jsonl(path: Path, key: str) -> dict[Any, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_key = record[key]
            if record_key in records:
                raise ValueError(f"duplicate {key}={record_key!r} in {path}")
            records[record_key] = record
    return records


def two_turn_answer(record: dict[str, Any], source_name: str) -> list[str]:
    choices = record.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError(f"{source_name} must contain exactly one choice")
    turns = choices[0].get("turns")
    if not isinstance(turns, list) or len(turns) != 2:
        raise ValueError(f"{source_name} must contain exactly two answer turns")
    if not all(isinstance(turn, str) and turn.strip() for turn in turns):
        raise ValueError(f"{source_name} contains an empty or non-text turn")
    return turns


def forced_choice_system_prompt(system_prompt: str) -> str:
    if system_prompt.count(THREE_OPTION_VERDICT) != 1:
        raise ValueError("the expected three-option verdict text was not found exactly once")
    return system_prompt.replace(THREE_OPTION_VERDICT, TWO_OPTION_VERDICT)


def warning_ablation(system_prompt: str) -> str:
    if system_prompt.count(WARNING) != 1:
        raise ValueError("the expected position-bias warning was not found exactly once")
    warning_with_separator = f"{WARNING} "
    if warning_with_separator not in system_prompt:
        raise ValueError("the expected warning separator was not found")
    return system_prompt.replace(warning_with_separator, "", 1)


def stable_trial_id(factors: dict[str, Any]) -> str:
    encoded = json.dumps(factors, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_trials(
    input_dir: Path,
    seed: int,
    *,
    question_ids: tuple[int, ...] = QUESTION_IDS,
    comparison_files: tuple[str, ...] = COMPARISON_FILES,
) -> list[dict[str, Any]]:
    questions = load_jsonl(input_dir / "question.jsonl", "question_id")
    prompts = load_jsonl(input_dir / "judge_prompts.jsonl", "name")
    prompt = prompts["pair-v2-multi-turn"]
    baseline_prompt = forced_choice_system_prompt(prompt["system_prompt"])
    conditions = {
        "original": baseline_prompt,
        "warning_ablation": warning_ablation(baseline_prompt),
    }

    answer_files = (BASELINE_FILE,) + comparison_files
    answers = {
        filename: load_jsonl(input_dir / filename, "question_id") for filename in answer_files
    }
    trials = []
    for question_id in question_ids:
        question = questions.get(question_id)
        if question is None:
            raise ValueError(f"missing selected question_id={question_id}")
        question_turns = question.get("turns")
        if not isinstance(question_turns, list) or len(question_turns) != 2:
            raise ValueError(f"question_id={question_id} must have exactly two turns")

        baseline_record = answers[BASELINE_FILE].get(question_id)
        if baseline_record is None:
            raise ValueError(f"{BASELINE_FILE} missing question_id={question_id}")
        baseline_turns = two_turn_answer(baseline_record, f"{BASELINE_FILE}:{question_id}")

        for comparison_file in comparison_files:
            comparison_record = answers[comparison_file].get(question_id)
            if comparison_record is None:
                raise ValueError(f"{comparison_file} missing question_id={question_id}")
            comparison_turns = two_turn_answer(
                comparison_record, f"{comparison_file}:{question_id}"
            )
            candidates = {
                "baseline": {
                    "file": BASELINE_FILE,
                    "model_id": baseline_record.get("model_id"),
                    "turns": baseline_turns,
                },
                "comparison": {
                    "file": comparison_file,
                    "model_id": comparison_record.get("model_id"),
                    "turns": comparison_turns,
                },
            }
            for order in ("baseline_comparison", "comparison_baseline"):
                a_key, b_key = order.split("_")
                for prompt_condition, system_prompt in conditions.items():
                    for repetition in range(1, REPETITIONS + 1):
                        factors = {
                            "question_id": question_id,
                            "comparison_file": comparison_file,
                            "order": order,
                            "prompt_condition": prompt_condition,
                            "repetition": repetition,
                            "seed": seed,
                        }
                        rendered = prompt["prompt_template"].format(
                            question_1=question_turns[0],
                            question_2=question_turns[1],
                            answer_a_1=candidates[a_key]["turns"][0],
                            answer_a_2=candidates[a_key]["turns"][1],
                            answer_b_1=candidates[b_key]["turns"][0],
                            answer_b_2=candidates[b_key]["turns"][1],
                        )
                        trials.append(
                            {
                                "trial_id": stable_trial_id(factors),
                                **factors,
                                "category": question.get("category"),
                                "candidate_a": a_key,
                                "candidate_b": b_key,
                                "candidate_a_source": candidates[a_key]["file"],
                                "candidate_b_source": candidates[b_key]["file"],
                                "candidate_a_model_id": candidates[a_key]["model_id"],
                                "candidate_b_model_id": candidates[b_key]["model_id"],
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": rendered},
                                ],
                            }
                        )
    random.Random(seed).shuffle(trials)
    for schedule_index, trial in enumerate(trials, start=1):
        trial["schedule_index"] = schedule_index
    return trials


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(
    input_dir: Path,
    output_dir: Path,
    seed: int,
    *,
    question_ids: tuple[int, ...] = QUESTION_IDS,
    comparison_files: tuple[str, ...] = COMPARISON_FILES,
) -> dict[str, Any]:
    if not question_ids or len(set(question_ids)) != len(question_ids):
        raise ValueError("question_ids must be nonempty and unique")
    if not comparison_files or len(set(comparison_files)) != len(comparison_files):
        raise ValueError("comparison_files must be nonempty and unique")
    unknown = set(comparison_files) - set(COMPARISON_FILES)
    if unknown:
        raise ValueError(f"unknown comparison files: {sorted(unknown)}")
    source_hashes = verify_sources(input_dir)
    trials = build_trials(
        input_dir,
        seed,
        question_ids=question_ids,
        comparison_files=comparison_files,
    )
    expected_trials = len(question_ids) * len(comparison_files) * 2 * 2 * REPETITIONS
    if len(trials) != expected_trials:
        raise AssertionError(f"expected {expected_trials} trials, built {len(trials)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = output_dir / "trials.jsonl"
    write_jsonl(schedule_path, trials)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "question_ids": list(question_ids),
        "baseline_file": BASELINE_FILE,
        "comparison_files": list(comparison_files),
        "prompt_conditions": ["original", "warning_ablation"],
        "orders": ["baseline_comparison", "comparison_baseline"],
        "repetitions": REPETITIONS,
        "trial_count": len(trials),
        "source_sha256": source_hashes,
        "schedule_sha256": sha256_file(schedule_path),
    }
    manifest_path = output_dir / "build-manifest.json"
    with manifest_path.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--question-ids", type=int, nargs="+")
    parser.add_argument("--comparison-files", nargs="+", choices=COMPARISON_FILES)
    args = parser.parse_args()
    manifest = build(
        args.input_dir,
        args.output_dir,
        args.seed,
        question_ids=tuple(args.question_ids or QUESTION_IDS),
        comparison_files=tuple(args.comparison_files or COMPARISON_FILES),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
