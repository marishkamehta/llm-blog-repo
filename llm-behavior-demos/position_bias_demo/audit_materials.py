"""Outcome-blind audit of the frozen position-bias demonstration materials."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from position_bias_demo.build_schedule import (
    BASELINE_FILE,
    COMPARISON_FILES,
    QUESTION_IDS,
    build_trials,
    load_jsonl,
    sha256_file,
    two_turn_answer,
    verify_sources,
)

GAP_THRESHOLD = 2.0
VERDICT_TOKENS = ("[[A]]", "[[B]]", "[[C]]")


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def tokenize_messages(endpoint: str, model: str, messages: list[dict[str, str]]) -> int:
    response = requests.post(
        endpoint.rstrip("/") + "/tokenize",
        json={"model": model, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    return int(response.json()["count"])


def audit(
    *,
    input_dir: Path,
    single_judgments: Path,
    pair_judgments: Path,
    output_dir: Path,
    tokenizer_endpoint: str,
    tokenizer_model: str,
    seed: int,
) -> dict[str, Any]:
    source_hashes = verify_sources(input_dir)
    questions = load_jsonl(input_dir / "question.jsonl", "question_id")
    answers = {
        filename: load_jsonl(input_dir / filename, "question_id")
        for filename in (BASELINE_FILE,) + COMPARISON_FILES
    }

    scores: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in load_jsonl_rows(single_judgments):
        if row.get("question_id") in QUESTION_IDS and row.get("model") in {
            BASELINE_FILE.removesuffix(".jsonl"),
            *(name.removesuffix(".jsonl") for name in COMPARISON_FILES),
        }:
            scores[(row["question_id"], row["model"])].append(float(row["score"]))

    available_pair_rows = set()
    selected_models = {
        BASELINE_FILE.removesuffix(".jsonl"),
        *(name.removesuffix(".jsonl") for name in COMPARISON_FILES),
    }
    for row in load_jsonl_rows(pair_judgments):
        if row.get("question_id") not in QUESTION_IDS or row.get("turn") != 2:
            continue
        models = frozenset((row.get("model_1"), row.get("model_2")))
        if models <= selected_models:
            available_pair_rows.add((row["question_id"], models))

    trials = build_trials(input_dir, seed)
    token_counts: dict[tuple[int, str], list[int]] = defaultdict(list)
    seen_messages = set()
    for trial in trials:
        message_key = json.dumps(trial["messages"], ensure_ascii=False, sort_keys=True)
        if message_key in seen_messages:
            continue
        seen_messages.add(message_key)
        token_counts[(trial["question_id"], trial["comparison_file"])].append(
            tokenize_messages(tokenizer_endpoint, tokenizer_model, trial["messages"])
        )

    output_rows = []
    for question_id in QUESTION_IDS:
        question = questions[question_id]
        for comparison_file in COMPARISON_FILES:
            baseline_model = BASELINE_FILE.removesuffix(".jsonl")
            comparison_model = comparison_file.removesuffix(".jsonl")
            baseline_turns = two_turn_answer(
                answers[BASELINE_FILE][question_id], f"{BASELINE_FILE}:{question_id}"
            )
            comparison_turns = two_turn_answer(
                answers[comparison_file][question_id],
                f"{comparison_file}:{question_id}",
            )
            baseline_scores = scores[(question_id, baseline_model)]
            comparison_scores = scores[(question_id, comparison_model)]
            if len(baseline_scores) != 2 or len(comparison_scores) != 2:
                raise ValueError(
                    f"expected two single scores for {question_id}, {comparison_model}"
                )
            baseline_mean = sum(baseline_scores) / 2
            comparison_mean = sum(comparison_scores) / 2
            gap = abs(comparison_mean - baseline_mean)
            pair_key = (question_id, frozenset((baseline_model, comparison_model)))
            if comparison_mean > baseline_mean:
                score_reference = comparison_model
            elif baseline_mean > comparison_mean:
                score_reference = baseline_model
            else:
                score_reference = "tie"
            candidate_text = "\n".join(baseline_turns + comparison_turns)
            output_rows.append(
                {
                    "question_id": question_id,
                    "category": question["category"],
                    "comparison_file": comparison_file,
                    "question_turns_complete": len(question.get("turns", [])) == 2,
                    "baseline_turns_complete": len(baseline_turns) == 2,
                    "comparison_turns_complete": len(comparison_turns) == 2,
                    "candidate_contains_verdict_token": any(
                        token in candidate_text for token in VERDICT_TOKENS
                    ),
                    "baseline_characters": sum(map(len, baseline_turns)),
                    "comparison_characters": sum(map(len, comparison_turns)),
                    "max_rendered_chat_tokens": max(token_counts[(question_id, comparison_file)]),
                    "tokenizer_context_limit": 32768,
                    "max_context_fraction": max(token_counts[(question_id, comparison_file)])
                    / 32768,
                    "baseline_published_mean_score": baseline_mean,
                    "comparison_published_mean_score": comparison_mean,
                    "absolute_score_gap": gap,
                    "quality_gap": "clear" if gap >= GAP_THRESHOLD else "close",
                    "published_score_reference": score_reference,
                    "exact_pairwise_judgment_available": pair_key in available_pair_rows,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "item-pair-audit.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    manifest = {
        "schema_version": 1,
        "selection_used_new_qwen_outcomes": False,
        "question_ids": list(QUESTION_IDS),
        "comparison_files": list(COMPARISON_FILES),
        "gap_rule": (
            "clear when absolute difference between two-turn mean published "
            f"GPT-4 single scores is >= {GAP_THRESHOLD}; otherwise close"
        ),
        "reference_rule": (
            "candidate with higher two-turn mean published GPT-4 single score; "
            "equal means are a tie"
        ),
        "exact_pairwise_archive_note": (
            "archived pairwise file contains the 12 vicuna-13b-v1.3 versus "
            "gpt-3.5-turbo rows, but not all 36 selected baseline pairs"
        ),
        "tokenizer_endpoint": tokenizer_endpoint,
        "tokenizer_model": tokenizer_model,
        "source_sha256": source_hashes,
        "single_judgments_sha256": sha256_file(single_judgments),
        "pair_judgments_sha256": sha256_file(pair_judgments),
        "row_count": len(output_rows),
        "output_sha256": sha256_file(csv_path),
    }
    with (output_dir / "audit-manifest.json").open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--single-judgments", type=Path, required=True)
    parser.add_argument("--pair-judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    result = audit(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
