"""Parse judge verdicts and summarize stability and position consistency."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from demo_records import (
    canonical_sha256,
    completed_trial_ids,
    load_json,
    load_schedule,
    sha256_file,
)

VERDICT = re.compile(r"\[\[([ABC])\]\]")


def parse_verdict(text: str) -> dict[str, Any]:
    """Primary rule: accept one unique A/B token anywhere in the response."""
    matches = list(VERDICT.finditer(text))
    if not matches:
        return {"valid": False, "verdict": None, "reason": "missing_verdict"}
    if len(matches) != 1:
        return {"valid": False, "verdict": None, "reason": "multiple_verdicts"}
    if matches[0].group(1) == "C":
        return {"valid": False, "verdict": None, "reason": "tie_not_allowed"}
    return {"valid": True, "verdict": matches[0].group(1), "reason": None}


def parse_verdict_fastchat(text: str) -> dict[str, Any]:
    """Literal FastChat rule: A-first, then B, then C containment."""
    if "[[A]]" in text:
        return {"valid": True, "verdict": "A", "reason": None}
    if "[[B]]" in text:
        return {"valid": True, "verdict": "B", "reason": None}
    if "[[C]]" in text:
        return {"valid": True, "verdict": "C", "reason": None}
    return {"valid": False, "verdict": None, "reason": "missing_verdict"}


def load_raw_by_trial(raw_dir: Path) -> dict[str, dict[str, Any]]:
    completed_trial_ids(raw_dir)  # Performs duplicate and record-hash checks.
    records = {}
    for path in sorted(raw_dir.glob("*.json")):
        record = load_json(path)
        records[record["trial_id"]] = record
    return records


def parse_trials(
    schedule: list[dict[str, Any]], raw: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    parsed = []
    for trial in schedule:
        record = raw.get(trial["trial_id"])
        base = {
            "trial_id": trial["trial_id"],
            "schedule_index": trial["schedule_index"],
            "question_id": trial["question_id"],
            "category": trial["category"],
            "comparison_file": trial["comparison_file"],
            "order": trial["order"],
            "prompt_condition": trial["prompt_condition"],
            "repetition": trial["repetition"],
            "candidate_a": trial["candidate_a"],
            "candidate_b": trial["candidate_b"],
        }
        if record is None:
            parsed.append(
                {
                    **base,
                    "collection_status": "missing",
                    "raw_record_sha256": None,
                    "valid": False,
                    "verdict": None,
                    "selected_candidate": None,
                    "parse_reason": "missing_raw_record",
                    "fastchat_valid": False,
                    "fastchat_verdict": None,
                    "fastchat_selected_candidate": None,
                    "fastchat_parse_reason": "missing_raw_record",
                }
            )
            continue
        expected_trial_hash = canonical_sha256(trial)
        if record.get("schedule_trial_sha256") != expected_trial_hash:
            raise ValueError(f"schedule-trial checksum mismatch: {trial['trial_id']}")
        result = parse_verdict(record.get("raw_text", ""))
        fastchat_result = parse_verdict_fastchat(record.get("raw_text", ""))
        verdict = result["verdict"]
        selected_candidate = None
        if verdict == "A":
            selected_candidate = trial["candidate_a"]
        elif verdict == "B":
            selected_candidate = trial["candidate_b"]
        fastchat_verdict = fastchat_result["verdict"]
        fastchat_selected_candidate = None
        if fastchat_verdict == "A":
            fastchat_selected_candidate = trial["candidate_a"]
        elif fastchat_verdict == "B":
            fastchat_selected_candidate = trial["candidate_b"]
        parsed.append(
            {
                **base,
                "collection_status": "collected",
                "raw_record_sha256": record["record_sha256"],
                "valid": result["valid"],
                "verdict": verdict,
                "selected_candidate": selected_candidate,
                "parse_reason": result["reason"],
                "fastchat_valid": fastchat_result["valid"],
                "fastchat_verdict": fastchat_verdict,
                "fastchat_selected_candidate": fastchat_selected_candidate,
                "fastchat_parse_reason": fastchat_result["reason"],
            }
        )
    return parsed


def unique_mode(values: list[str]) -> tuple[str | None, int]:
    if not values:
        return None, 0
    counts = Counter(values)
    highest = max(counts.values())
    modes = [value for value, count in counts.items() if count == highest]
    return (modes[0], highest) if len(modes) == 1 else (None, highest)


def summarize_repetitions(
    parsed: list[dict[str, Any]],
    *,
    valid_key: str = "valid",
    verdict_key: str = "verdict",
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        key = (
            row["question_id"],
            row["comparison_file"],
            row["order"],
            row["prompt_condition"],
        )
        groups[key].append(row)

    summaries = []
    for key, rows in sorted(groups.items()):
        valid_verdicts = [row[verdict_key] for row in rows if row[valid_key]]
        modal_verdict, modal_count = unique_mode(valid_verdicts)
        scheduled = len(rows)
        valid_n = len(valid_verdicts)
        summaries.append(
            {
                "question_id": key[0],
                "comparison_file": key[1],
                "order": key[2],
                "prompt_condition": key[3],
                "scheduled_n": scheduled,
                "collected_n": sum(row["collection_status"] == "collected" for row in rows),
                "valid_n": valid_n,
                "invalid_n": scheduled - valid_n,
                "modal_verdict": modal_verdict,
                "modal_count": modal_count,
                "stability_valid": modal_count / valid_n if valid_n else None,
                "stability_all_scheduled": modal_count / scheduled if scheduled else None,
            }
        )
    return summaries


def underlying_candidate(order: str, verdict: str | None) -> str | None:
    if verdict not in {"A", "B"}:
        return None
    if order == "baseline_comparison":
        return "baseline" if verdict == "A" else "comparison"
    if order == "comparison_baseline":
        return "comparison" if verdict == "A" else "baseline"
    raise ValueError(f"unknown order: {order}")


def summarize_position(
    repetition_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in repetition_rows:
        key = (
            row["question_id"],
            row["comparison_file"],
            row["prompt_condition"],
        )
        groups[key][row["order"]] = row

    output = []
    for key, orders in sorted(groups.items()):
        first = orders.get("baseline_comparison")
        second = orders.get("comparison_baseline")
        if first is None or second is None:
            raise ValueError(f"missing an answer order for {key}")
        first_verdict = first["modal_verdict"]
        second_verdict = second["modal_verdict"]
        first_candidate = underlying_candidate(first["order"], first_verdict)
        second_candidate = underlying_candidate(second["order"], second_verdict)
        evaluable = first_candidate is not None and second_candidate is not None
        consistent = first_candidate == second_candidate if evaluable else None
        preference = None
        if evaluable and not consistent:
            if first_verdict == second_verdict == "A":
                preference = "primacy"
            elif first_verdict == second_verdict == "B":
                preference = "recency"
            else:
                raise AssertionError("unexpected two-option inconsistency pattern")
        output.append(
            {
                "question_id": key[0],
                "comparison_file": key[1],
                "prompt_condition": key[2],
                "baseline_first_modal_verdict": first_verdict,
                "comparison_first_modal_verdict": second_verdict,
                "baseline_first_selected_candidate": first_candidate,
                "comparison_first_selected_candidate": second_candidate,
                "evaluable": evaluable,
                "position_consistent": consistent,
                "inconsistent_preference": preference,
            }
        )
    return output


def aggregate(
    parsed: list[dict[str, Any]],
    repetitions: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    *,
    valid_key: str = "valid",
) -> list[dict[str, Any]]:
    conditions = sorted({row["prompt_condition"] for row in parsed})
    output = []
    for condition in conditions:
        trial_rows = [row for row in parsed if row["prompt_condition"] == condition]
        rep_rows = [row for row in repetitions if row["prompt_condition"] == condition]
        pos_rows = [row for row in positions if row["prompt_condition"] == condition]
        evaluable = [row for row in pos_rows if row["evaluable"]]
        inconsistent = [row for row in evaluable if row["position_consistent"] is False]
        valid_stabilities = [
            row["stability_valid"] for row in rep_rows if row["stability_valid"] is not None
        ]
        scheduled_stabilities = [row["stability_all_scheduled"] for row in rep_rows]
        output.append(
            {
                "prompt_condition": condition,
                "scheduled_trials": len(trial_rows),
                "collected_trials": sum(
                    row["collection_status"] == "collected" for row in trial_rows
                ),
                "valid_trials": sum(row[valid_key] for row in trial_rows),
                "invalid_or_missing_trials": sum(not row[valid_key] for row in trial_rows),
                "invalid_or_missing_rate": (
                    sum(not row[valid_key] for row in trial_rows) / len(trial_rows)
                    if trial_rows
                    else None
                ),
                "mean_repetition_stability_valid": (
                    sum(valid_stabilities) / len(valid_stabilities) if valid_stabilities else None
                ),
                "mean_repetition_stability_all_scheduled": (
                    sum(scheduled_stabilities) / len(scheduled_stabilities)
                    if scheduled_stabilities
                    else None
                ),
                "position_pairs": len(pos_rows),
                "evaluable_position_pairs": len(evaluable),
                "position_consistent_pairs": sum(
                    row["position_consistent"] is True for row in evaluable
                ),
                "position_consistency": (
                    sum(row["position_consistent"] is True for row in evaluable) / len(evaluable)
                    if evaluable
                    else None
                ),
                "primacy_inconsistent_pairs": sum(
                    row["inconsistent_preference"] == "primacy" for row in inconsistent
                ),
                "recency_inconsistent_pairs": sum(
                    row["inconsistent_preference"] == "recency" for row in inconsistent
                ),
            }
        )
    return output


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(schedule_path: Path, run_dir: Path, output_dir: Path) -> dict[str, Any]:
    schedule = load_schedule(schedule_path)
    raw = load_raw_by_trial(run_dir / "raw-responses")
    parsed = parse_trials(schedule, raw)
    repetitions = summarize_repetitions(parsed)
    positions = summarize_position(repetitions)
    summary = aggregate(parsed, repetitions, positions)
    fastchat_repetitions = summarize_repetitions(
        parsed, valid_key="fastchat_valid", verdict_key="fastchat_verdict"
    )
    fastchat_positions = summarize_position(fastchat_repetitions)
    fastchat_summary = aggregate(
        parsed, fastchat_repetitions, fastchat_positions, valid_key="fastchat_valid"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "parsed-trials.csv", parsed)
    write_csv(output_dir / "repetition-cells.csv", repetitions)
    write_csv(output_dir / "position-pairs.csv", positions)
    write_csv(output_dir / "summary-by-prompt.csv", summary)
    write_csv(output_dir / "fastchat-repetition-cells.csv", fastchat_repetitions)
    write_csv(output_dir / "fastchat-position-pairs.csv", fastchat_positions)
    write_csv(output_dir / "fastchat-summary-by-prompt.csv", fastchat_summary)
    table_files = [
        "parsed-trials.csv",
        "repetition-cells.csv",
        "position-pairs.csv",
        "summary-by-prompt.csv",
        "fastchat-repetition-cells.csv",
        "fastchat-position-pairs.csv",
        "fastchat-summary-by-prompt.csv",
    ]
    manifest = {
        "schema_version": 2,
        "schedule_sha256": sha256_file(schedule_path),
        "raw_record_count": len(raw),
        "parsed_trial_count": len(parsed),
        "repetition_cell_count": len(repetitions),
        "position_pair_count": len(positions),
        "parser_rules": {
            "primary": (
                "exactly one case-sensitive [[A]] or [[B]] token anywhere; "
                "reject missing, tie, or multiple tokens"
            ),
            "sensitivity_fastchat_literal": (
                "case-sensitive containment in [[A]], [[B]], [[C]] priority order"
            ),
        },
        "output_sha256": {filename: sha256_file(output_dir / filename) for filename in table_files},
    }
    with (output_dir / "analysis-manifest.json").open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return {
        "manifest": manifest,
        "primary_summary": summary,
        "fastchat_sensitivity_summary": fastchat_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.schedule, args.run_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
