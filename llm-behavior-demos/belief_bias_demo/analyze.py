"""Parse and summarize the NeuBAROCO demonstration."""

from __future__ import annotations

import argparse
import csv
import json
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

LABEL_MAP = {
    "entailment": "entailment",
    "contradiction": "contradiction",
    "neither": "neutral",
}


def parse_published(text: str) -> dict[str, Any]:
    """Reproduce the official English parser: final line, first lowercased token."""
    cleaned = text.replace("The answer is: ", "")
    line = cleaned.strip().split("\n")[-1] if cleaned.strip() else ""
    token = line.strip().split()[0].lower() if line.strip() else ""
    if token.endswith("."):
        token = token[:-1]
    response = LABEL_MAP.get(token)
    return {
        "published_valid": response is not None,
        "published_response": response,
        "published_parse_reason": None if response else "unrecognized_final_line_first_token",
    }


def parse_strict(text: str) -> dict[str, Any]:
    token = text.strip().lower()
    response = LABEL_MAP.get(token)
    return {
        "strict_valid": response is not None,
        "strict_response": response,
        "strict_parse_reason": None if response else "not_exact_label",
    }


def load_raw(raw_dir: Path) -> dict[str, dict[str, Any]]:
    completed_trial_ids(raw_dir)
    return {load_json(path)["trial_id"]: load_json(path) for path in sorted(raw_dir.glob("*.json"))}


def parse_trials(
    schedule: list[dict[str, Any]], raw: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    keys = (
        "trial_id",
        "schedule_index",
        "item_id",
        "source_row",
        "repetition",
        "content_type",
        "gold",
        "expected_response",
        "mood",
        "conversion",
        "atmosphere",
    )
    for trial in schedule:
        base = {key: trial[key] for key in keys}
        record = raw.get(trial["trial_id"])
        if record is None:
            output.append(
                {
                    **base,
                    "collection_status": "missing",
                    "raw_record_sha256": None,
                    "published_valid": False,
                    "published_response": None,
                    "published_parse_reason": "missing_raw_record",
                    "published_correct": False,
                    "strict_valid": False,
                    "strict_response": None,
                    "strict_parse_reason": "missing_raw_record",
                    "strict_correct": False,
                }
            )
            continue
        if record.get("schedule_trial_sha256") != canonical_sha256(trial):
            raise ValueError(f"schedule-trial checksum mismatch: {trial['trial_id']}")
        published = parse_published(record.get("raw_text", ""))
        strict = parse_strict(record.get("raw_text", ""))
        output.append(
            {
                **base,
                "collection_status": "collected",
                "raw_record_sha256": record["record_sha256"],
                **published,
                "published_correct": published["published_response"] == trial["gold"],
                **strict,
                "strict_correct": strict["strict_response"] == trial["gold"],
            }
        )
    return output


def item_summaries(parsed: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        groups[row["item_id"]].append(row)
    output = []
    for item_id, rows in sorted(groups.items()):
        responses = [row[f"{prefix}_response"] for row in rows if row[f"{prefix}_valid"]]
        counts = Counter(responses)
        mode = None
        if counts:
            high = max(counts.values())
            modes = [value for value, count in counts.items() if count == high]
            mode = modes[0] if len(modes) == 1 else None
        first = rows[0]
        output.append(
            {
                "item_id": item_id,
                "source_row": first["source_row"],
                "content_type": first["content_type"],
                "gold": first["gold"],
                "mood": first["mood"],
                "conversion": first["conversion"],
                "atmosphere": first["atmosphere"],
                "scheduled_n": len(rows),
                "valid_n": len(responses),
                "modal_response": mode,
                "modal_correct": mode == first["gold"] if mode else False,
                "repetition_stability": max(counts.values()) / len(responses)
                if responses
                else None,
            }
        )
    return output


def aggregate(
    parsed: list[dict[str, Any]], prefix: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        cell_groups[(row["content_type"], row["gold"])].append(row)
    cells = []
    for key, rows in sorted(cell_groups.items()):
        valid = [row for row in rows if row[f"{prefix}_valid"]]
        cells.append(
            {
                "content_type": key[0],
                "gold": key[1],
                "scheduled_trials": len(rows),
                "valid_trials": len(valid),
                "invalid_or_missing_trials": len(rows) - len(valid),
                "accuracy": sum(row[f"{prefix}_correct"] for row in valid) / len(valid)
                if valid
                else None,
            }
        )
    contents = []
    for content_type in sorted({row["content_type"] for row in parsed}):
        rows = [row for row in parsed if row["content_type"] == content_type]
        valid = [row for row in rows if row[f"{prefix}_valid"]]
        contents.append(
            {
                "content_type": content_type,
                "scheduled_trials": len(rows),
                "valid_trials": len(valid),
                "invalid_or_missing_trials": len(rows) - len(valid),
                "accuracy": sum(row[f"{prefix}_correct"] for row in valid) / len(valid)
                if valid
                else None,
            }
        )
    return cells, contents


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    with path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(schedule_path: Path, run_dir: Path, output_dir: Path) -> dict[str, Any]:
    parsed = parse_trials(load_schedule(schedule_path), load_raw(run_dir / "raw-responses"))
    published_items = item_summaries(parsed, "published")
    strict_items = item_summaries(parsed, "strict")
    published_cells, published_contents = aggregate(parsed, "published")
    strict_cells, strict_contents = aggregate(parsed, "strict")
    output_dir.mkdir(parents=True, exist_ok=False)
    tables = {
        "parsed-trials.csv": parsed,
        "published-item-summaries.csv": published_items,
        "published-cell-summaries.csv": published_cells,
        "published-content-summaries.csv": published_contents,
        "strict-item-summaries.csv": strict_items,
        "strict-cell-summaries.csv": strict_cells,
        "strict-content-summaries.csv": strict_contents,
    }
    for filename, rows in tables.items():
        write_csv(output_dir / filename, rows)
    manifest = {
        "schema_version": 2,
        "schedule_sha256": sha256_file(schedule_path),
        "raw_record_count": sum(row["collection_status"] == "collected" for row in parsed),
        "parsed_trial_count": len(parsed),
        "parser_rules": {
            "primary_published": "official final-line, first-token, lowercase rule; optional period",
            "sensitivity_strict": "entire stripped response must be entailment, contradiction, or neither",
        },
        "output_sha256": {name: sha256_file(output_dir / name) for name in tables},
    }
    with (output_dir / "analysis-manifest.json").open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return {
        "manifest": manifest,
        "published_content_summary": published_contents,
        "strict_content_summary": strict_contents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.schedule, args.run_dir, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
