"""Build a frozen NeuBAROCO demonstration schedule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SOURCE_SHA256 = "b3f98f4a8f5f36079b9335c460fcea3b1044884dbb6f0ec627aa1b2b596666a5"
SOURCE_COMMIT = "447929fdabe07bc3d13efae8e0c527fd458df177"
CONTENT_TYPES = ("consistent", "inconsistent", "symbol")
GOLD_LABELS = ("entailment", "contradiction", "neutral")
ITEMS_PER_CELL = 5
REPETITIONS = 3
EXCLUDED_SOURCE_ROWS = {
    190: "malformed published English premise: missing noun and fused token",
}
THREE_PREMISE_SOURCE_ROWS = {234, 235, 236, 237, 238, 239, 250, 251, 252}
INSTRUCTIONS = """Determine the correct logical relationship between the given premises and the hypothesis.
- Answer "entailment" if the hypothesis follows logically from the premises.
- Answer "contradiction" if the premises and the hypothesis are logically incompatible with each other.
- Answer "neither" if the relationship is neither "entailment" nor "contradiction".
Your answer must be one word: "entailment", "contradiction", or "neither"."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_source(path: Path) -> list[dict[str, str]]:
    observed = sha256_file(path)
    if observed != SOURCE_SHA256:
        raise ValueError(f"source checksum mismatch: expected {SOURCE_SHA256}, observed {observed}")
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    required = {
        "premises_en",
        "hypothesis_en",
        "gold",
        "figure",
        "conversion",
        "content-type",
        "atmosphere",
        "syllogism-type",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("NeuBAROCO source schema is missing required columns")
    for source_row, row in enumerate(rows, start=1):
        row["source_row"] = str(source_row)
        if not row["premises_en"].strip() or not row["hypothesis_en"].strip():
            raise ValueError(f"blank English stimulus at source row {source_row}")
    return rows


def select_items(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    eligible = [
        row
        for row in rows
        if row["syllogism-type"] == "syllogism"
        and row["content-type"] in CONTENT_TYPES
        and row["gold"] in GOLD_LABELS
        and int(row["source_row"]) not in EXCLUDED_SOURCE_ROWS
    ]
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        groups[(row["content-type"], row["gold"])].append(row)
    selected = []
    for content_type in CONTENT_TYPES:
        for gold in GOLD_LABELS:
            cell = groups[(content_type, gold)]
            if len(cell) < ITEMS_PER_CELL:
                raise ValueError(f"insufficient items in cell {(content_type, gold)}")
            cell_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{content_type}:{gold}".encode()).digest()[:8],
                "big",
            )
            selected.extend(random.Random(cell_seed).sample(cell, ITEMS_PER_CELL))
    if Counter((row["content-type"], row["gold"]) for row in selected) != Counter(
        {(content, gold): ITEMS_PER_CELL for content in CONTENT_TYPES for gold in GOLD_LABELS}
    ):
        raise AssertionError("selected subset is not balanced")
    return sorted(selected, key=lambda row: int(row["source_row"]))


def select_full_dataset(
    rows: list[dict[str, str]], exclude_three_premise_items: bool = False
) -> list[dict[str, str]]:
    """Retain the published NALOMA records in source order."""
    if len(rows) != 375:
        raise ValueError(f"expected the complete 375-record source, found {len(rows)}")
    if exclude_three_premise_items:
        return [row for row in rows if int(row["source_row"]) not in THREE_PREMISE_SOURCE_ROWS]
    return list(rows)


def split_premises(text: str) -> list[str]:
    parts = [part.strip() for part in text.split(".") if part.strip()]
    if len(parts) not in {2, 3}:
        raise ValueError(f"expected two or three period-delimited premises: {text!r}")
    return parts


def render_prompt(row: dict[str, str]) -> str:
    premises = split_premises(row["premises_en"])
    hypothesis = row["hypothesis_en"].strip()
    return (
        f"{INSTRUCTIONS}\n"
        + "".join(f"Premise {index}: {premise}.\n" for index, premise in enumerate(premises, 1))
        + f"Hypothesis: {hypothesis}\n"
        "The answer is: "
    )


def build_trials(items: list[dict[str, str]], seed: int) -> list[dict[str, Any]]:
    trials = []
    for row in items:
        item_id = f"neubaroco-naloma-{int(row['source_row']):03d}"
        expected = "neither" if row["gold"] == "neutral" else row["gold"]
        for repetition in range(1, REPETITIONS + 1):
            factors = {"item_id": item_id, "repetition": repetition, "seed": seed}
            trials.append(
                {
                    "trial_id": canonical_sha256(factors)[:20],
                    **factors,
                    "source_row": int(row["source_row"]),
                    "content_type": row["content-type"],
                    "gold": row["gold"],
                    "expected_response": expected,
                    "mood": row["figure"],
                    "conversion": row["conversion"],
                    "atmosphere": row["atmosphere"],
                    "messages": [{"role": "user", "content": render_prompt(row)}],
                }
            )
    random.Random(seed).shuffle(trials)
    for index, trial in enumerate(trials, start=1):
        trial["schedule_index"] = index
    return trials


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("balanced-subset", "full"), default="balanced-subset")
    parser.add_argument("--exclude-three-premise-items", action="store_true")
    args = parser.parse_args()
    rows = load_source(args.source)
    selected = (
        select_full_dataset(rows, args.exclude_three_premise_items)
        if args.mode == "full"
        else select_items(rows, args.seed)
    )
    trials = build_trials(selected, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "selected-items.jsonl", selected)
    write_jsonl(args.output_dir / "trials.jsonl", trials)
    manifest = {
        "schema_version": 2,
        "study_type": (
            "published NeuBAROCO NALOMA experiment with repeated responses"
            if args.mode == "full"
            else "balanced demonstration using the published NeuBAROCO NLI experiment"
        ),
        "source_repository": "https://github.com/kmineshima/NeuBAROCO",
        "source_commit": SOURCE_COMMIT,
        "source_file": "acl2024/NeuBAROCO_NALOMA.tsv",
        "source_license": "CC BY 4.0",
        "source_sha256": SOURCE_SHA256,
        "selection_mode": args.mode,
        "selection_rule": (
            (
                "all two-premise published records in source order"
                if args.exclude_three_premise_items
                else "all 375 published records in source order"
            )
            if args.mode == "full"
            else "five basic syllogisms per content-type x gold cell; seeded within-cell sampling"
        ),
        "prespecified_exclusions": (
            {
                str(
                    row
                ): "three-premise item excluded because the published formatter omits its third premise"
                for row in sorted(THREE_PREMISE_SOURCE_ROWS)
            }
            if args.mode == "full" and args.exclude_three_premise_items
            else ({} if args.mode == "full" else EXCLUDED_SOURCE_ROWS)
        ),
        "selection_uses_new_model_outcomes": False,
        "seed": args.seed,
        "item_count": len(selected),
        "trial_count": len(trials),
        "repetitions": REPETITIONS,
        "selected_items_sha256": sha256_file(args.output_dir / "selected-items.jsonl"),
        "schedule_sha256": sha256_file(args.output_dir / "trials.jsonl"),
        "prompt": INSTRUCTIONS,
        "three_premise_items_excluded": args.exclude_three_premise_items,
    }
    with (args.output_dir / "build-manifest.json").open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
