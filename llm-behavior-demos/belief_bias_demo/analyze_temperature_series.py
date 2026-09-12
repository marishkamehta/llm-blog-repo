"""Combine completed NeuBAROCO analyses across a temperature series."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

TEMPERATURES = [round(value / 10, 1) for value in range(11)]
THREE_PREMISE_SOURCE_ROWS = {234, 235, 236, 237, 238, 239, 250, 251, 252}
COLORS = {
    "consistent": "#0072B2",
    "inconsistent": "#D55E00",
    "symbol": "#009E73",
    "others": "#7A5195",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def temperature_label(value: float) -> str:
    return f"temp-{value:.1f}".replace(".", "p")


def build_summaries(
    analyses_dir: Path, excluded_source_rows: set[int] | None = None
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    excluded_source_rows = excluded_source_rows or set()
    overall: list[dict[str, object]] = []
    content: list[dict[str, object]] = []
    for temperature in TEMPERATURES:
        folder = analyses_dir / temperature_label(temperature)
        for parser in ("published", "strict"):
            trials = [
                row
                for row in read_csv(folder / "parsed-trials.csv")
                if int(row["source_row"]) not in excluded_source_rows
            ]
            items = [
                row
                for row in read_csv(folder / f"{parser}-item-summaries.csv")
                if int(row["source_row"]) not in excluded_source_rows
            ]
            valid = [row for row in trials if row[f"{parser}_valid"] == "True"]
            correct = sum(row[f"{parser}_correct"] == "True" for row in valid)
            modal_correct = sum(row["modal_correct"] == "True" for row in items)
            lower, upper = wilson(correct, len(valid))
            overall.append(
                {
                    "temperature": f"{temperature:.1f}",
                    "parser": parser,
                    "scheduled_trials": len(trials),
                    "valid_trials": len(valid),
                    "valid_rate": len(valid) / len(trials),
                    "correct_trials": correct,
                    "trial_accuracy": correct / len(valid),
                    "trial_accuracy_ci95_low": lower,
                    "trial_accuracy_ci95_high": upper,
                    "items": len(items),
                    "modal_correct_items": modal_correct,
                    "modal_item_accuracy": modal_correct / len(items),
                    "mean_repetition_stability": sum(
                        float(row["repetition_stability"]) for row in items
                    )
                    / len(items),
                }
            )
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in trials:
                groups[row["content_type"]].append(row)
            item_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in items:
                item_groups[row["content_type"]].append(row)
            for content_type in sorted(groups):
                rows = groups[content_type]
                usable = [row for row in rows if row[f"{parser}_valid"] == "True"]
                successes = sum(row[f"{parser}_correct"] == "True" for row in usable)
                item_rows = item_groups[content_type]
                item_successes = sum(row["modal_correct"] == "True" for row in item_rows)
                low, high = wilson(successes, len(usable))
                content.append(
                    {
                        "temperature": f"{temperature:.1f}",
                        "parser": parser,
                        "content_type": content_type,
                        "scheduled_trials": len(rows),
                        "valid_trials": len(usable),
                        "correct_trials": successes,
                        "trial_accuracy": successes / len(usable),
                        "trial_accuracy_ci95_low": low,
                        "trial_accuracy_ci95_high": high,
                        "items": len(item_rows),
                        "modal_correct_items": item_successes,
                        "modal_item_accuracy": item_successes / len(item_rows),
                        "mean_repetition_stability": sum(
                            float(row["repetition_stability"]) for row in item_rows
                        )
                        / len(item_rows),
                    }
                )
    return overall, content


def make_svg(
    path: Path, rows: list[dict[str, object]], metric: str, title: str, y_label: str
) -> None:
    rows = [row for row in rows if row["parser"] == "published"]
    width, height = 900, 540
    left, right, top, bottom = 90, 35, 65, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [float(row[metric]) for row in rows]
    ymin = max(0.0, math.floor((min(values) - 0.05) * 10) / 10)
    ymax = min(1.0, math.ceil((max(values) + 0.05) * 10) / 10)
    if ymax <= ymin:
        ymax = min(1.0, ymin + 0.1)

    def x(temp: float) -> float:
        return left + float(temp) * plot_w

    def y(value: float) -> float:
        return top + (ymax - float(value)) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="600">{title}</text>',
    ]
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        yy = y(value)
        parts += [
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#dddddd"/>',
            f'<text x="{left - 12}" y="{yy + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.2f}</text>',
        ]
    for i in range(11):
        xx = x(i / 10)
        parts += [
            f'<line x1="{xx:.1f}" y1="{height - bottom}" x2="{xx:.1f}" y2="{height - bottom + 5}" stroke="#333"/>',
            f'<text x="{xx:.1f}" y="{height - bottom + 25}" text-anchor="middle" font-family="sans-serif" font-size="13">{i / 10:.1f}</text>',
        ]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["content_type"])].append(row)
    legend_x = left
    for content_type in ("consistent", "inconsistent", "symbol", "others"):
        points = sorted(grouped[content_type], key=lambda row: float(row["temperature"]))
        color = COLORS[content_type]
        coordinates = " ".join(
            f"{x(float(row['temperature'])):.1f},{y(float(row[metric])):.1f}" for row in points
        )
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for row in points:
            parts.append(
                f'<circle cx="{x(float(row["temperature"])):.1f}" cy="{y(float(row[metric])):.1f}" r="4" fill="{color}"/>'
            )
        parts += [
            f'<line x1="{legend_x}" y1="{height - 18}" x2="{legend_x + 24}" y2="{height - 18}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{legend_x + 30}" y="{height - 13}" font-family="sans-serif" font-size="13">{content_type}</text>',
        ]
        legend_x += 175
    parts += [
        f'<text x="{left + plot_w / 2}" y="{height - 42}" text-anchor="middle" font-family="sans-serif" font-size="15">Temperature</text>',
        f'<text transform="translate(22 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="15">{y_label}</text>',
        "</svg>",
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyses-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-three-premise-items",
        action="store_true",
        help="exclude the nine records whose published prompts require three premises",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    excluded_source_rows = THREE_PREMISE_SOURCE_ROWS if args.exclude_three_premise_items else set()
    overall, content = build_summaries(args.analyses_dir, excluded_source_rows)
    write_csv(args.output_dir / "temperature-overall-summary.csv", overall)
    write_csv(args.output_dir / "temperature-content-summary.csv", content)
    make_svg(
        args.output_dir / "accuracy-by-temperature.svg",
        content,
        "trial_accuracy",
        "NeuBAROCO accuracy across temperatures",
        "Trial accuracy",
    )
    make_svg(
        args.output_dir / "stability-by-temperature.svg",
        content,
        "mean_repetition_stability",
        "Response stability across temperatures",
        "Mean repetition stability",
    )
    outputs = sorted(args.output_dir.glob("*"))
    manifest = {
        "schema_version": 2,
        "temperatures": TEMPERATURES,
        "parsers": ["published", "strict"],
        "excluded_source_rows": sorted(excluded_source_rows),
        "exclusion_reason": (
            "three-premise items excluded because the accompanying published prompt formatter "
            "omits the third premise, which can change the correct logical classification"
            if excluded_source_rows
            else None
        ),
        "confidence_interval": "Wilson 95% interval over valid trials; descriptive because repetitions and temperatures reuse items",
        "output_sha256": {path.name: sha256(path) for path in outputs},
    }
    (args.output_dir / "series-analysis-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
