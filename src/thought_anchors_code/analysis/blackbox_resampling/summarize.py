"""Summaries for black-box resampling outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize_resampling_file(input_path: str | Path) -> list[dict]:
    groups: dict[tuple[str, int, bool | None], list[dict]] = defaultdict(list)
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            resamples = payload.get("resamples") or []
            correctness = [
                resample.get("is_correct")
                for resample in resamples
                if resample.get("is_correct") is not None
            ]
            pass_rate = mean(bool(value) for value in correctness) if correctness else None
            changed = (
                None
                if pass_rate is None or payload.get("original_is_correct") is None
                else pass_rate - float(bool(payload.get("original_is_correct")))
            )
            row = {
                "dataset_name": payload.get("dataset_name"),
                "task_id": payload.get("task_id"),
                "sample_id": payload.get("sample_id"),
                "sentence_index": payload.get("sentence_index"),
                "selection": payload.get("selection"),
                "sentence_score": payload.get("sentence_score"),
                "code_sentence_score": payload.get("code_sentence_score"),
                "original_is_correct": payload.get("original_is_correct"),
                "num_resamples": len(resamples),
                "evaluated_resamples": len(correctness),
                "resample_pass_rate": pass_rate,
                "pass_rate_delta": changed,
            }
            groups[
                (
                    str(row["dataset_name"]),
                    int(row["sentence_index"]),
                    row["original_is_correct"],
                )
            ].append(row)

    summary_rows = []
    for (dataset_name, sentence_index, original_is_correct), rows in sorted(groups.items()):
        pass_rates = [
            row["resample_pass_rate"]
            for row in rows
            if row["resample_pass_rate"] is not None
        ]
        deltas = [row["pass_rate_delta"] for row in rows if row["pass_rate_delta"] is not None]
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "sentence_index": sentence_index,
                "original_is_correct": original_is_correct,
                "interventions": len(rows),
                "mean_sentence_score": mean(
                    row["sentence_score"]
                    for row in rows
                    if row["sentence_score"] is not None
                )
                if any(row["sentence_score"] is not None for row in rows)
                else None,
                "mean_code_sentence_score": mean(
                    row["code_sentence_score"]
                    for row in rows
                    if row["code_sentence_score"] is not None
                )
                if any(row["code_sentence_score"] is not None for row in rows)
                else None,
                "mean_resample_pass_rate": mean(pass_rates) if pass_rates else None,
                "mean_pass_rate_delta": mean(deltas) if deltas else None,
            }
        )
    return summary_rows


def write_summary_csv(rows: list[dict], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "sentence_index",
        "original_is_correct",
        "interventions",
        "mean_sentence_score",
        "mean_code_sentence_score",
        "mean_resample_pass_rate",
        "mean_pass_rate_delta",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize black-box resampling JSONL into sentence-level CSV."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/blackbox_resampling_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_resampling_file(args.input)
    write_summary_csv(rows, args.output)
    print(f"Wrote {len(rows)} summary rows to {args.output}")


if __name__ == "__main__":
    main()
