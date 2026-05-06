"""Visualize how much data exists at each pipeline stage.

Usage:
    uv run python plot_data_coverage.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
ROLLOUT_DIR = PROJECT_ROOT / "assets" / "rollouts"
OUTPUT_DIR = RESULTS_DIR / "figures" / "coverage"

DATASETS = {
    "openai_humaneval": {
        "label": "HumanEval",
        "prefix": "humaneval",
        "rollout": "humaneval_qwen3_5_0_8b_full.jsonl",
    },
    "mbpp": {
        "label": "MBPP",
        "prefix": "mbpp",
        "rollout": "mbpp_qwen3_5_0_8b_full.jsonl",
    },
}


@dataclass
class CoverageRow:
    dataset_name: str
    dataset_label: str
    generated_rollouts: int = 0
    complete_rollouts: int = 0
    correct_rollouts: int = 0
    labeled_rollouts: int = 0
    labeled_sentences: int = 0
    receiver_rollouts: int = 0
    receiver_sentence_scores: int = 0
    code_sentence_scores: int = 0
    causal_matrix_rollouts: int = 0
    causal_matrix_sentences: int = 0
    blackbox_interventions: int = 0
    blackbox_resamples: int = 0


def main() -> None:
    args = parse_args()
    rows = collect_coverage()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "data_collection_coverage.json"
    json_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2),
        encoding="utf-8",
    )

    figure_path = args.output_dir / "data_collection_coverage.png"
    plot_coverage(rows, figure_path)
    print(f"Wrote {figure_path}")
    print(f"Wrote {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the coverage figure and JSON summary will be written.",
    )
    return parser.parse_args()


def collect_coverage() -> list[CoverageRow]:
    rows = [
        CoverageRow(dataset_name=name, dataset_label=meta["label"])
        for name, meta in DATASETS.items()
    ]
    by_dataset = {row.dataset_name: row for row in rows}

    collect_rollout_counts(by_dataset)
    collect_label_counts(by_dataset)
    collect_receiver_counts(by_dataset)
    collect_causal_counts(by_dataset)
    collect_blackbox_counts(by_dataset)
    return rows


def collect_rollout_counts(by_dataset: dict[str, CoverageRow]) -> None:
    for dataset_name, meta in DATASETS.items():
        row = by_dataset[dataset_name]
        path = ROLLOUT_DIR / meta["rollout"]
        if not path.exists():
            continue
        for payload in read_jsonl(path):
            row.generated_rollouts += 1
            row.complete_rollouts += int(payload.get("complete") is True)
            row.correct_rollouts += int(payload.get("is_correct") is True)


def collect_label_counts(by_dataset: dict[str, CoverageRow]) -> None:
    seen_rollouts: dict[str, set[tuple[str, int]]] = defaultdict(set)
    seen_sentences: dict[str, set[tuple[str, int, int]]] = defaultdict(set)

    label_paths = [
        *sorted(RESULTS_DIR.glob("sentence_labels_*_qwen3_5_0_8b*.jsonl")),
        *sorted(RESULTS_DIR.glob("labeled_*_qwen3_5_0_8b_full.jsonl")),
    ]
    for path in label_paths:
        for payload in read_jsonl(path):
            dataset_name = str(payload.get("dataset_name"))
            if dataset_name not in by_dataset:
                continue
            task_id = str(payload.get("task_id"))
            sample_id = int(payload.get("sample_id", 0))
            labels = payload.get("labels") or {}
            if not isinstance(labels, dict) or not labels:
                continue
            seen_rollouts[dataset_name].add((task_id, sample_id))
            for raw_index, entry in labels.items():
                if isinstance(entry, dict) and entry.get("function_tags"):
                    seen_sentences[dataset_name].add((task_id, sample_id, int(raw_index)))

    for dataset_name, row in by_dataset.items():
        row.labeled_rollouts = len(seen_rollouts[dataset_name])
        row.labeled_sentences = len(seen_sentences[dataset_name])


def collect_receiver_counts(by_dataset: dict[str, CoverageRow]) -> None:
    for dataset_name, meta in DATASETS.items():
        row = by_dataset[dataset_name]
        path = RESULTS_DIR / f"receiver_head_summary_{meta['prefix']}_qwen3_5_0_8b.jsonl"
        if not path.exists():
            continue
        for payload in read_jsonl(path):
            row.receiver_rollouts += 1
            row.receiver_sentence_scores += count_finite(payload.get("sentence_scores") or [])
            row.code_sentence_scores += count_finite(payload.get("code_sentence_scores") or [])


def collect_causal_counts(by_dataset: dict[str, CoverageRow]) -> None:
    for dataset_name, meta in DATASETS.items():
        row = by_dataset[dataset_name]
        directory = RESULTS_DIR / f"causal_matrices_{meta['prefix']}_qwen3_5_0_8b"
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.npz")):
            row.causal_matrix_rollouts += 1
            with np.load(path, allow_pickle=True) as data:
                row.causal_matrix_sentences += int(data.get("num_sentences", 0))


def collect_blackbox_counts(by_dataset: dict[str, CoverageRow]) -> None:
    seen_interventions: dict[str, set[tuple[str, int, int, str]]] = defaultdict(set)
    resample_counts: dict[str, int] = defaultdict(int)

    for path in sorted(RESULTS_DIR.glob("blackbox_resampling_*.jsonl")):
        for payload in read_jsonl(path):
            dataset_name = str(payload.get("dataset_name"))
            if dataset_name not in by_dataset:
                continue
            key = (
                str(payload.get("task_id")),
                int(payload.get("sample_id", 0)),
                int(payload.get("sentence_index", -1)),
                str(payload.get("selection", "sentence")),
            )
            seen_interventions[dataset_name].add(key)
            resample_counts[dataset_name] += len(payload.get("resamples") or [])

    for dataset_name, row in by_dataset.items():
        row.blackbox_interventions = len(seen_interventions[dataset_name])
        row.blackbox_resamples = resample_counts[dataset_name]


def plot_coverage(rows: list[CoverageRow], output_path: Path) -> None:
    dataset_labels = [row.dataset_label for row in rows]
    colors = ["#4c78a8", "#f58518"]

    rollout_metrics = [
        ("Generated rollouts", "generated_rollouts"),
        ("Complete rollouts", "complete_rollouts"),
        ("Correct rollouts", "correct_rollouts"),
        ("Labeled rollouts", "labeled_rollouts"),
        ("Receiver summaries", "receiver_rollouts"),
        ("Causal matrices", "causal_matrix_rollouts"),
    ]
    sentence_metrics = [
        ("Labeled sentences", "labeled_sentences"),
        ("R-score values", "receiver_sentence_scores"),
        ("C-score values", "code_sentence_scores"),
        ("Causal-mask sentences", "causal_matrix_sentences"),
        ("Black-box interventions", "blackbox_interventions"),
        ("Black-box resamples", "blackbox_resamples"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Data Collected Across Thought-Anchor Pipeline", fontsize=17)
    plot_grouped_bars(axes[0], rows, rollout_metrics, dataset_labels, colors, "Rollout-level coverage")
    plot_grouped_bars(axes[1], rows, sentence_metrics, dataset_labels, colors, "Sentence/intervention-level coverage")

    fig.text(
        0.5,
        0.01,
        "Counts are unique by dataset/task/sample where possible; black-box totals include all available resampling files.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars(
    ax: plt.Axes,
    rows: list[CoverageRow],
    metrics: list[tuple[str, str]],
    dataset_labels: list[str],
    colors: list[str],
    title: str,
) -> None:
    y = np.arange(len(metrics))
    height = 0.36
    offsets = np.linspace(-height / 2, height / 2, len(rows))

    for row, label, color, offset in zip(rows, dataset_labels, colors, offsets, strict=True):
        values = [int(getattr(row, field)) for _, field in metrics]
        ax.barh(y + offset, values, height=height, color=color, alpha=0.9, label=label)
        for index, value in enumerate(values):
            ax.text(
                value + max(values) * 0.015 if max(values) else 0.2,
                y[index] + offset,
                f"{value:,}",
                va="center",
                fontsize=8,
            )

    ax.set_title(title, fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _ in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right")


def count_finite(values: list[object]) -> int:
    count = 0
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        count += int(np.isfinite(number))
    return count


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
