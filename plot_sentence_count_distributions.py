"""Plot sentence-count distributions across pipeline artifacts.

Usage:
    uv run python plot_sentence_count_distributions.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thought_anchors_code.analysis.whitebox_attention.trace_utils import split_reasoning_steps


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

STAGE_LABELS = {
    "generated": "Generated",
    "complete": "Complete",
    "correct": "Correct",
    "labeled": "Labeled",
    "receiver_r": "R-score\nanalyzed",
    "causal": "Causal\nmatrix",
    "blackbox": "Black-box\ninterventions",
}


@dataclass
class StageSummary:
    dataset_name: str
    dataset_label: str
    stage: str
    stage_label: str
    count: int
    mean: float | None
    median: float | None
    p25: float | None
    p75: float | None
    min: float | None
    max: float | None


def main() -> None:
    args = parse_args()
    values = collect_sentence_count_values()
    summaries = summarize(values)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = args.output_dir / "sentence_count_distributions.png"
    json_path = args.output_dir / "sentence_count_distributions.json"
    plot_distributions(values, figure_path)
    json_path.write_text(json.dumps([asdict(row) for row in summaries], indent=2), encoding="utf-8")

    print(f"Wrote {figure_path}")
    print(f"Wrote {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the sentence-count figure and JSON summary will be written.",
    )
    return parser.parse_args()


def collect_sentence_count_values() -> dict[str, dict[str, list[int]]]:
    values: dict[str, dict[str, list[int]]] = {
        dataset_name: {stage: [] for stage in STAGE_LABELS}
        for dataset_name in DATASETS
    }
    collect_rollout_values(values)
    collect_label_values(values)
    collect_receiver_values(values)
    collect_causal_values(values)
    collect_blackbox_values(values)
    return values


def collect_rollout_values(values: dict[str, dict[str, list[int]]]) -> None:
    for dataset_name, meta in DATASETS.items():
        path = ROLLOUT_DIR / meta["rollout"]
        if not path.exists():
            continue
        for payload in read_jsonl(path):
            sentence_count = len(split_reasoning_steps(str(payload.get("reasoning") or "")))
            values[dataset_name]["generated"].append(sentence_count)
            if payload.get("complete") is True:
                values[dataset_name]["complete"].append(sentence_count)
            if payload.get("is_correct") is True:
                values[dataset_name]["correct"].append(sentence_count)


def collect_label_values(values: dict[str, dict[str, list[int]]]) -> None:
    labeled_counts: dict[str, dict[tuple[str, int], int]] = defaultdict(dict)
    paths = [
        *sorted(RESULTS_DIR.glob("sentence_labels_*_qwen3_5_0_8b*.jsonl")),
        *sorted(RESULTS_DIR.glob("labeled_*_qwen3_5_0_8b_full.jsonl")),
    ]
    for path in paths:
        for payload in read_jsonl(path):
            dataset_name = str(payload.get("dataset_name"))
            if dataset_name not in values:
                continue
            labels = payload.get("labels") or {}
            if not isinstance(labels, dict) or not labels:
                continue
            key = (str(payload.get("task_id")), int(payload.get("sample_id", 0)))
            count = int(payload.get("num_sentences") or len(payload.get("sentences") or []) or len(labels))
            labeled_counts[dataset_name][key] = max(count, labeled_counts[dataset_name].get(key, 0))

    for dataset_name, by_rollout in labeled_counts.items():
        values[dataset_name]["labeled"].extend(by_rollout.values())


def collect_receiver_values(values: dict[str, dict[str, list[int]]]) -> None:
    for dataset_name, meta in DATASETS.items():
        path = RESULTS_DIR / f"receiver_head_summary_{meta['prefix']}_qwen3_5_0_8b.jsonl"
        if not path.exists():
            continue
        for payload in read_jsonl(path):
            values[dataset_name]["receiver_r"].append(count_finite(payload.get("sentence_scores") or []))


def collect_causal_values(values: dict[str, dict[str, list[int]]]) -> None:
    for dataset_name, meta in DATASETS.items():
        directory = RESULTS_DIR / f"causal_matrices_{meta['prefix']}_qwen3_5_0_8b"
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.npz")):
            with np.load(path, allow_pickle=True) as data:
                values[dataset_name]["causal"].append(int(data.get("num_sentences", 0)))


def collect_blackbox_values(values: dict[str, dict[str, list[int]]]) -> None:
    interventions_by_rollout: dict[str, dict[tuple[str, int], int]] = defaultdict(lambda: defaultdict(int))
    for path in sorted(RESULTS_DIR.glob("blackbox_resampling_*.jsonl")):
        for payload in read_jsonl(path):
            dataset_name = str(payload.get("dataset_name"))
            if dataset_name not in values:
                continue
            key = (str(payload.get("task_id")), int(payload.get("sample_id", 0)))
            interventions_by_rollout[dataset_name][key] += 1

    for dataset_name, by_rollout in interventions_by_rollout.items():
        values[dataset_name]["blackbox"].extend(by_rollout.values())


def summarize(values: dict[str, dict[str, list[int]]]) -> list[StageSummary]:
    rows: list[StageSummary] = []
    for dataset_name, stages in values.items():
        for stage, stage_values in stages.items():
            arr = np.array(stage_values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            rows.append(
                StageSummary(
                    dataset_name=dataset_name,
                    dataset_label=DATASETS[dataset_name]["label"],
                    stage=stage,
                    stage_label=STAGE_LABELS[stage].replace("\n", " "),
                    count=len(finite),
                    mean=clean_stat(np.mean(finite)) if len(finite) else None,
                    median=clean_stat(np.median(finite)) if len(finite) else None,
                    p25=clean_stat(np.percentile(finite, 25)) if len(finite) else None,
                    p75=clean_stat(np.percentile(finite, 75)) if len(finite) else None,
                    min=clean_stat(np.min(finite)) if len(finite) else None,
                    max=clean_stat(np.max(finite)) if len(finite) else None,
                )
            )
    return rows


def plot_distributions(values: dict[str, dict[str, list[int]]], output_path: Path) -> None:
    stages = list(STAGE_LABELS)
    x = np.arange(len(stages), dtype=np.float64)
    offsets = {"openai_humaneval": -0.18, "mbpp": 0.18}
    colors = {"openai_humaneval": "#4c78a8", "mbpp": "#f58518"}

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle("Sentence Count Distributions Across Collected Artifacts", fontsize=17)

    ax = axes[0]
    legend_handles = []
    for dataset_name, meta in DATASETS.items():
        positions = x + offsets[dataset_name]
        data = [values[dataset_name][stage] for stage in stages]
        non_empty_positions = [pos for pos, vals in zip(positions, data, strict=True) if vals]
        non_empty_data = [vals for vals in data if vals]
        if not non_empty_data:
            continue
        box = ax.boxplot(
            non_empty_data,
            positions=non_empty_positions,
            widths=0.28,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": colors[dataset_name],
                "markersize": 5,
            },
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": colors[dataset_name], "linewidth": 1.2},
            capprops={"color": colors[dataset_name], "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markerfacecolor": colors[dataset_name],
                "markeredgecolor": colors[dataset_name],
                "markersize": 3,
                "alpha": 0.28,
            },
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[dataset_name])
            patch.set_alpha(0.55)
            patch.set_edgecolor(colors[dataset_name])
        legend_handles.append(box["boxes"][0])

    ax.set_ylabel("Sentences per rollout / artifact")
    ax.set_yscale("symlog", linthresh=20)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(legend_handles, [meta["label"] for meta in DATASETS.values()], loc="upper right")
    ax.text(
        0.01,
        0.96,
        "Boxes show IQR; whiskers use Matplotlib's 1.5x IQR rule; diamonds mark means.",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        color="#555555",
    )

    count_ax = axes[1]
    for dataset_name, meta in DATASETS.items():
        counts = [len(values[dataset_name][stage]) for stage in stages]
        count_ax.bar(
            x + offsets[dataset_name],
            counts,
            width=0.28,
            color=colors[dataset_name],
            alpha=0.75,
            label=meta["label"],
        )
        for pos, count in zip(x + offsets[dataset_name], counts, strict=True):
            count_ax.text(pos, count + max(counts or [1]) * 0.02, f"{count:,}", ha="center", fontsize=8)

    count_ax.set_ylabel("Artifacts")
    count_ax.set_xticks(x)
    count_ax.set_xticklabels([STAGE_LABELS[stage] for stage in stages])
    count_ax.grid(True, axis="y", color="#dddddd", linewidth=0.8, alpha=0.7)
    count_ax.spines["top"].set_visible(False)
    count_ax.spines["right"].set_visible(False)
    count_ax.set_xlabel("Pipeline stage")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def count_finite(values: list[object]) -> int:
    count = 0
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        count += int(math.isfinite(number))
    return count


def clean_stat(value: float) -> float:
    return round(float(value), 3)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
