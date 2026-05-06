"""Plot sentence-category position vs importance effects.

This script joins sentence labels with the available sentence-level importance
outputs and saves paper-style category-effect charts.

Usage:
    uv run python plot_category_effects.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "figures" / "category_effects"

DATASET_PREFIXES = {
    "openai_humaneval": "humaneval",
    "mbpp": "mbpp",
}

DATASET_LABELS = {
    "openai_humaneval": "HumanEval",
    "mbpp": "MBPP",
}

TAG_ORDER = [
    "problem_setup",
    "plan_generation",
    "fact_retrieval",
    "active_computation",
    "uncertainty_management",
    "self_checking",
    "result_consolidation",
    "final_answer_emission",
    "unknown",
]

TAG_LABELS = {
    "problem_setup": "Problem setup",
    "plan_generation": "Planning",
    "fact_retrieval": "Fact retrieval",
    "active_computation": "Computation",
    "uncertainty_management": "Uncertainty",
    "self_checking": "Self-checking",
    "result_consolidation": "Consolidation",
    "final_answer_emission": "Final answer",
    "unknown": "Unknown",
}

TAG_COLORS = {
    "problem_setup": "#e64b5d",
    "plan_generation": "#f3a53f",
    "fact_retrieval": "#ffd84d",
    "active_computation": "#45b76e",
    "uncertainty_management": "#35bfd1",
    "self_checking": "#4d7fe6",
    "result_consolidation": "#a455c7",
    "final_answer_emission": "#8f8f8f",
    "unknown": "#6f7782",
}


@dataclass(frozen=True)
class LabelInfo:
    tags_by_index: dict[int, str]
    num_sentences: int


@dataclass(frozen=True)
class Point:
    dataset_name: str
    tag: str
    position: float
    importance: float


@dataclass(frozen=True)
class MethodData:
    name: str
    title: str
    ylabel: str
    output_stem: str
    points: list[Point]
    notes: list[str]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_all_labels()
    methods = [
        build_receiver_method(
            labels,
            field="sentence_scores",
            name="receiver_reasoning",
            title="Receiver-Head Reasoning Importance",
            ylabel="Mean receiver-head R-score",
            output_stem="category_effect_receiver_reasoning",
        ),
        build_receiver_method(
            labels,
            field="code_sentence_scores",
            name="code_attention",
            title="Code-to-Reasoning Importance",
            ylabel="Mean final-code attention score",
            output_stem="category_effect_code_attention",
        ),
        build_blackbox_method(labels),
    ]

    written: list[Path] = []
    for method in methods:
        if not method.points:
            print(f"Skipping {method.name}: no joined points.")
            for note in method.notes:
                print(f"  - {note}")
            continue
        path = output_dir / f"{method.output_stem}.png"
        plot_method(method, path)
        written.append(path)
        print(f"Wrote {path} ({len(method.points)} sentence points)")
        for note in method.notes:
            print(f"  - {note}")

    causal_method = build_causal_masking_method(labels)
    if causal_method.points:
        path = output_dir / f"{causal_method.output_stem}.png"
        plot_method(causal_method, path)
        written.append(path)
        print(f"Wrote {path} ({len(causal_method.points)} sentence points)")
    elif causal_method.notes:
        print("Skipping causal_masking:")
        for note in causal_method.notes:
            print(f"  - {note}")

    if not written:
        raise SystemExit("No figures were generated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where figure PNGs will be written.",
    )
    return parser.parse_args()


def load_all_labels() -> dict[tuple[str, str, int], LabelInfo]:
    labels: dict[tuple[str, str, int], LabelInfo] = {}

    for path in sorted(RESULTS_DIR.glob("sentence_labels_*_qwen3_5_0_8b*.jsonl")):
        for row in read_jsonl(path):
            add_label_row(labels, row)

    for path in sorted(RESULTS_DIR.glob("labeled_*_qwen3_5_0_8b_full.jsonl")):
        for row in read_jsonl(path):
            add_label_row(labels, row)

    return labels


def add_label_row(
    labels: dict[tuple[str, str, int], LabelInfo],
    row: dict,
) -> None:
    dataset_name = str(row.get("dataset_name"))
    task_id = str(row.get("task_id"))
    sample_id = int(row.get("sample_id", 0))
    raw_labels = row.get("labels") or {}
    if not isinstance(raw_labels, dict) or not raw_labels:
        return

    tags_by_index: dict[int, str] = {}
    for raw_index, entry in raw_labels.items():
        if not isinstance(entry, dict):
            continue
        tags = entry.get("function_tags") or ["unknown"]
        tag = str(tags[0]) if tags else "unknown"
        tags_by_index[int(raw_index) - 1] = tag if tag in TAG_LABELS else "unknown"

    num_sentences = int(
        row.get("num_sentences")
        or len(row.get("sentences") or [])
        or (max(tags_by_index) + 1 if tags_by_index else 0)
    )
    if not tags_by_index or num_sentences == 0:
        return

    key = (dataset_name, task_id, sample_id)
    previous = labels.get(key)
    if previous is None or len(tags_by_index) >= len(previous.tags_by_index):
        labels[key] = LabelInfo(tags_by_index=tags_by_index, num_sentences=num_sentences)


def build_receiver_method(
    labels: dict[tuple[str, str, int], LabelInfo],
    *,
    field: str,
    name: str,
    title: str,
    ylabel: str,
    output_stem: str,
) -> MethodData:
    points: list[Point] = []
    notes: list[str] = []
    matched_rollouts = 0

    for dataset_name, prefix in DATASET_PREFIXES.items():
        path = RESULTS_DIR / f"receiver_head_summary_{prefix}_qwen3_5_0_8b.jsonl"
        if not path.exists():
            notes.append(f"Missing receiver summary for {DATASET_LABELS[dataset_name]}: {path}")
            continue
        for row in read_jsonl(path):
            key = (
                str(row.get("dataset_name", dataset_name)),
                str(row.get("task_id")),
                int(row.get("sample_id", 0)),
            )
            label_info = labels.get(key)
            if label_info is None:
                continue
            matched_rollouts += 1
            scores = row.get(field) or []
            points.extend(points_from_scores(key[0], label_info, scores))

    notes.append(f"Joined labels for {matched_rollouts} receiver-summary rollouts.")
    return MethodData(
        name=name,
        title=title,
        ylabel=ylabel,
        output_stem=output_stem,
        points=points,
        notes=notes,
    )


def build_blackbox_method(labels: dict[tuple[str, str, int], LabelInfo]) -> MethodData:
    points: list[Point] = []
    notes: list[str] = []
    paths = sorted(RESULTS_DIR.glob("blackbox_resampling_*.jsonl"))
    if not paths:
        notes.append("No black-box resampling JSONL files found.")
    matched_rows = 0

    for path in paths:
        for row in read_jsonl(path):
            key = (
                str(row.get("dataset_name")),
                str(row.get("task_id")),
                int(row.get("sample_id", 0)),
            )
            label_info = labels.get(key)
            if label_info is None:
                continue
            sentence_index = int(row.get("sentence_index", -1))
            if sentence_index not in label_info.tags_by_index:
                continue
            importance = blackbox_importance(row)
            if importance is None:
                continue
            matched_rows += 1
            points.append(
                Point(
                    dataset_name=key[0],
                    tag=label_info.tags_by_index[sentence_index],
                    position=normalized_position(sentence_index, label_info.num_sentences),
                    importance=importance,
                )
            )

    if paths:
        notes.append(
            f"Loaded {len(paths)} black-box file(s); joined {matched_rows} sentence interventions."
        )
        if any("smoke" in path.name for path in paths):
            notes.append("Available black-box data is a smoke run, so treat this chart as illustrative.")

    return MethodData(
        name="blackbox_counterfactual",
        title="Black-Box Counterfactual Importance",
        ylabel="Mean absolute correctness change",
        output_stem="category_effect_blackbox_counterfactual",
        points=points,
        notes=notes,
    )


def build_causal_masking_method(labels: dict[tuple[str, str, int], LabelInfo]) -> MethodData:
    points: list[Point] = []
    notes: list[str] = []
    matched_rollouts = 0

    for dataset_name, prefix in DATASET_PREFIXES.items():
        directory = RESULTS_DIR / f"causal_matrices_{prefix}_qwen3_5_0_8b"
        if not directory.exists():
            notes.append(f"Missing causal-mask matrix directory: {directory}")
            continue
        for path in sorted(directory.glob("*.npz")):
            with np.load(path, allow_pickle=True) as data:
                key = (
                    str(data["dataset_name"]),
                    str(data["task_id"]),
                    int(data["sample_id"]),
                )
                label_info = labels.get(key)
                if label_info is None:
                    continue
                matched_rollouts += 1
                matrix = data["causal_matrix"].astype(np.float32)
                influence = influence_exerted(matrix)
                points.extend(points_from_scores(key[0], label_info, influence))

    if matched_rollouts:
        notes.append(f"Joined labels for {matched_rollouts} causal-mask rollouts.")

    return MethodData(
        name="causal_masking",
        title="White-Box Causal Masking Importance",
        ylabel="Mean causal influence exerted",
        output_stem="category_effect_causal_masking",
        points=points,
        notes=notes,
    )


def points_from_scores(
    dataset_name: str,
    label_info: LabelInfo,
    scores: Iterable[float | None],
) -> list[Point]:
    points: list[Point] = []
    for sentence_index, score in enumerate(scores):
        value = clean_float(score)
        if value is None or sentence_index not in label_info.tags_by_index:
            continue
        points.append(
            Point(
                dataset_name=dataset_name,
                tag=label_info.tags_by_index[sentence_index],
                position=normalized_position(sentence_index, label_info.num_sentences),
                importance=value,
            )
        )
    return points


def blackbox_importance(row: dict) -> float | None:
    resamples = row.get("resamples") or []
    correctness = [
        bool(resample.get("is_correct"))
        for resample in resamples
        if resample.get("is_correct") is not None
    ]
    if not correctness or row.get("original_is_correct") is None:
        return None
    pass_rate = float(np.mean(correctness))
    original = float(bool(row.get("original_is_correct")))
    return abs(pass_rate - original)


def influence_exerted(matrix: np.ndarray) -> np.ndarray:
    influence = np.full(matrix.shape[0], np.nan, dtype=np.float32)
    for index in range(matrix.shape[0] - 1):
        values = matrix[index, index + 1 :]
        if np.any(np.isfinite(values)):
            influence[index] = float(np.nanmean(values))
    return influence


def plot_method(method: MethodData, output_path: Path) -> None:
    datasets = [dataset for dataset in DATASET_PREFIXES if any(p.dataset_name == dataset for p in method.points)]
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.4 * len(datasets), 5.2),
        squeeze=False,
        sharey=True,
    )
    fig.suptitle(method.title, fontsize=17, y=1.02)

    for ax, dataset_name in zip(axes[0], datasets, strict=True):
        dataset_points = [point for point in method.points if point.dataset_name == dataset_name]
        plot_dataset(ax, dataset_points, DATASET_LABELS.get(dataset_name, dataset_name), method.ylabel)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dataset(ax: plt.Axes, points: list[Point], title: str, ylabel: str) -> None:
    grouped: dict[str, list[Point]] = defaultdict(list)
    for point in points:
        grouped[point.tag].append(point)

    plotted = False
    for tag in TAG_ORDER:
        tag_points = grouped.get(tag, [])
        if not tag_points:
            continue
        positions = np.array([point.position for point in tag_points], dtype=np.float64)
        importances = np.array([point.importance for point in tag_points], dtype=np.float64)
        x_mean, x_sem = mean_and_sem(positions)
        y_mean, y_sem = mean_and_sem(importances)
        ax.errorbar(
            x_mean,
            y_mean,
            xerr=x_sem,
            yerr=y_sem,
            fmt="o",
            markersize=8,
            capsize=4,
            elinewidth=1.8,
            color=TAG_COLORS[tag],
            label=f"{TAG_LABELS[tag]} (n={len(tag_points)})",
            alpha=0.9,
        )
        plotted = True

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Normalized position in trace (0-1)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(-0.03, 1.03)
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if plotted:
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.text(0.5, 0.5, "No joined data", ha="center", va="center", transform=ax.transAxes)


def mean_and_sem(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return math.nan, math.nan
    if len(finite) == 1:
        return float(finite[0]), 0.0
    return float(np.mean(finite)), float(np.std(finite, ddof=1) / np.sqrt(len(finite)))


def normalized_position(index: int, num_sentences: int) -> float:
    return index / max(num_sentences - 1, 1)


def clean_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


if __name__ == "__main__":
    main()
