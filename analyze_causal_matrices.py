"""Aggregate analysis of causal matrices vs receiver-head R-scores + sentence labels.

Figures saved to results/figures/:
  causal_correlation_{dataset}.png   — scatter R-score vs influence + anchor position histogram
  causal_heatmaps_{dataset}.png      — representative heatmaps, tag-coloured axes
  causal_position_{dataset}.png      — scatter coloured by sentence position
  causal_by_tag_{dataset}.png        — mean influence exerted per function tag (bar)
  causal_tag_pair_{dataset}.png      — mean causal influence averaged by (source tag, target tag)

Usage:
    uv run python analyze_causal_matrices.py --dataset humaneval
    uv run python analyze_causal_matrices.py --dataset humaneval mbpp
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

_DATASET_FILE_PREFIX = {
    "openai_humaneval": "humaneval",
    "mbpp": "mbpp",
}

DATASET_ALIASES = {
    "humaneval": "openai_humaneval",
    "human_eval": "openai_humaneval",
    "openai_humaneval": "openai_humaneval",
    "mbpp": "mbpp",
}

ALL_TAGS = [
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

TAG_COLORS = {
    "problem_setup":          "#e6194b",
    "plan_generation":        "#f58231",
    "fact_retrieval":         "#ffe119",
    "active_computation":     "#3cb44b",
    "uncertainty_management": "#42d4f4",
    "self_checking":          "#4363d8",
    "result_consolidation":   "#911eb4",
    "final_answer_emission":  "#a9a9a9",
    "unknown":                "#808080",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _causal_dir(canonical: str) -> Path:
    prefix = _DATASET_FILE_PREFIX.get(canonical, canonical)
    return RESULTS_DIR / f"causal_matrices_{prefix}_qwen3_5_0_8b"


def _rh_file(canonical: str) -> Path:
    prefix = _DATASET_FILE_PREFIX.get(canonical, canonical)
    return RESULTS_DIR / f"receiver_head_summary_{prefix}_qwen3_5_0_8b.jsonl"


def _labels_file(canonical: str) -> Path:
    prefix = _DATASET_FILE_PREFIX.get(canonical, canonical)
    return RESULTS_DIR / f"sentence_labels_{prefix}_qwen3_5_0_8b.jsonl"


def load_index(path: Path, key_fields: tuple, value_field: str) -> dict:
    index = {}
    if not path.exists():
        return index
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            key = tuple(row[f] if f != "sample_id" else int(row[f]) for f in key_fields)
            index[key] = row[value_field]
    return index


def _dominant_tag(entry: dict) -> str:
    tags = entry.get("function_tags", [])
    return tags[0] if tags else "unknown"


def load_joined_data(canonical: str) -> list[dict]:
    causal_dir = _causal_dir(canonical)
    rh_path = _rh_file(canonical)
    labels_path = _labels_file(canonical)

    if not causal_dir.exists():
        raise FileNotFoundError(f"Causal matrix dir not found: {causal_dir}")
    if not rh_path.exists():
        raise FileNotFoundError(f"Receiver-head summary not found: {rh_path}")

    rh_index = load_index(rh_path, ("task_id", "sample_id"), "sentence_scores")

    # Labels: {(task_id, sample_id) -> {str_idx: {function_tags, depends_on}}}
    labels_raw = load_index(labels_path, ("task_id", "sample_id"), "labels") if labels_path.exists() else {}

    rows = []
    for npz_path in sorted(causal_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        task_id = str(data["task_id"])
        sample_id = int(data["sample_id"])
        matrix = data["causal_matrix"].astype(np.float32)
        M = int(data["num_sentences"])

        key = (task_id, sample_id)
        if key not in rh_index:
            continue
        rh_scores = np.array(rh_index[key], dtype=np.float32)
        if len(rh_scores) != M:
            continue

        # Tags: list of dominant tag per sentence (0-indexed), or None if no labels
        tags: list[str] | None = None
        if key in labels_raw:
            label_dict = labels_raw[key]
            if len(label_dict) == M:
                tags = [_dominant_tag(label_dict[str(i + 1)]) for i in range(M)]

        rows.append({
            "task_id": task_id,
            "sample_id": sample_id,
            "matrix": matrix,
            "rh_scores": rh_scores,
            "tags": tags,
            "M": M,
        })

    return rows


# ── Per-sentence metrics ───────────────────────────────────────────────────────

def influence_exerted(matrix: np.ndarray) -> np.ndarray:
    M = matrix.shape[0]
    out = np.full(M, np.nan)
    for i in range(M - 1):
        vals = matrix[i, i + 1:]
        if not np.all(np.isnan(vals)):
            out[i] = np.nanmean(vals)
    return out


def influence_received(matrix: np.ndarray) -> np.ndarray:
    M = matrix.shape[0]
    out = np.full(M, np.nan)
    for j in range(1, M):
        vals = matrix[:j, j]
        if not np.all(np.isnan(vals)):
            out[j] = np.nanmean(vals)
    return out


# ── Analysis ───────────────────────────────────────────────────────────────────

def run_analysis(rows: list[dict], dataset_label: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_rh: list[float] = []
    all_exerted: list[float] = []
    peak_agreements: list[bool] = []
    norm_positions: list[float] = []

    # Tag-level accumulators
    tag_influence: defaultdict[str, list[float]] = defaultdict(list)
    tag_rh: defaultdict[str, list[float]] = defaultdict(list)
    # Tag-pair: (source_tag, target_tag) -> list of causal_matrix[i,j] values
    tag_pair_values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)

    for row in rows:
        matrix = row["matrix"]
        rh = row["rh_scores"]
        tags = row["tags"]
        M = row["M"]

        exerted = influence_exerted(matrix)

        for i in range(M):
            if np.isfinite(rh[i]) and np.isfinite(exerted[i]):
                all_rh.append(float(rh[i]))
                all_exerted.append(float(exerted[i]))
                if tags:
                    tag_influence[tags[i]].append(float(exerted[i]))
                    tag_rh[tags[i]].append(float(rh[i]))

        # Tag-pair matrix
        if tags:
            for i in range(M):
                for j in range(i + 1, M):
                    val = matrix[i, j]
                    if np.isfinite(val):
                        tag_pair_values[(tags[i], tags[j])].append(float(val))

        valid = np.isfinite(exerted)
        if valid.sum() >= 2:
            peak_rh = int(np.argmax(rh))
            peak_inf = int(np.nanargmax(np.where(valid, exerted, np.nan)))
            peak_agreements.append(peak_rh == peak_inf)
        if valid.sum() >= 1:
            peak_inf = int(np.nanargmax(np.where(valid, exerted, np.nan)))
            norm_positions.append(peak_inf / max(M - 1, 1))

    rh_arr = np.array(all_rh)
    ext_arr = np.array(all_exerted)

    # ── Figure 1: Correlation scatter + anchor position ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Causal Influence vs Receiver-Head R-Score — {dataset_label}", fontsize=13)

    ax = axes[0]
    ax.scatter(rh_arr, ext_arr, alpha=0.35, s=18, color="steelblue")
    r, p = stats.pearsonr(rh_arr, ext_arr)
    m_fit, b_fit = np.polyfit(rh_arr, ext_arr, 1)
    x_line = np.linspace(rh_arr.min(), rh_arr.max(), 100)
    ax.plot(x_line, m_fit * x_line + b_fit, color="tomato", linewidth=1.5)
    ax.set_xlabel("Receiver-head R-score")
    ax.set_ylabel("Mean causal influence exerted")
    ax.set_title(f"r = {r:.3f}  (p = {p:.2e},  n = {len(rh_arr)})")

    ax2 = axes[1]
    ax2.hist(norm_positions, bins=10, range=(0, 1), color="steelblue", alpha=0.75, edgecolor="white")
    ax2.axvline(np.mean(norm_positions), color="tomato", linestyle="--",
                label=f"mean = {np.mean(norm_positions):.2f}")
    ax2.set_xlabel("Relative position of most-influential sentence (0=first, 1=last)")
    ax2.set_ylabel("Count (rollouts)")
    ax2.set_title("Where do thought anchors appear?")
    ax2.legend()

    plt.tight_layout()
    _save(FIGURES_DIR / f"causal_correlation_{dataset_label}.png")

    # ── Figure 2: Representative heatmaps, tag-coloured axes ──────────────────
    labeled_rows = [r for r in rows if r["tags"] is not None]
    sample_rows = sorted(labeled_rows or rows, key=lambda r: r["M"], reverse=True)[:4]
    fig, axes = plt.subplots(1, len(sample_rows), figsize=(5 * len(sample_rows), 5))
    if len(sample_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, sample_rows):
        matrix = row["matrix"]
        tags = row["tags"]
        M = row["M"]
        vmax = np.nanpercentile(np.abs(matrix), 95)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{row['task_id']} s{row['sample_id']}\nM={M}", fontsize=9)
        ax.set_xlabel("Target sentence j")
        ax.set_ylabel("Source sentence i")
        plt.colorbar(im, ax=ax, shrink=0.7)

        if tags:
            ticks = list(range(M))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(ticks, fontsize=6)
            ax.set_yticklabels(ticks, fontsize=6)
            for tick_label, tag in zip(ax.get_xticklabels(), tags):
                tick_label.set_color(TAG_COLORS.get(tag, "#808080"))
            for tick_label, tag in zip(ax.get_yticklabels(), tags):
                tick_label.set_color(TAG_COLORS.get(tag, "#808080"))

    # Tag colour legend
    handles = [mpatches.Patch(color=c, label=t) for t, c in TAG_COLORS.items()
               if t not in ("unknown",)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.08))
    plt.suptitle(f"Causal Matrices (largest M, tick colours = tag) — {dataset_label}", fontsize=11)
    plt.tight_layout()
    _save(FIGURES_DIR / f"causal_heatmaps_{dataset_label}.png")

    # ── Figure 3: Scatter coloured by sentence position ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    pos_list, rh2_list, ext2_list = [], [], []
    for row in rows:
        M = row["M"]
        exerted = influence_exerted(row["matrix"])
        for i in range(M):
            if np.isfinite(row["rh_scores"][i]) and np.isfinite(exerted[i]):
                rh2_list.append(float(row["rh_scores"][i]))
                ext2_list.append(float(exerted[i]))
                pos_list.append(i / max(M - 1, 1))

    sc = ax.scatter(rh2_list, ext2_list, c=pos_list, cmap="plasma", alpha=0.4, s=18)
    plt.colorbar(sc, ax=ax, label="Relative sentence position")
    ax.set_xlabel("Receiver-head R-score")
    ax.set_ylabel("Mean causal influence exerted")
    ax.set_title(f"Sentence position in trace — {dataset_label}")
    plt.tight_layout()
    _save(FIGURES_DIR / f"causal_position_{dataset_label}.png")

    # ── Figure 4: Mean influence exerted per tag (bar chart) ──────────────────
    if tag_influence:
        present_tags = [t for t in ALL_TAGS if t in tag_influence]
        means = [np.mean(tag_influence[t]) for t in present_tags]
        sems = [np.std(tag_influence[t]) / np.sqrt(len(tag_influence[t])) for t in present_tags]
        colors = [TAG_COLORS.get(t, "#808080") for t in present_tags]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Causal Influence by Function Tag — {dataset_label}", fontsize=13)

        ax = axes[0]
        y_pos = np.arange(len(present_tags))
        bars = ax.barh(y_pos, means, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(present_tags)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean causal influence exerted (col-normalised log-KL)")
        ax.set_title("Which tags act as thought anchors?")
        counts = [len(tag_influence[t]) for t in present_tags]
        for i, (mean, count) in enumerate(zip(means, counts)):
            ax.text(mean + 0.02 if mean >= 0 else mean - 0.02, i,
                    f"n={count}", va="center", ha="left" if mean >= 0 else "right", fontsize=8)

        # Right panel: mean R-score per tag
        ax2 = axes[1]
        rh_means = [np.mean(tag_rh[t]) for t in present_tags]
        rh_sems = [np.std(tag_rh[t]) / np.sqrt(len(tag_rh[t])) for t in present_tags]
        ax2.barh(y_pos, rh_means, color=colors, alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(present_tags)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Mean receiver-head R-score")
        ax2.set_title("Which tags attract receiver-head attention?")

        plt.tight_layout()
        _save(FIGURES_DIR / f"causal_by_tag_{dataset_label}.png")

    # ── Figure 5: Tag-pair mean causal influence heatmap ──────────────────────
    if tag_pair_values:
        present_tags = [t for t in ALL_TAGS if any(
            t == src or t == tgt for src, tgt in tag_pair_values
        )]
        n = len(present_tags)
        tag_idx = {t: i for i, t in enumerate(present_tags)}
        pair_matrix = np.full((n, n), np.nan)

        for (src, tgt), vals in tag_pair_values.items():
            if src in tag_idx and tgt in tag_idx:
                pair_matrix[tag_idx[src], tag_idx[tgt]] = np.mean(vals)

        fig, ax = plt.subplots(figsize=(9, 7))
        vmax = np.nanpercentile(np.abs(pair_matrix), 95)
        im = ax.imshow(pair_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(present_tags, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(present_tags, fontsize=9)
        ax.set_xlabel("Target tag (j)")
        ax.set_ylabel("Source tag (i)")
        ax.set_title(f"Mean causal influence by tag pair — {dataset_label}\n"
                     f"(red = source tag causally drives target tag)")
        plt.colorbar(im, ax=ax, label="Mean log-KL (col-normalised)")

        # Annotate cells with mean value
        for i in range(n):
            for j in range(n):
                val = pair_matrix[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black")

        plt.tight_layout()
        _save(FIGURES_DIR / f"causal_tag_pair_{dataset_label}.png")

    # ── Print summary ──────────────────────────────────────────────────────────
    agree_pct = 100 * np.mean(peak_agreements) if peak_agreements else float("nan")
    labeled_count = sum(1 for r in rows if r["tags"] is not None)
    print(f"\n{'─'*55}")
    print(f"Dataset:           {dataset_label}")
    print(f"Rollouts joined:   {len(rows)}  ({labeled_count} with labels)")
    print(f"Sentence pairs:    {len(rh_arr)}")
    print(f"Pearson r:         {r:.3f}  (p={p:.2e})")
    print(f"Peak agreement:    {agree_pct:.1f}%  ({sum(peak_agreements)}/{len(peak_agreements)} rollouts)")
    print(f"Mean anchor pos:   {np.mean(norm_positions):.2f}  (0=start, 1=end)")
    if tag_influence:
        best_tag = max(tag_influence, key=lambda t: np.mean(tag_influence[t]))
        print(f"Highest-influence tag: {best_tag} ({np.mean(tag_influence[best_tag]):.3f})")
    print(f"{'─'*55}\n")


def _save(path: Path) -> None:
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["humaneval"],
                        help="humaneval, mbpp, or both")
    args = parser.parse_args()

    for ds in args.dataset:
        canonical = DATASET_ALIASES.get(ds.lower())
        if canonical is None:
            print(f"Unknown dataset: {ds}")
            continue
        label = _DATASET_FILE_PREFIX.get(canonical, canonical)
        print(f"\nLoading {label} …")
        rows = load_joined_data(canonical)
        print(f"Joined {len(rows)} rollouts.")
        if not rows:
            print("No data — skipping.")
            continue
        run_analysis(rows, label)


if __name__ == "__main__":
    main()
