"""Aggregate analysis of causal matrices vs receiver-head R-scores.

For each rollout we join the M×M causal matrix with per-sentence receiver-head
R-scores and compute:

  1. Sentence-level correlation: does the sentence that exerts the most causal
     influence also have the highest R-score?
  2. Peak-sentence agreement: does argmax(R-score) == argmax(influence exerted)?
  3. Normalised-position distribution: where in the trace do the most influential
     sentences tend to appear?
  4. Representative heatmaps for a handful of rollouts.

Figures are saved to results/figures/.

Usage:
    uv run python analyze_causal_matrices.py --dataset humaneval
    uv run python analyze_causal_matrices.py --dataset mbpp
    uv run python analyze_causal_matrices.py --dataset humaneval --dataset mbpp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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


# ── Data loading ───────────────────────────────────────────────────────────────

def _causal_dir(canonical: str) -> Path:
    prefix = _DATASET_FILE_PREFIX.get(canonical, canonical)
    return RESULTS_DIR / f"causal_matrices_{prefix}_qwen3_5_0_8b"


def _rh_file(canonical: str) -> Path:
    prefix = _DATASET_FILE_PREFIX.get(canonical, canonical)
    return RESULTS_DIR / f"receiver_head_summary_{prefix}_qwen3_5_0_8b.jsonl"


def load_receiver_head_index(path: Path) -> dict[tuple[str, int], list[float]]:
    index: dict[tuple[str, int], list[float]] = {}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            key = (str(row["task_id"]), int(row["sample_id"]))
            index[key] = row["sentence_scores"]
    return index


def load_joined_data(canonical: str) -> list[dict]:
    causal_dir = _causal_dir(canonical)
    rh_path = _rh_file(canonical)

    if not causal_dir.exists():
        raise FileNotFoundError(f"Causal matrix dir not found: {causal_dir}")
    if not rh_path.exists():
        raise FileNotFoundError(f"Receiver-head summary not found: {rh_path}")

    rh_index = load_receiver_head_index(rh_path)
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
            continue  # sentence count mismatch — skip

        rows.append({
            "task_id": task_id,
            "sample_id": sample_id,
            "matrix": matrix,
            "rh_scores": rh_scores,
            "M": M,
            "dataset": canonical,
        })

    return rows


# ── Per-sentence metrics ───────────────────────────────────────────────────────

def influence_exerted(matrix: np.ndarray) -> np.ndarray:
    """Mean causal influence sentence i exerts on all later sentences (row mean)."""
    M = matrix.shape[0]
    out = np.full(M, np.nan)
    for i in range(M - 1):
        vals = matrix[i, i + 1:]
        if not np.all(np.isnan(vals)):
            out[i] = np.nanmean(vals)
    return out


def influence_received(matrix: np.ndarray) -> np.ndarray:
    """Mean causal influence sentence j receives from all earlier sentences (col mean)."""
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
    all_received: list[float] = []
    peak_agreements: list[bool] = []
    norm_positions: list[float] = []   # relative position of most-influential sentence

    for row in rows:
        matrix = row["matrix"]
        rh = row["rh_scores"]
        M = row["M"]

        exerted = influence_exerted(matrix)
        received = influence_received(matrix)

        # Collect paired (R-score, influence) for sentences where both are finite
        for i in range(M):
            if np.isfinite(rh[i]) and np.isfinite(exerted[i]):
                all_rh.append(float(rh[i]))
                all_exerted.append(float(exerted[i]))
            if np.isfinite(rh[i]) and np.isfinite(received[i]):
                all_received.append(float(received[i]))

        # Peak agreement: does the sentence with highest R-score also exert the most influence?
        valid = np.isfinite(exerted)
        if valid.sum() >= 2:
            peak_rh = int(np.argmax(rh))
            peak_inf = int(np.nanargmax(np.where(valid, exerted, np.nan)))
            peak_agreements.append(peak_rh == peak_inf)

        # Normalised position of most-influential sentence
        if valid.sum() >= 1:
            peak_inf = int(np.nanargmax(np.where(valid, exerted, np.nan)))
            norm_positions.append(peak_inf / max(M - 1, 1))

    rh_arr = np.array(all_rh)
    ext_arr = np.array(all_exerted)

    # ── Figure 1: Correlation scatter ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Causal Influence vs Receiver-Head R-Score — {dataset_label}", fontsize=13)

    ax = axes[0]
    ax.scatter(rh_arr, ext_arr, alpha=0.35, s=18, color="steelblue")
    r, p = stats.pearsonr(rh_arr, ext_arr)
    m, b = np.polyfit(rh_arr, ext_arr, 1)
    x_line = np.linspace(rh_arr.min(), rh_arr.max(), 100)
    ax.plot(x_line, m * x_line + b, color="tomato", linewidth=1.5)
    ax.set_xlabel("Receiver-head R-score")
    ax.set_ylabel("Mean causal influence exerted")
    ax.set_title(f"r = {r:.3f}  (p = {p:.2e},  n = {len(rh_arr)})")

    # ── Figure 1b: Normalised-position histogram ───────────────────────────────
    ax2 = axes[1]
    ax2.hist(norm_positions, bins=10, range=(0, 1), color="steelblue", alpha=0.75, edgecolor="white")
    ax2.axvline(np.mean(norm_positions), color="tomato", linestyle="--",
                label=f"mean = {np.mean(norm_positions):.2f}")
    ax2.set_xlabel("Relative position of most-influential sentence (0=first, 1=last)")
    ax2.set_ylabel("Count (rollouts)")
    ax2.set_title("Where do thought anchors appear?")
    ax2.legend()

    plt.tight_layout()
    out = FIGURES_DIR / f"causal_correlation_{dataset_label}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ── Figure 2: Representative heatmaps (4 rollouts, largest M) ─────────────
    sample_rows = sorted(rows, key=lambda r: r["M"], reverse=True)[:4]
    fig, axes = plt.subplots(1, len(sample_rows), figsize=(5 * len(sample_rows), 5))
    if len(sample_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, sample_rows):
        matrix = row["matrix"]
        vmax = np.nanpercentile(np.abs(matrix), 95)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{row['task_id']} s{row['sample_id']}\nM={row['M']}", fontsize=9)
        ax.set_xlabel("Target sentence j")
        ax.set_ylabel("Source sentence i")
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle(f"Causal Matrices (largest M) — {dataset_label}", fontsize=12)
    plt.tight_layout()
    out2 = FIGURES_DIR / f"causal_heatmaps_{dataset_label}.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved: {out2}")

    # ── Figure 3: R-score vs influence per sentence, colour by position ────────
    fig, ax = plt.subplots(figsize=(7, 5))
    all_pos: list[float] = []
    all_rh2: list[float] = []
    all_ext2: list[float] = []
    for row in rows:
        M = row["M"]
        exerted = influence_exerted(row["matrix"])
        for i in range(M):
            if np.isfinite(row["rh_scores"][i]) and np.isfinite(exerted[i]):
                all_rh2.append(float(row["rh_scores"][i]))
                all_ext2.append(float(exerted[i]))
                all_pos.append(i / max(M - 1, 1))

    sc = ax.scatter(all_rh2, all_ext2, c=all_pos, cmap="plasma", alpha=0.4, s=18)
    plt.colorbar(sc, ax=ax, label="Relative sentence position")
    ax.set_xlabel("Receiver-head R-score")
    ax.set_ylabel("Mean causal influence exerted")
    ax.set_title(f"Sentence position in trace — {dataset_label}")
    plt.tight_layout()
    out3 = FIGURES_DIR / f"causal_position_{dataset_label}.png"
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"Saved: {out3}")

    # ── Print summary ──────────────────────────────────────────────────────────
    agree_pct = 100 * np.mean(peak_agreements) if peak_agreements else float("nan")
    print(f"\n{'─'*50}")
    print(f"Dataset:          {dataset_label}")
    print(f"Rollouts joined:  {len(rows)}")
    print(f"Sentence pairs:   {len(rh_arr)}")
    print(f"Pearson r:        {r:.3f}  (p={p:.2e})")
    print(f"Peak agreement:   {agree_pct:.1f}%  ({sum(peak_agreements)}/{len(peak_agreements)} rollouts)")
    print(f"Mean anchor pos:  {np.mean(norm_positions):.2f}  (0=start, 1=end)")
    print(f"{'─'*50}\n")


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
