"""Plotting helpers for Figure 4 style receiver-head visualizations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from thought_anchors_code.analysis.whitebox_attention.attention_extraction import (
    build_sentence_attention_cache,
)
from thought_anchors_code.analysis.whitebox_attention.receiver_heads import (
    get_trace_vertical_scores,
)
from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    split_reasoning_steps,
)
from thought_anchors_code.analysis.whitebox_attention.types import CodeRollout


@dataclass(frozen=True)
class Figure4Artifacts:
    figure_path: Path
    matrix_path: Path
    histogram_path: Path
    metadata_path: Path


def generate_figure4_artifacts(
    rollouts: Sequence[CodeRollout],
    model_name_or_path: str,
    output_dir: Path,
    rollout_index: int = 0,
    layer_index: int | None = None,
    proximity_ignore: int = 4,
    control_depth: bool = False,
    cache_dir: Path | None = None,
) -> Figure4Artifacts:
    if not rollouts:
        raise ValueError("No rollouts available for plotting.")
    if rollout_index < 0 or rollout_index >= len(rollouts):
        raise ValueError(
            f"rollout_index={rollout_index} is out of range for {len(rollouts)} rollouts."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rollout = _clean_rollout_for_plotting(rollouts[rollout_index])
    sentences, vertical_scores = get_trace_vertical_scores(
        trace=rollout,
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )

    num_layers = vertical_scores.shape[0]
    chosen_layer = num_layers - 1 if layer_index is None else layer_index
    if chosen_layer < 0 or chosen_layer >= num_layers:
        raise ValueError(
            f"layer_index={chosen_layer} is out of range for {num_layers} layers."
        )

    layer_scores = vertical_scores[chosen_layer]
    head_kurtoses = stats.kurtosis(
        layer_scores, axis=1, fisher=True, bias=True, nan_policy="omit"
    )
    valid_heads = np.flatnonzero(~np.isnan(head_kurtoses))
    if len(valid_heads) == 0:
        raise ValueError(
            f"Layer {chosen_layer} produced no valid head kurtosis values."
        )
    highlighted_head = int(valid_heads[np.argmax(head_kurtoses[valid_heads])])

    matrices = build_sentence_attention_cache(
        text=rollout.reasoning,
        sentences=sentences,
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
    )
    head_matrix = matrices[chosen_layer, highlighted_head]

    all_head_kurtoses = compute_rollout_head_kurtoses(
        rollouts=rollouts,
        model_name_or_path=model_name_or_path,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
        cache_dir=cache_dir,
    )

    figure_path = output_dir / "figure4_receiver_heads.png"
    matrix_path = output_dir / "figure4_head_matrix.png"
    histogram_path = output_dir / "figure4_kurtosis_histogram.png"
    metadata_path = output_dir / "figure4_metadata.txt"

    _plot_combined_figure(
        figure_path=figure_path,
        layer_scores=layer_scores,
        highlighted_head=highlighted_head,
        layer_index=chosen_layer,
    )
    _plot_head_matrix(
        matrix_path=matrix_path,
        head_matrix=head_matrix,
        highlighted_head=highlighted_head,
        layer_index=chosen_layer,
    )
    _plot_kurtosis_histogram(
        histogram_path=histogram_path,
        all_head_kurtoses=all_head_kurtoses,
    )
    metadata_path.write_text(
        _build_metadata_text(
            rollout=rollout,
            sentences=sentences,
            layer_index=chosen_layer,
            highlighted_head=highlighted_head,
            head_kurtoses=head_kurtoses,
            all_head_kurtoses=all_head_kurtoses,
        ),
        encoding="utf-8",
    )
    return Figure4Artifacts(
        figure_path=figure_path,
        matrix_path=matrix_path,
        histogram_path=histogram_path,
        metadata_path=metadata_path,
    )


def compute_rollout_head_kurtoses(
    rollouts: Sequence[CodeRollout],
    model_name_or_path: str,
    proximity_ignore: int = 4,
    control_depth: bool = False,
    cache_dir: Path | None = None,
) -> np.ndarray:
    per_rollout = []
    for rollout in rollouts:
        cleaned_rollout = _clean_rollout_for_plotting(rollout)
        try:
            _, vertical_scores = get_trace_vertical_scores(
                trace=cleaned_rollout,
                model_name_or_path=model_name_or_path,
                cache_dir=cache_dir,
                proximity_ignore=proximity_ignore,
                control_depth=control_depth,
            )
        except ValueError:
            continue
        per_rollout.append(
            stats.kurtosis(
                vertical_scores, axis=2, fisher=True, bias=True, nan_policy="omit"
            )
        )

    if not per_rollout:
        raise ValueError("No valid rollouts available for kurtosis plotting.")
    stacked = np.asarray(per_rollout, dtype=np.float32)
    return np.nanmedian(stacked, axis=0)


def _clean_rollout_for_plotting(rollout: CodeRollout) -> CodeRollout:
    cleaned_reasoning = rollout.reasoning.replace("<think>", "").replace("</think>", "")
    return replace(rollout, reasoning=cleaned_reasoning)


def _plot_combined_figure(
    figure_path: Path,
    layer_scores: np.ndarray,
    highlighted_head: int,
    layer_index: int,
) -> None:
    sentence_positions = np.arange(layer_scores.shape[1])
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for head_index, head_scores in enumerate(layer_scores):
        color = "#0b4cc2" if head_index == highlighted_head else None
        linewidth = 2.5 if head_index == highlighted_head else 1.2
        alpha = 1.0 if head_index == highlighted_head else 0.8
        ax.plot(
            sentence_positions,
            head_scores,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_title(f"Layer {layer_index + 1} attention heads")
    ax.set_xlabel("Sentence position")
    ax.set_ylabel("Vertical attention score")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_head_matrix(
    matrix_path: Path,
    head_matrix: np.ndarray,
    highlighted_head: int,
    layer_index: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    image = ax.imshow(head_matrix, cmap="Blues", aspect="auto", origin="upper")
    ax.set_title(f"Layer {layer_index + 1}, Head {highlighted_head} matrix")
    ax.set_xlabel("Source sentence")
    ax.set_ylabel("Receiving sentence")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(matrix_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_kurtosis_histogram(
    histogram_path: Path, all_head_kurtoses: np.ndarray
) -> None:
    values = all_head_kurtoses.reshape(-1)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("No valid kurtosis values available for histogram.")

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.hist(
        values,
        bins=min(40, max(10, len(values) // 3)),
        color="#2890ff",
        edgecolor="#2890ff",
    )
    ax.set_title("Histogram of attention head\nvertical score kurtoses")
    ax.set_xlabel("Kurtosis")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_metadata_text(
    rollout: CodeRollout,
    sentences: Sequence[str],
    layer_index: int,
    highlighted_head: int,
    head_kurtoses: np.ndarray,
    all_head_kurtoses: np.ndarray,
) -> str:
    top_head_indices = np.argsort(np.nan_to_num(head_kurtoses, nan=-np.inf))[::-1][:10]
    top_lines = [
        f"  head {int(head)}: kurtosis={float(head_kurtoses[head]):.4f}"
        for head in top_head_indices
        if not np.isnan(head_kurtoses[head])
    ]
    sentence_lines = [
        f"  {index}: {sentence.strip()}" for index, sentence in enumerate(sentences)
    ]
    return "\n".join(
        [
            f"task_id: {rollout.task_id}",
            f"sample_id: {rollout.sample_id}",
            f"dataset_name: {rollout.dataset_name}",
            f"model_id: {rollout.model_id}",
            f"complete: {rollout.complete}",
            f"layer_index: {layer_index}",
            f"highlighted_head: {highlighted_head}",
            f"num_sentences: {len(sentences)}",
            f"global_kurtosis_median_min: {float(np.nanmin(all_head_kurtoses)):.4f}",
            f"global_kurtosis_median_max: {float(np.nanmax(all_head_kurtoses)):.4f}",
            "top_heads_in_layer:",
            *top_lines,
            "sentences:",
            *sentence_lines,
        ]
    )
