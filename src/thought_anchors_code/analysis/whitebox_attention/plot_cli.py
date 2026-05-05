"""CLI for Figure 4 style receiver-head plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_anchors_code.analysis.whitebox_attention.plots import (
    generate_figure4_artifacts,
)
from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    load_rollouts_jsonl,
    truncate_rollouts_to_sentence_percentile,
)
from thought_anchors_code.config import (
    CACHE_DIR,
    DEFAULT_MODEL_ID,
    ensure_analysis_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 4 style receiver-head plots from rollout JSONL."
    )
    parser.add_argument(
        "rollout_file", type=Path, help="JSONL file containing code rollouts."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, help="Local model path or model id alias."
    )
    parser.add_argument(
        "--rollout-index",
        type=int,
        default=0,
        help="Which rollout to visualize for the line plot and matrix inset.",
    )
    parser.add_argument(
        "--all-rollouts",
        action="store_true",
        help="Generate a separate demo folder for each rollout in the JSONL file.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="Zero-based layer index to visualize. Defaults to the final layer.",
    )
    parser.add_argument("--proximity-ignore", type=int, default=4)
    parser.add_argument("--control-depth", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figure4"),
        help="Directory for generated PNGs and metadata.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR / "whitebox_attention",
        help="Cache directory for sentence-averaged attention tensors.",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Analyze full reasoning traces instead of truncating at the input file's 75th percentile sentence count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_analysis_dirs()
    rollouts = load_rollouts_jsonl(args.rollout_file)
    if not args.no_truncate:
        rollouts, max_sentences = truncate_rollouts_to_sentence_percentile(rollouts)
        if max_sentences is not None:
            print(
                f"[attention] Truncated reasoning traces to p75 sentence count: {max_sentences}"
            )
    if args.all_rollouts:
        for rollout_index in range(len(rollouts)):
            rollout_output_dir = args.output_dir / f"rollout_{rollout_index}"
            artifacts = generate_figure4_artifacts(
                rollouts=rollouts,
                model_name_or_path=str(args.model),
                output_dir=rollout_output_dir,
                rollout_index=rollout_index,
                layer_index=args.layer_index,
                proximity_ignore=args.proximity_ignore,
                control_depth=args.control_depth,
                cache_dir=args.cache_dir,
            )
            print(f"Wrote Figure 4 plot to {artifacts.figure_path}")
            print(f"Wrote head matrix plot to {artifacts.matrix_path}")
            print(f"Wrote kurtosis histogram to {artifacts.histogram_path}")
            print(f"Wrote plot metadata to {artifacts.metadata_path}")
        return

    artifacts = generate_figure4_artifacts(
        rollouts=rollouts,
        model_name_or_path=str(args.model),
        output_dir=args.output_dir,
        rollout_index=args.rollout_index,
        layer_index=args.layer_index,
        proximity_ignore=args.proximity_ignore,
        control_depth=args.control_depth,
        cache_dir=args.cache_dir,
    )
    print(f"Wrote Figure 4 plot to {artifacts.figure_path}")
    print(f"Wrote head matrix plot to {artifacts.matrix_path}")
    print(f"Wrote kurtosis histogram to {artifacts.histogram_path}")
    print(f"Wrote plot metadata to {artifacts.metadata_path}")


if __name__ == "__main__":
    main()
