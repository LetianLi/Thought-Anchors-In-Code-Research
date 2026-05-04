"""CLI entrypoint for receiver-head pilot analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_anchors_code.analysis.whitebox_attention.receiver_heads import (
    export_receiver_head_summary,
    rank_receiver_heads,
)
from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    load_rollouts_jsonl,
)
from thought_anchors_code.config import (
    CACHE_DIR,
    DEFAULT_MODEL_ID,
    ensure_analysis_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run receiver-head analysis on code reasoning rollouts."
    )
    parser.add_argument(
        "rollout_file", type=Path, help="JSONL file containing code rollouts."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, help="Local model path or model id alias."
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Number of receiver heads to keep."
    )
    parser.add_argument("--proximity-ignore", type=int, default=4)
    parser.add_argument("--control-depth", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/receiver_head_summary.jsonl"),
        help="Output JSONL path for sentence-level receiver-head scores.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR / "whitebox_attention",
        help="Cache directory for sentence-averaged attention tensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_analysis_dirs()
    rollouts = load_rollouts_jsonl(args.rollout_file)
    receiver_heads = rank_receiver_heads(
        rollouts=rollouts,
        model_name_or_path=str(args.model),
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        proximity_ignore=args.proximity_ignore,
        control_depth=args.control_depth,
    )
    export_receiver_head_summary(
        output_path=args.output,
        rollouts=rollouts,
        receiver_heads=receiver_heads,
        model_name_or_path=str(args.model),
        cache_dir=args.cache_dir,
        proximity_ignore=args.proximity_ignore,
        control_depth=args.control_depth,
    )
    print(f"Wrote receiver-head summary to {args.output}")


if __name__ == "__main__":
    main()
