"""CLI for black-box sentence resampling experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_anchors_code.analysis.blackbox_resampling.core import (
    run_blackbox_resampling_to_jsonl,
)
from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    load_rollouts_jsonl,
)
from thought_anchors_code.config import DEFAULT_MODEL_ID
from thought_anchors_code.data_loading import load_local_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exhaustive black-box sentence resampling over every sentence in a rollout."
    )
    parser.add_argument("rollout_file", type=Path)
    parser.add_argument("attention_summary_file", type=Path)
    parser.add_argument("dataset", choices=["humaneval", "openai_humaneval", "mbpp"])
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, default=Path("results/blackbox_resampling.jsonl"))
    parser.add_argument("--limit-rollouts", type=int, default=None)
    parser.add_argument("--num-resamples", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of intervention/resample continuations to generate together.",
    )
    parser.add_argument(
        "--truncate-percentile",
        type=float,
        default=75.0,
        help="Truncate each rollout to this reasoning sentence percentile before exhaustive resampling.",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Disable percentile truncation and resample every sentence in the full rollout.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollouts = load_rollouts_jsonl(args.rollout_file)
    dataset = load_local_dataset(args.dataset)
    rows_by_task_id = {str(row.get("task_id") or row.get("id")): row for row in dataset}
    written = run_blackbox_resampling_to_jsonl(
        rollouts=rollouts,
        rows_by_task_id=rows_by_task_id,
        attention_summary_path=args.attention_summary_file,
        output_path=args.output,
        model_name_or_path=args.model,
        num_resamples=args.num_resamples,
        limit_rollouts=args.limit_rollouts,
        truncate_to_percentile=None if args.no_truncate else args.truncate_percentile,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        evaluate=not args.no_eval,
    )
    print(f"Wrote {written} black-box resampling rows to {args.output}")


if __name__ == "__main__":
    main()
