"""CLI for labeling reasoning rollout sentences with hosted LLMs."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_anchors_code.analysis.labeling.core import run_labeling_to_jsonl
from thought_anchors_code.analysis.labeling.providers import (
    DEFAULT_MODELS,
    build_client,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create sentence-level function/dependency labels for code reasoning "
            "rollout JSONL files."
        )
    )
    parser.add_argument("rollout_file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/labeled_rollouts.jsonl"),
        help="Destination JSONL path for labeled rollout records.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(DEFAULT_MODELS),
        required=True,
        help="Hosted LLM API provider to use.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model name. Defaults to a small capable model per provider.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key value. Prefer environment variables for normal use.",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible providers, e.g. https://host/v1.",
    )
    parser.add_argument("--prompt-path", type=Path, default=None)
    parser.add_argument("--limit-rollouts", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of rollout labeling requests to launch per batch.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent API requests within each batch.",
    )
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls for rate-limit friendliness.",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a careful annotation assistant. Return only the requested JSON."
        ),
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also label rollouts marked complete=false if they contain reasoning.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite output instead of skipping already labeled rollout keys.",
    )
    parser.add_argument(
        "--no-strict-json-instruction",
        action="store_true",
        help="Do not append an extra JSON-only instruction to prompt.py's prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = build_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        retries=args.retries,
    )
    written = run_labeling_to_jsonl(
        rollout_file=args.rollout_file,
        output_path=args.output,
        client=client,
        prompt_path=args.prompt_path,
        limit_rollouts=args.limit_rollouts,
        resume=not args.no_resume,
        include_incomplete=args.include_incomplete,
        system_prompt=args.system_prompt,
        request_interval_seconds=args.request_interval_seconds,
        strict_json_instruction=not args.no_strict_json_instruction,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )
    print(
        f"Wrote {written} labeled rollout rows to {args.output} "
        f"using {client.config.provider}/{client.config.model}"
    )


if __name__ == "__main__":
    main()

