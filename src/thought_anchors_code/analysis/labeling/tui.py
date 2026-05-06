"""Interactive terminal UI for batch labeling reasoning rollouts."""

from __future__ import annotations

from getpass import getpass
import os
from pathlib import Path
import sys
from textwrap import shorten

from thought_anchors_code.analysis.labeling.core import (
    append_labeled_rollout_jsonl,
    build_labeling_jobs,
    label_jobs_concurrently,
    load_classification_prompt,
    read_completed_label_keys,
    summarize_rollout_file_for_labeling,
)
from thought_anchors_code.analysis.labeling.providers import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_MODELS,
    build_client,
)
from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    load_rollouts_jsonl,
)
from thought_anchors_code.config import PROJECT_ROOT, ROLLOUT_DIR


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful annotation assistant. Return only the requested JSON."
)


def main() -> None:
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        print(
            "Interactive rollout labeling TUI.\n\n"
            "Run with:\n"
            "  label-code-rollouts-tui\n"
            "or:\n"
            "  python -m thought_anchors_code.analysis.labeling.tui\n\n"
            "The interface will prompt for rollout file, output path, provider, "
            "model, batch size, concurrency, and resume settings."
        )
        return
    try:
        run_tui()
    except KeyboardInterrupt:
        print("\nStopped.")


def run_tui() -> None:
    clear_screen()
    print_header("Thought Anchor Rollout Labeler")
    print(
        "Import a rollout JSONL file, inspect complete reasoning traces, then label "
        "them in controlled concurrent batches.\n"
    )

    rollout_path = choose_rollout_file()
    output_path = prompt_path(
        "Output JSONL",
        default=PROJECT_ROOT / "results" / f"labeled_{rollout_path.stem}.jsonl",
    )
    provider = choose_one("Provider", sorted(DEFAULT_MODELS), default="gemini")
    model = prompt_text("Model", default=DEFAULT_MODELS[provider])
    api_key_env = prompt_text(
        "API key environment variable",
        default=DEFAULT_API_KEY_ENVS[provider],
    )
    api_key = None
    if api_key_env and not os.environ.get(api_key_env):
        if prompt_yes_no(
            f"{api_key_env} is not set. Paste a key for this run?",
            default=False,
        ):
            api_key = getpass("API key (hidden, not saved): ").strip() or None
    elif not api_key_env:
        api_key = getpass("API key (hidden, not saved): ").strip() or None
    base_url = prompt_text(
        "Base URL override (blank for provider default)",
        default="",
    ) or None
    prompt_file = prompt_path(
        "Prompt.py path",
        default=PROJECT_ROOT / "prompt.py",
        must_exist=True,
    )
    batch_size = prompt_int("Batch size", default=5, minimum=1)
    concurrency = prompt_int(
        "Concurrent requests per batch",
        default=batch_size,
        minimum=1,
    )
    limit_rollouts = prompt_optional_int("Limit rollouts (blank for all)")
    include_incomplete = prompt_yes_no("Include incomplete rollouts?", default=False)
    resume = prompt_yes_no("Resume and skip already labeled output rows?", default=True)
    strict_json = prompt_yes_no("Append strict JSON-only instruction?", default=True)
    max_output_tokens = prompt_int("Max output tokens per request", default=4096, minimum=1)
    temperature = prompt_float("Temperature", default=0.0, minimum=0.0)
    timeout_seconds = prompt_float("Timeout seconds per request", default=60.0, minimum=1.0)
    retries = prompt_int("Retries per request", default=3, minimum=0)

    summary = summarize_rollout_file_for_labeling(
        rollout_file=rollout_path,
        output_path=output_path,
        include_incomplete=include_incomplete,
        resume=resume,
        limit_rollouts=limit_rollouts,
    )
    clear_screen()
    print_header("Run Summary")
    print(f"Rollout file:      {relative_display(rollout_path)}")
    print(f"Output file:       {relative_display(output_path)}")
    print(f"Provider/model:    {provider} / {model}")
    print(f"Batch/concurrency: {batch_size} / {concurrency}")
    print(f"Total rollouts:    {summary.total_rollouts}")
    print(f"Complete rollouts: {summary.complete_rollouts}")
    print(f"Incomplete:        {summary.incomplete_rollouts}")
    print(f"With reasoning:    {summary.rollouts_with_reasoning}")
    print(f"Selected:          {summary.selected_rollouts}")
    print(f"Already labeled:   {summary.already_labeled}")
    print(f"Pending jobs:      {summary.pending_jobs}\n")
    if summary.pending_jobs == 0:
        print("No pending reasoning traces to label.")
        return
    if not prompt_yes_no("Start labeling?", default=True):
        print("No requests sent.")
        return

    client = build_client(
        provider=provider,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env or None,
        base_url=base_url,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output_path.write_text("", encoding="utf-8")
    elif not output_path.exists():
        output_path.write_text("", encoding="utf-8")

    completed_keys = read_completed_label_keys(output_path) if resume else set()
    jobs = build_labeling_jobs(
        load_rollouts_jsonl(rollout_path),
        classification_prompt=load_classification_prompt(prompt_file),
        completed_keys=completed_keys,
        include_incomplete=include_incomplete,
        limit_rollouts=limit_rollouts,
        strict_json_instruction=strict_json,
    )
    run_batches(
        jobs,
        output_path=output_path,
        client=client,
        batch_size=batch_size,
        concurrency=concurrency,
    )


def run_batches(
    jobs,
    *,
    output_path: Path,
    client,
    batch_size: int,
    concurrency: int,
) -> None:
    completed = 0
    failures_total = 0
    cursor = 0
    auto_run = False
    while cursor < len(jobs):
        batch = jobs[cursor : cursor + batch_size]
        clear_screen()
        print_header("Batch Preview")
        print(
            f"Pending: {len(jobs) - cursor} | Completed this run: {completed} | "
            f"Batch size: {len(batch)} | Concurrency: {min(concurrency, len(batch))}\n"
        )
        print_job_preview(batch)
        if not auto_run:
            action = prompt_text(
                "\nEnter=run batch, a=auto-run remaining, s=skip batch, q=quit",
                default="",
            ).lower()
            if action == "q":
                break
            if action == "s":
                cursor += len(batch)
                continue
            if action == "a":
                auto_run = True

        print("\nSending requests...")

        def on_success(labeled) -> None:
            append_labeled_rollout_jsonl(labeled, output_path)
            print(
                "  ok  "
                f"{labeled.dataset_name}/{labeled.task_id} "
                f"sample={labeled.sample_id}"
            )

        def on_failure(failure) -> None:
            rollout = failure.job.rollout
            print(
                "  err "
                f"{rollout.dataset_name}/{rollout.task_id} sample={rollout.sample_id}: "
                f"{failure.error}"
            )

        successes, failures = label_jobs_concurrently(
            batch,
            client=client,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_workers=concurrency,
            on_success=on_success,
            on_failure=on_failure,
        )
        completed += len(successes)
        failures_total += len(failures)
        if failures and not auto_run:
            if prompt_yes_no("Retry failed requests now?", default=True):
                jobs[cursor : cursor + len(batch)] = [failure.job for failure in failures]
                continue
        cursor += len(batch)
        if not auto_run:
            prompt_text("Press Enter for the next batch", default="")

    clear_screen()
    print_header("Labeling Complete")
    print(f"Wrote {completed} labeled rollout rows to {relative_display(output_path)}.")
    if failures_total:
        print(f"Encountered {failures_total} failed request(s); successful rows were saved.")


def choose_rollout_file() -> Path:
    candidates = discover_jsonl_files()
    if candidates:
        print("Available JSONL files:\n")
        for index, path in enumerate(candidates, start=1):
            print(f"  {index:>2}. {relative_display(path)}")
        print("   m. Manual path\n")
        choice = prompt_text("Choose file", default="1")
        if choice.lower() != "m":
            try:
                return candidates[int(choice) - 1]
            except (ValueError, IndexError):
                print("Invalid choice; falling back to manual path.")
    return prompt_path("Rollout JSONL path")


def discover_jsonl_files() -> list[Path]:
    search_roots = [ROLLOUT_DIR, PROJECT_ROOT / "results"]
    files: list[Path] = []
    for root in search_roots:
        if root.exists():
            files.extend(sorted(root.glob("*.jsonl")))
    return files


def print_job_preview(jobs) -> None:
    print("  #   task id                         sample  sentences  correct")
    print("  --  ------------------------------  ------  ---------  -------")
    for offset, job in enumerate(jobs, start=1):
        rollout = job.rollout
        correct = "?" if rollout.is_correct is None else str(bool(rollout.is_correct))
        print(
            f"  {offset:>2}  "
            f"{shorten(rollout.task_id, width=30, placeholder='...'):<30}  "
            f"{rollout.sample_id:>6}  "
            f"{len(job.sentences):>9}  "
            f"{correct:<7}"
        )


def choose_one(label: str, options: list[str], *, default: str) -> str:
    default_index = options.index(default) + 1 if default in options else 1
    print(f"{label}:")
    for index, option in enumerate(options, start=1):
        suffix = " (default)" if index == default_index else ""
        print(f"  {index}. {option}{suffix}")
    while True:
        raw = prompt_text(f"{label} number", default=str(default_index))
        try:
            return options[int(raw) - 1]
        except (ValueError, IndexError):
            print("Please choose one of the listed numbers.")


def prompt_text(label: str, *, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else default


def prompt_path(
    label: str,
    *,
    default: Path | None = None,
    must_exist: bool = False,
) -> Path:
    default_text = str(default) if default is not None else ""
    while True:
        value = prompt_text(label, default=default_text)
        if not value:
            print("Path is required.")
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if must_exist and not path.exists():
            print(f"Path does not exist: {path}")
            continue
        return path


def prompt_int(label: str, *, default: int, minimum: int) -> int:
    while True:
        raw = prompt_text(label, default=str(default))
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def prompt_optional_int(label: str) -> int | None:
    while True:
        raw = input(f"{label}: ").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer or leave blank.")
            continue
        if value < 1:
            print("Please enter a positive integer or leave blank.")
            continue
        return value


def prompt_float(label: str, *, default: float, minimum: float) -> float:
    while True:
        raw = prompt_text(label, default=str(default))
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def prompt_yes_no(label: str, *, default: bool) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{default_text}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer y or n.")


def print_header(title: str) -> None:
    width = max(72, len(title) + 8)
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def relative_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()

