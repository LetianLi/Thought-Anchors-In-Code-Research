"""Prompt helpers for collecting code reasoning rollouts."""

from __future__ import annotations


DEFAULT_REASONING_INSTRUCTIONS = (
    "Solve the coding task. Keep the reasoning concise, then write the final Python solution. "
    "Put the final code after the reasoning."
)


def build_code_reasoning_prompt(
    task_prompt: str,
    starter_code: str | None = None,
    test_context: str | None = None,
    reasoning_instructions: str = DEFAULT_REASONING_INSTRUCTIONS,
) -> str:
    sections = [reasoning_instructions.strip(), "", "Task:", task_prompt.strip()]
    if starter_code and starter_code.strip():
        sections.extend(["", "Starter code:", starter_code.rstrip()])
    if test_context and test_context.strip():
        sections.extend(["", "Tests / context:", test_context.rstrip()])
    sections.extend(
        [
            "",
            "Respond using this format:",
            "<reasoning>",
            "your step-by-step reasoning",
            "</reasoning>",
            "<code>",
            "your final Python code",
            "</code>",
        ]
    )
    return "\n".join(sections)
