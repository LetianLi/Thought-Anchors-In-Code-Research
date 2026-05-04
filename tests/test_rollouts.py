from thought_anchors_code.rollouts.collect import parse_reasoning_and_code
from thought_anchors_code.rollouts.prompting import build_code_reasoning_prompt


def test_build_code_reasoning_prompt_includes_expected_sections() -> None:
    prompt = build_code_reasoning_prompt(
        task_prompt="Write a function.",
        starter_code="def foo():\n    pass",
        test_context="assert foo() == 1",
    )
    assert "Task:" in prompt
    assert "Starter code:" in prompt
    assert "Tests / context:" in prompt
    assert "<reasoning>" in prompt
    assert "<code>" in prompt


def test_parse_reasoning_and_code_prefers_tagged_sections() -> None:
    reasoning, code, complete = parse_reasoning_and_code(
        "<reasoning>Plan first.</reasoning><code>def foo():\n    return 1</code>"
    )
    assert reasoning == "Plan first."
    assert code == "def foo():\n    return 1"
    assert complete is True


def test_parse_reasoning_and_code_requires_reasoning_block() -> None:
    reasoning, code, complete = parse_reasoning_and_code("def foo():\n    return 1")
    assert reasoning == "def foo():\n    return 1"
    assert code == ""
    assert complete is False


def test_parse_reasoning_and_code_stores_empty_generation_as_incomplete() -> None:
    reasoning, code, complete = parse_reasoning_and_code("   ")
    assert reasoning == ""
    assert code == ""
    assert complete is False


def test_parse_reasoning_and_code_allows_empty_code() -> None:
    reasoning, code, complete = parse_reasoning_and_code(
        "<reasoning>Plan first.</reasoning>"
    )
    assert reasoning == "Plan first."
    assert code == ""
    assert complete is True


def test_parse_reasoning_and_code_treats_open_reasoning_as_cutoff() -> None:
    reasoning, code, complete = parse_reasoning_and_code(
        "<reasoning>Plan first. Still thinking"
    )
    assert reasoning == "Plan first. Still thinking"
    assert code == ""
    assert complete is False


def test_parse_reasoning_and_code_treats_open_code_as_cutoff_answer() -> None:
    reasoning, code, complete = parse_reasoning_and_code(
        "<reasoning>Plan first.</reasoning><code>def foo():\n    return 1"
    )
    assert reasoning == "Plan first."
    assert code == "def foo():\n    return 1"
    assert complete is False


def test_parse_reasoning_and_code_treats_open_reasoning_and_code_as_cutoff() -> None:
    reasoning, code, complete = parse_reasoning_and_code(
        "<reasoning>Plan first.\n<code>def foo():\n    return 1"
    )
    assert reasoning == "Plan first."
    assert code == "def foo():\n    return 1"
    assert complete is False
