import torch

from thought_anchors_code.rollouts.collect import (
    _format_model_prompt,
    _get_eos_token_ids,
    _set_generation_seed,
    _trim_after_eos,
    append_rollout_jsonl,
    parse_reasoning_and_code,
    read_completed_rollout_keys,
)
from thought_anchors_code.analysis.whitebox_attention.types import CodeRollout
from thought_anchors_code.rollouts.evaluate import evaluate_generated_code
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


def test_parse_reasoning_and_code_accepts_think_and_fenced_code() -> None:
    reasoning, code, complete = parse_reasoning_and_code(
        "<think>Plan first.</think>\n```python\ndef foo():\n    return 1\n```"
    )
    assert reasoning == "Plan first."
    assert code == "def foo():\n    return 1"
    assert complete is True


def test_parse_reasoning_and_code_treats_open_think_as_cutoff() -> None:
    reasoning, code, complete = parse_reasoning_and_code("<think>Plan first.")
    assert reasoning == "Plan first."
    assert code == ""
    assert complete is False


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


def test_parse_reasoning_and_code_recovers_code_from_malformed_reasoning_tag() -> None:
    reasoning, code, complete = parse_reasoning_and_code(
        "<reasoning\nPlan first.\n<code>def foo():\n    return 1</code>"
    )
    assert reasoning == "Plan first."
    assert code == "def foo():\n    return 1"
    assert complete is False


def test_evaluate_generated_code_checks_mbpp_visible_and_challenge_tests() -> None:
    row = {
        "test_setup_code": "",
        "test_list": ['assert remove_Occ("hello", "l") == "heo"'],
        "challenge_test_list": ['assert remove_Occ("hellolloll", "l") == "helollol"'],
    }
    bad_code = "def remove_Occ(s, ch):\n    return s.replace(ch, '')"
    good_code = """
def remove_Occ(s, ch):
    first = s.find(ch)
    if first == -1:
        return s
    s = s[:first] + s[first + 1:]
    last = s.rfind(ch)
    if last == -1:
        return s
    return s[:last] + s[last + 1:]
"""
    assert evaluate_generated_code(bad_code, row, "mbpp") is False
    assert evaluate_generated_code(good_code, row, "mbpp") is True


def test_read_completed_rollout_keys(tmp_path) -> None:
    output = tmp_path / "rollouts.jsonl"
    output.write_text(
        '{"model_id":"m","dataset_name":"mbpp","task_id":"11","sample_id":0,"complete":true,"is_correct":true}\n'
        '{"model_id":"m","dataset_name":"mbpp","task_id":"12","sample_id":0,"complete":false,"is_correct":false}\n'
        '{"model_id":"m","dataset_name":"mbpp","task_id":"13","sample_id":0,"complete":true,"is_correct":false}\n',
        encoding="utf-8",
    )
    assert read_completed_rollout_keys(output) == {("m", "mbpp", "11", 0)}


def test_format_model_prompt_uses_chat_template_when_available() -> None:
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is False
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["enable_thinking"] is False
            return f"chat:{messages[-1]['content']}"

    assert _format_model_prompt(Tokenizer(), "hello") == "chat:hello"


def test_get_eos_token_ids_includes_chat_end_token() -> None:
    class Tokenizer:
        eos_token_id = 1

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 2, "<|endoftext|>": 1}.get(token, -1)

    assert _get_eos_token_ids(Tokenizer()) == [1, 2]


def test_trim_after_eos_removes_batched_padding_tokens() -> None:
    token_ids = torch.tensor([10, 11, 2, 2, 2])
    trimmed = _trim_after_eos(token_ids, [1, 2])
    assert trimmed.tolist() == [10, 11, 2]


def test_set_generation_seed_makes_torch_sampling_reproducible() -> None:
    _set_generation_seed(123)
    first = torch.rand(3)
    _set_generation_seed(123)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_append_rollout_jsonl_keeps_unicode_readable(tmp_path) -> None:
    output = tmp_path / "rollouts.jsonl"
    append_rollout_jsonl(
        CodeRollout(
            model_id="m",
            dataset_name="mbpp",
            task_id="1",
            sample_id=0,
            complete=True,
            prompt="p",
            raw="4 × x²",
            reasoning="4 × x²",
            answer="def f():\n    return 1",
            is_correct=True,
        ),
        output,
    )
    text = output.read_text(encoding="utf-8")
    assert "×" in text
    assert "²" in text
    assert "\\u00d7" not in text
