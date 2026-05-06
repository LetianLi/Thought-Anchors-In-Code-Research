from types import SimpleNamespace

from thought_anchors_code.analysis.labeling.core import (
    build_labeling_jobs,
    label_jobs_concurrently,
    payload_has_concrete_labels,
    read_completed_label_keys,
)
from thought_anchors_code.analysis.labeling.core import (
    build_labeling_prompt,
    extract_json_object,
    normalize_label_payload,
)
from thought_anchors_code.analysis.whitebox_attention.types import CodeRollout
from thought_anchors_code.analysis.labeling import providers
from thought_anchors_code.analysis.labeling.providers import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_MODELS,
    LLMClient,
    LLMClientConfig,
    _extract_openai_responses_text,
    _redact_url,
)
from thought_anchors_code.analysis.labeling.review_ui import normalize_labeled_row


def test_build_labeling_prompt_replaces_problem_and_sentences() -> None:
    prompt = "Problem: <PROBLEM>\nSentences:\n<SENTENCES>"

    rendered = build_labeling_prompt(
        prompt,
        problem="Write add_one.",
        sentences=["Plan it. ", "Return x + 1."],
        strict_json_instruction=False,
    )

    assert "Problem: Write add_one." in rendered
    assert "1. Plan it." in rendered
    assert "2. Return x + 1." in rendered


def test_extract_json_object_handles_markdown_fence() -> None:
    payload = extract_json_object(
        '```json\n{"1": {"function_tags": ["problem_setup"], "depends_on": []}}\n```'
    )

    assert payload["1"]["function_tags"] == ["problem_setup"]


def test_normalize_label_payload_cleans_invalid_dependencies() -> None:
    labels, warnings = normalize_label_payload(
        {
            "1": {"function_tags": ["problem_setup"], "depends_on": [""]},
            "2": {
                "function_tags": ["active_computation", "made_up"],
                "depends_on": ["1", "2", "9", "abc"],
            },
        },
        sentence_count=3,
    )

    assert labels["1"].function_tags == ["problem_setup"]
    assert labels["2"].function_tags == ["active_computation", "unknown"]
    assert labels["2"].depends_on == ["1"]
    assert labels["3"].function_tags == ["unknown"]
    assert warnings


def test_default_models_use_current_generation_names() -> None:
    assert DEFAULT_MODELS["gemini"] == "gemini-3.1-pro-preview"
    assert DEFAULT_MODELS["openai"].startswith("gpt-5")
    assert DEFAULT_MODELS["claude"].startswith("claude-sonnet-4")
    assert DEFAULT_API_KEY_ENVS["gemini"] == "GEMINI_API_KEY"


def test_build_labeling_jobs_filters_incomplete_and_completed() -> None:
    rollouts = [
        CodeRollout(
            model_id="model",
            dataset_name="mbpp",
            task_id="1",
            sample_id=0,
            complete=True,
            prompt="Task:\nAdd one.\n\nRespond using this format:",
            raw="",
            reasoning="Plan. Return x + 1.",
            answer="",
            is_correct=True,
        ),
        CodeRollout(
            model_id="model",
            dataset_name="mbpp",
            task_id="2",
            sample_id=0,
            complete=False,
            prompt="Task:\nSubtract one.\n\nRespond using this format:",
            raw="",
            reasoning="Plan.",
            answer="",
            is_correct=None,
        ),
    ]

    jobs = build_labeling_jobs(
        rollouts,
        classification_prompt="Problem: <PROBLEM>\n<SENTENCES>",
        completed_keys={("mbpp", "1", 0)},
    )

    assert jobs == []

    jobs = build_labeling_jobs(
        rollouts,
        classification_prompt="Problem: <PROBLEM>\n<SENTENCES>",
        include_incomplete=True,
    )

    assert [job.rollout.task_id for job in jobs] == ["1", "2"]


def test_label_jobs_concurrently_returns_successes() -> None:
    class FakeClient:
        config = SimpleNamespace(provider="fake", model="fake-model")

        def generate(self, prompt, *, system_prompt=None):
            return '{"1": {"function_tags": ["problem_setup"], "depends_on": []}}'

    rollout = CodeRollout(
        model_id="model",
        dataset_name="mbpp",
        task_id="1",
        sample_id=0,
        complete=True,
        prompt="Task:\nAdd one.\n\nRespond using this format:",
        raw="",
        reasoning="Plan.",
        answer="",
        is_correct=True,
    )
    jobs = build_labeling_jobs(
        [rollout],
        classification_prompt="Problem: <PROBLEM>\n<SENTENCES>",
    )

    successes, failures = label_jobs_concurrently(
        jobs,
        client=FakeClient(),
        max_workers=5,
    )

    assert not failures
    assert successes[0].labels["1"].function_tags == ["problem_setup"]


def test_openai_first_party_uses_responses_api(monkeypatch) -> None:
    calls = []

    def fake_post_json(url, payload, *, headers=None, timeout_seconds=60):
        calls.append((url, payload, headers, timeout_seconds))
        return {"output_text": '{"1": {"function_tags": ["problem_setup"], "depends_on": []}}'}

    monkeypatch.setattr(providers, "_post_json", fake_post_json)
    client = LLMClient(
        LLMClientConfig(
            provider="openai",
            model="gpt-5.4-mini",
            api_key="test-key",
        )
    )

    assert client.generate("label this", system_prompt="json only").startswith("{")

    url, payload, headers, _ = calls[0]
    assert url == "https://api.openai.com/v1/responses"
    assert payload["max_output_tokens"] == 8192
    assert payload["text"]["format"]["type"] == "json_object"
    assert headers["Authorization"] == "Bearer test-key"


def test_gemini_uses_api_key_header_not_query_string(monkeypatch) -> None:
    calls = []

    def fake_post_json(url, payload, *, headers=None, timeout_seconds=60):
        calls.append((url, payload, headers, timeout_seconds))
        return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "{}"}]}}]}

    monkeypatch.setattr(providers, "_post_json", fake_post_json)
    client = LLMClient(
        LLMClientConfig(
            provider="gemini",
            model="gemini-3.1-pro-preview",
            api_key="test-key",
        )
    )

    assert client.generate("label this") == "{}"

    url, payload, headers, _ = calls[0]
    assert url.endswith("/models/gemini-3.1-pro-preview:generateContent")
    assert "?" not in url
    assert headers["x-goog-api-key"] == "test-key"
    assert payload["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_max_tokens_raises(monkeypatch) -> None:
    def fake_post_json(url, payload, *, headers=None, timeout_seconds=60):
        return {
            "candidates": [
                {
                    "finishReason": "MAX_TOKENS",
                    "content": {"parts": [{"text": "partial json"}]},
                }
            ]
        }

    monkeypatch.setattr(providers, "_post_json", fake_post_json)
    client = LLMClient(
        LLMClientConfig(
            provider="gemini",
            model="gemini-3.1-pro-preview",
            api_key="test-key",
        )
    )

    try:
        client.generate("label this")
    except RuntimeError as exc:
        assert "MAX_TOKENS" in str(exc)
    else:
        raise AssertionError("Expected MAX_TOKENS response to raise")


def test_review_ui_counts_unknown_as_unlabeled() -> None:
    row = normalize_labeled_row(
        {
            "task_id": "x",
            "sentences": [
                {"index": "1", "text": "Plan."},
                {"index": "2", "text": "??"},
                {"index": "3", "text": "Return."},
            ],
            "labels": {
                "1": {"function_tags": ["problem_setup"], "depends_on": []},
                "2": {"function_tags": ["unknown"], "depends_on": []},
            },
        },
        line_number=1,
    )

    assert row["tag_counts"] == {"problem_setup": 1}
    assert row["labeled_sentence_count"] == 1
    assert row["unlabeled_sentence_count"] == 2


def test_unknown_only_payload_is_not_completed_label() -> None:
    payload = {
        "dataset_name": "mbpp",
        "task_id": "1",
        "sample_id": 0,
        "labels": {"1": {"function_tags": ["unknown"], "depends_on": []}},
    }

    assert not payload_has_concrete_labels(payload)


def test_read_completed_label_keys_skips_unknown_only_rows(tmp_path) -> None:
    output = tmp_path / "labels.jsonl"
    output.write_text(
        "\n".join(
            [
                '{"dataset_name":"mbpp","task_id":"1","sample_id":0,"labels":{"1":{"function_tags":["unknown"],"depends_on":[]}}}',
                '{"dataset_name":"mbpp","task_id":"2","sample_id":0,"labels":{"1":{"function_tags":["problem_setup"],"depends_on":[]}}}',
            ]
        ),
        encoding="utf-8",
    )

    assert read_completed_label_keys(output) == {("mbpp", "2", 0)}


def test_openai_responses_fallback_text_extraction() -> None:
    text = _extract_openai_responses_text(
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "{\"1\": {}}"}],
                }
            ]
        }
    )

    assert text == '{"1": {}}'


def test_redact_url_hides_api_key_query_params() -> None:
    assert _redact_url("https://host/path?key=secret&x=1") == (
        "https://host/path?key=%5BREDACTED%5D&x=1"
    )

