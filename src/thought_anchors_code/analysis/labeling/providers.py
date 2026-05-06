"""Small HTTP clients for rollout labeling LLM providers.

These clients intentionally avoid provider SDK dependencies so the labeling CLI
can run in a fresh research environment with only API keys configured.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any
from urllib import error, parse, request


DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "openai-compatible": "gpt-5.4-mini",
    "gemini": "gemini-3-flash-preview",
    "claude": "claude-sonnet-4-6",
}

DEFAULT_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}


@dataclass(frozen=True)
class LLMClientConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_output_tokens: int = 8192
    retries: int = 3


class LLMClient:
    """Provider-neutral text generation client."""

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                return self._generate_once(prompt, system_prompt=system_prompt)
            except Exception as exc:  # noqa: BLE001 - surface provider errors after retries.
                last_error = exc
                if attempt >= self.config.retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"{self.config.provider} request failed: {last_error}"
        ) from last_error

    def _generate_once(self, prompt: str, *, system_prompt: str | None) -> str:
        provider = self.config.provider
        if provider in {"openai", "openai-compatible"}:
            return self._generate_openai(prompt, system_prompt=system_prompt)
        if provider == "gemini":
            return self._generate_gemini(prompt, system_prompt=system_prompt)
        if provider == "claude":
            return self._generate_claude(prompt, system_prompt=system_prompt)
        raise ValueError(f"Unsupported provider: {provider}")

    def _generate_openai(self, prompt: str, *, system_prompt: str | None) -> str:
        if self.config.provider == "openai-compatible":
            return self._generate_openai_compatible(prompt, system_prompt=system_prompt)
        base_url = self.config.base_url or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/responses"
        input_messages = []
        if system_prompt:
            input_messages.append({"role": "system", "content": system_prompt})
        input_messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.model,
            "input": input_messages,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "text": {"format": {"type": "json_object"}},
        }
        response = _post_json(
            url,
            payload,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout_seconds=self.config.timeout_seconds,
        )
        _raise_for_openai_incomplete(response)
        output_text = response.get("output_text")
        if output_text:
            return str(output_text)
        return _extract_openai_responses_text(response)

    def _generate_openai_compatible(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
    ) -> str:
        base_url = self.config.base_url or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        response = _post_json(
            url,
            payload,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout_seconds=self.config.timeout_seconds,
        )
        finish_reason = response["choices"][0].get("finish_reason")
        if finish_reason and finish_reason not in {"stop"}:
            raise RuntimeError(
                f"OpenAI-compatible response ended with finish_reason={finish_reason}"
            )
        return str(response["choices"][0]["message"]["content"])

    def _generate_gemini(self, prompt: str, *, system_prompt: str | None) -> str:
        model = parse.quote(self.config.model, safe="")
        base_url = self.config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        url = (
            f"{base_url.rstrip('/')}/models/"
            f"{model}:generateContent"
        )
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        response = _post_json(
            url,
            payload,
            headers={"x-goog-api-key": self.config.api_key},
            timeout_seconds=self.config.timeout_seconds,
        )
        candidate = response["candidates"][0]
        finish_reason = candidate.get("finishReason")
        if finish_reason and finish_reason != "STOP":
            raise RuntimeError(f"Gemini response ended with finishReason={finish_reason}")
        parts = candidate["content"]["parts"]
        return "".join(str(part.get("text", "")) for part in parts)

    def _generate_claude(self, prompt: str, *, system_prompt: str | None) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }
        if system_prompt:
            payload["system"] = system_prompt
        response = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout_seconds=self.config.timeout_seconds,
        )
        stop_reason = response.get("stop_reason")
        if stop_reason == "max_tokens":
            raise RuntimeError("Claude response ended because max_tokens was reached")
        return "".join(
            str(block.get("text", ""))
            for block in response["content"]
            if block.get("type") == "text"
        )


def build_client(
    *,
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 60.0,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
    retries: int = 3,
) -> LLMClient:
    provider = provider.strip().lower()
    if provider not in DEFAULT_MODELS:
        allowed = ", ".join(sorted(DEFAULT_MODELS))
        raise ValueError(f"Unsupported provider '{provider}'. Choose from: {allowed}")
    env_name = api_key_env or DEFAULT_API_KEY_ENVS[provider]
    resolved_key = api_key or os.environ.get(env_name)
    if not resolved_key:
        raise ValueError(
            f"Missing API key. Pass --api-key or set the {env_name} environment variable."
        )
    return LLMClient(
        LLMClientConfig(
            provider=provider,
            model=model or DEFAULT_MODELS[provider],
            api_key=resolved_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
    )


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=data,
        headers=request_headers,
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {_redact_url(url)}: {body}") from exc


def _extract_openai_responses_text(response: dict[str, Any]) -> str:
    chunks = []
    for output_item in response.get("output", []):
        if output_item.get("type") != "message":
            continue
        for content_item in output_item.get("content", []):
            text = content_item.get("text")
            if text:
                chunks.append(str(text))
    if not chunks:
        raise RuntimeError("OpenAI Responses API response did not include output text")
    return "".join(chunks)


def _raise_for_openai_incomplete(response: dict[str, Any]) -> None:
    status = response.get("status")
    if status in {"failed", "incomplete", "cancelled"}:
        details = response.get("incomplete_details") or response.get("error") or {}
        raise RuntimeError(f"OpenAI Responses API returned status={status}: {details}")


def _redact_url(url: str) -> str:
    parsed = parse.urlsplit(url)
    if not parsed.query:
        return url
    redacted_query = parse.urlencode(
        [
            (key, "[REDACTED]" if key.lower() in {"key", "api_key"} else value)
            for key, value in parse.parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    return parse.urlunsplit(parsed._replace(query=redacted_query))

