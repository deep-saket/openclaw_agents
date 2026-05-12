"""Created: 2026-03-31

Purpose: Implements endpoint-backed llm adapters for the shared llm platform layer.
"""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from urllib import request

from src.llm.base import BaseLLM
from src.llm.huggingface import HuggingFaceLLM, LLMGeneration
from src.platform_logging.tracing import record_llm_call


@dataclass(slots=True)
class EndpointLLM(BaseLLM):
    """Calls a hosted text-generation endpoint and normalizes its response."""

    endpoint_url: str
    model_name: str = "remote-endpoint"
    timeout_seconds: float = 300.0
    api_key: str | None = None
    auth_header_name: str = "Authorization"
    auth_scheme: str = "Bearer"
    max_new_tokens: int | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    default_body: dict[str, Any] = field(default_factory=dict)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate_result(system_prompt, user_prompt).content

    def generate_result(self, system_prompt: str, user_prompt: str) -> LLMGeneration:
        started = perf_counter()
        payload = self._build_request_payload(system_prompt=system_prompt, user_prompt=user_prompt)
        response_payload = self._post_json(payload)
        result = self._parse_response_payload(response_payload)
        record_llm_call(
            model_name=self.model_name,
            call_kind="generate",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )
        return result

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self.generate_result(system_prompt, user_prompt).content
        candidate = HuggingFaceLLM._extract_json_object(response)
        return HuggingFaceLLM._load_dirty_json(candidate)

    def structured_generate(self, prompt: str, schema: type, **kwargs: Any) -> Any:
        system_prompt = kwargs.pop("system_prompt", "Return only structured JSON.")
        payload = self.generate_json(system_prompt, prompt)
        return schema.model_validate(payload)

    def _build_request_payload(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "model_name": self.model_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.max_new_tokens is not None:
            payload["max_new_tokens"] = self.max_new_tokens
        payload.update(self.default_body)
        return payload

    def _post_json(self, payload: dict[str, Any]) -> Any:
        encoded_body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(self.default_headers)
        if self.api_key:
            header_value = self.api_key
            if self.auth_scheme:
                header_value = f"{self.auth_scheme} {self.api_key}"
            headers[self.auth_header_name] = header_value

        req = request.Request(
            self.endpoint_url,
            data=encoded_body,
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds, context=self._build_ssl_context()) as response:
            body = response.read().decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext | None:
        """Builds SSL context with certifi CA bundle when available.

        Some macOS/Python environments cannot locate system CA roots reliably.
        Using certifi keeps TLS verification enabled while avoiding local trust-store issues.
        """
        try:
            import certifi  # type: ignore

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return None

    @classmethod
    def _parse_response_payload(cls, payload: Any) -> LLMGeneration:
        if isinstance(payload, str):
            return LLMGeneration(content=payload.strip(), raw_text=payload)

        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported endpoint response type: {type(payload).__name__}")

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = cls._coerce_text(message.get("content"))
                    reasoning = cls._coerce_text(message.get("reasoning") or message.get("thinking_content"))
                    raw_text = cls._coerce_text(first_choice.get("text")) or content
                    return LLMGeneration(content=content, thinking_content=reasoning, raw_text=raw_text)
                choice_text = cls._coerce_text(first_choice.get("text"))
                if choice_text:
                    return LLMGeneration(content=choice_text, raw_text=choice_text)

        content = cls._coerce_text(
            payload.get("content")
            or payload.get("output")
            or payload.get("response")
            or payload.get("text")
            or payload.get("generated_text")
        )
        thinking_content = cls._coerce_text(payload.get("thinking_content") or payload.get("reasoning"))
        raw_text = cls._coerce_text(payload.get("raw_text")) or content

        if content is None:
            raise ValueError("Endpoint response did not contain supported text fields.")
        return LLMGeneration(content=content, thinking_content=thinking_content, raw_text=raw_text)

    @staticmethod
    def _coerce_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in value]
            return "".join(parts).strip()
        return str(value).strip()


@dataclass(slots=True)
class RemoteLLM(EndpointLLM):
    """Backward-compatible name for the endpoint-backed adapter."""


@dataclass(slots=True)
class OpenAICompatibleLLM(EndpointLLM):
    """Calls an OpenAI-compatible chat completions endpoint."""

    api_path: str = "/v1/chat/completions"
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.endpoint_url.endswith("/"):
            self.endpoint_url = self.endpoint_url[:-1]
        if self.api_path:
            self.endpoint_url = f"{self.endpoint_url}{self.api_path}"

    def _build_request_payload(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.max_new_tokens is not None:
            payload["max_tokens"] = self.max_new_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(self.default_body)
        return payload


@dataclass(slots=True)
class OpenAILLM(OpenAICompatibleLLM):
    """Calls the hosted OpenAI Chat Completions API using an API key."""

    endpoint_url: str = "https://api.openai.com"


@dataclass(slots=True)
class NvidiaLLM(OpenAICompatibleLLM):
    """Calls NVIDIA hosted Integrate API using an API key."""

    endpoint_url: str = "https://integrate.api.nvidia.com"


@dataclass(slots=True)
class GroqLLM(OpenAICompatibleLLM):
    """Calls the hosted Groq OpenAI-compatible Chat Completions API."""

    endpoint_url: str = "https://api.groq.com/openai"

    def _post_json(self, payload: dict[str, Any]) -> Any:
        # Use the official SDK path for Groq because some environments reject
        # direct urllib calls with provider-edge 403 while SDK requests succeed.
        from groq import Groq

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for GroqLLM.")

        client = Groq(api_key=self.api_key, timeout=self.timeout_seconds)

        request_payload: dict[str, Any] = {
            "model": str(payload.get("model", self.model_name)),
            "messages": payload.get("messages", []),
        }
        if payload.get("max_tokens") is not None:
            request_payload["max_completion_tokens"] = payload["max_tokens"]
        if payload.get("temperature") is not None:
            request_payload["temperature"] = payload["temperature"]

        # Pass through any extra OpenAI-compatible fields configured via default_body.
        for key, value in payload.items():
            if key in {"model", "messages", "max_tokens", "temperature"}:
                continue
            request_payload[key] = value

        completion = client.chat.completions.create(**request_payload)

        content = ""
        if completion.choices and completion.choices[0].message:
            content = completion.choices[0].message.content or ""

        return {
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        }
