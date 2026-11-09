"""OpenAI-compatible chat client shared by proposals, polish, and eval."""

from __future__ import annotations

from scripts.llm.endpoint import Endpoint
from scripts.llm.parsing import ModelOutputError


def thinking_controls(disable_thinking: bool) -> tuple[str, dict]:
    """Suffix and request body used to turn Qwen3 thinking off.

    llama.cpp honors ``chat_template_kwargs.enable_thinking`` and Qwen3 also
    treats a trailing ``/no_think`` as an instruction to skip the think block.
    """
    if not disable_thinking:
        return "", {}
    return "\n/no_think", {"chat_template_kwargs": {"enable_thinking": False}}


def build_chat_kwargs(
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    extra_body: dict | None = None,
) -> dict:
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def message_text(message: object) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(getattr(part, "text", "") or ""))
        return "\n".join(parts)
    return content or ""


def chat_complete(
    endpoint: Endpoint,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
    extra_body: dict | None = None,
) -> str:
    """Return the assistant text from one chat completion."""
    from openai import OpenAI

    client = OpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key or "local",
        timeout=timeout,
    )
    kwargs = build_chat_kwargs(
        endpoint.model or "",
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    response = client.chat.completions.create(**kwargs)
    if not response.choices:
        raise ModelOutputError("model returned no choices")
    return message_text(response.choices[0].message)
