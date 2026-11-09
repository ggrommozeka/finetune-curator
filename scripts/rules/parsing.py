"""Strip model thinking traces and recover a JSON object from a chat reply."""

from __future__ import annotations

import json
import re


class ModelOutputError(ValueError):
    """The model reply was not usable rule JSON."""


_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_ONLY = re.compile(r".*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")


def strip_thinking(text: str) -> str:
    """Remove Qwen-style thinking blocks, including unclosed ones."""
    if not text:
        return ""
    text = _THINK_CLOSED.sub("", text)
    text = _THINK_UNCLOSED.sub("", text)
    text = _THINK_CLOSE_ONLY.sub("", text)
    return text.strip()


def parse_config_text(text: str):
    """Parse JSON or JSONC (comments and trailing commas)."""
    return json.loads(strip_jsonc(text))


def strip_jsonc(src: str) -> str:
    """Drop // and /* */ comments and trailing commas outside strings."""
    out: list[str] = []
    i = 0
    n = len(src)
    in_str = False
    esc = False
    while i < n:
        c = src[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            i += 2
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            continue
        out.append(c)
        i += 1
    text = "".join(out)
    return re.sub(r",(\s*[}\]])", r"\1", text)


def extract_json(text: str):
    """Parse a JSON object or array from a model reply.

    Accepts markdown fences, thinking tags, leading prose, and trailing commas.
    """
    cleaned = strip_thinking(text or "")
    cleaned = _FENCE_OPEN.sub("", cleaned.strip())
    cleaned = _FENCE_CLOSE.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        raise ModelOutputError("empty model reply")

    try:
        return json.loads(strip_jsonc(cleaned))
    except json.JSONDecodeError:
        pass

    starts = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
    if not starts:
        raise ModelOutputError("no JSON object in model reply")
    snippet = strip_jsonc(cleaned[min(starts) :]).lstrip()
    try:
        obj, _end = json.JSONDecoder().raw_decode(snippet)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"invalid JSON: {exc.msg}") from exc
    return obj
