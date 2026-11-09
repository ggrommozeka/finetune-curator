"""OpenAI-style tool messages and a Qwen3 chat rendering.

Unsloth trains from the ``messages`` list with the tokenizer's Qwen3 template.
``render_qwen3`` is the same ChatML shape that template emits, including tool calls.
"""

from __future__ import annotations

import json

from scripts.rules.sessions import estimate_tokens, message_text

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def _arguments(value) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def to_tool_message(message: dict) -> dict:
    """Normalize one transcript turn to OpenCode / OpenAI tool-calling messages."""
    role = str(message.get("role") or "user").lower()
    if role not in {"system", "user", "assistant", "tool"}:
        role = "user"
    content = message_text(message.get("content"))
    out: dict = {"role": role, "content": content}
    calls = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name") or call.get("tool") or ""
        arguments = function.get("arguments")
        if arguments is None:
            arguments = call.get("arguments") or call.get("input") or {}
        calls.append(
            {
                "id": str(call.get("id") or call.get("tool_call_id") or name),
                "type": "function",
                "function": {"name": str(name), "arguments": _arguments(arguments)},
            }
        )
    if calls and role == "assistant":
        out["tool_calls"] = calls
    if role == "tool":
        out["tool_call_id"] = str(message.get("tool_call_id") or message.get("id") or "")
    return out


def render_tool_call(call: dict) -> str:
    function = call.get("function") or {}
    name = function.get("name") or "tool"
    raw = function.get("arguments") or "{}"
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        parsed = {"input": raw}
    if not isinstance(parsed, dict):
        parsed = {"input": parsed}
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in parsed.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"<parameter={key}>")
        lines.append(str(text))
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def render_qwen3(messages: list[dict]) -> str:
    """ChatML rendering Qwen3 tokenizers apply to this message schema."""
    blocks = []
    for message in messages:
        role = message.get("role") or "user"
        body = message.get("content") or ""
        if message.get("tool_calls"):
            rendered = "\n".join(render_tool_call(call) for call in message["tool_calls"])
            body = (body + "\n" if body else "") + rendered
        if role == "tool":
            body = f"<tool_response>\n{body}\n</tool_response>"
        blocks.append(f"{IM_START}{role}\n{body}{IM_END}")
    return "\n".join(blocks) + "\n"


def sequence_tokens(messages: list[dict], chars_per_token: float) -> int:
    return estimate_tokens(json.dumps(messages, ensure_ascii=False), chars_per_token)


def _truncate(message: dict, max_tokens: int, chars_per_token: float) -> dict:
    budget = max(1, int(max_tokens * chars_per_token))
    copied = dict(message)
    content = copied.get("content") or ""
    if len(content) > budget:
        copied["content"] = content[:budget] + "\n… [truncated]"
    return copied


def shrink_messages(messages: list[dict], max_tokens: int, chars_per_token: float) -> list[dict]:
    """Shorten the longest text until the rendered messages fit the budget."""
    current = [dict(message) for message in messages]
    guard = 0
    while current and sequence_tokens(current, chars_per_token) > max_tokens and guard < 20:
        guard += 1
        longest = max(range(len(current)), key=lambda index: len(current[index].get("content") or ""))
        content = current[longest].get("content") or ""
        if len(content) <= 1:
            current.pop(longest)
            continue
        current[longest]["content"] = content[: max(1, len(content) // 2)] + "\n… [truncated]"
    return current


def split_messages(messages: list[dict], max_tokens: int, chars_per_token: float) -> list[list[dict]]:
    """Split a long conversation on turn boundaries. Each piece stays under the budget."""
    if not messages:
        return []
    system = [message for message in messages if message.get("role") == "system"][:1]
    body = [message for message in messages if message.get("role") != "system"]
    if sequence_tokens(system + body, chars_per_token) <= max_tokens:
        return [system + body]
    chunks: list[list[dict]] = []
    current: list[dict] = list(system)
    for message in body:
        trial = current + [message]
        if sequence_tokens(trial, chars_per_token) <= max_tokens:
            current = trial
            continue
        if len(current) > len(system):
            chunks.append(shrink_messages(current, max_tokens, chars_per_token))
        current = shrink_messages(system + [message], max_tokens, chars_per_token)
    if len(current) > len(system):
        chunks.append(shrink_messages(current, max_tokens, chars_per_token))
    return [chunk for chunk in chunks if chunk] or [shrink_messages(system + body, max_tokens, chars_per_token)]
