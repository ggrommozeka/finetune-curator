"""OpenAI-style tool messages and a Qwen3 chat rendering.

Unsloth trains from the ``messages`` list with the tokenizer's Qwen3 template.
``render_qwen3`` is the same ChatML shape that template emits, including tool calls.
"""

from __future__ import annotations

import json
import re

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


def _message_body(message: dict) -> str:
    role = message.get("role") or "user"
    body = message.get("content") or ""
    if message.get("tool_calls"):
        rendered = "\n".join(render_tool_call(call) for call in message["tool_calls"])
        body = (body + "\n" if body else "") + rendered
    if role == "tool":
        body = f"<tool_response>\n{body}\n</tool_response>"
    return body


def render_qwen3(messages: list[dict]) -> str:
    """ChatML rendering Qwen3 tokenizers apply to this message schema."""
    return render_chat(messages, "qwen3")


# Names match Unsloth ``get_chat_template(..., chat_template=name)`` where that family exists.
CHAT_TEMPLATES = ("qwen3", "chatml", "llama3", "gemma")


def render_chat(messages: list[dict], template: str = "qwen3") -> str:
    """Render messages with a chat template. Qwen3 is the default; other 9B families are selectable."""
    name = (template or "qwen3").lower()
    if name not in CHAT_TEMPLATES:
        raise ValueError(f"Unknown chat template {template!r}. Choose one of: {', '.join(CHAT_TEMPLATES)}")
    blocks = []
    for message in messages:
        role = message.get("role") or "user"
        body = _message_body(message)
        if name in {"qwen3", "chatml"}:
            blocks.append(f"{IM_START}{role}\n{body}{IM_END}")
        elif name == "llama3":
            blocks.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{body}<|eot_id|>")
        else:
            blocks.append(f"<start_of_turn>{role}\n{body}<end_of_turn>")
    return "\n".join(blocks) + "\n"


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def clip_diff_context(diff: str, line_start: int | None, line_end: int | None, context_lines: int) -> str:
    """Keep the diff lines around an anchor. Other lines in the hunk are dropped.

    ``context_lines`` is how many new-file lines to keep on each side of the comment.
    Without an anchor, only the first hunk and that many body lines are kept.
    """
    if not diff or context_lines <= 0:
        return diff
    anchor_lo = (line_start or 0) - context_lines if line_start else None
    anchor_hi = (line_end or line_start or 0) + context_lines if line_start else None
    kept: list[str] = []
    header = ""
    new_line = 0
    body_kept = 0
    body_budget = context_lines * 2
    for line in diff.splitlines():
        match = _HUNK_HEADER.match(line)
        if match:
            header = line
            new_line = int(match.group(3))
            body_kept = 0
            continue
        if not header:
            continue
        numbered = new_line
        if line.startswith("+") or line.startswith(" "):
            new_line += 1
        include = False
        if anchor_lo is None:
            include = body_kept < body_budget
        elif anchor_lo <= numbered <= anchor_hi:
            include = True
        if include:
            if not kept or kept[-1] != header:
                kept.append(header)
            kept.append(line)
            body_kept += 1
    return "\n".join(kept) if kept else "\n".join(diff.splitlines()[: body_budget + 1])


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
    """Split a long conversation on turn boundaries. Each piece stays under the budget.

    Later pieces repeat the system prompt and the first user turn, so a chunk
    still knows the task after the earlier tool calls have been cut off.
    """
    if not messages:
        return []
    system = [message for message in messages if message.get("role") == "system"][:1]
    body = [message for message in messages if message.get("role") != "system"]
    if sequence_tokens(system + body, chars_per_token) <= max_tokens:
        return [system + body]
    first_user = next((message for message in body if message.get("role") == "user"), None)
    head = system + ([first_user] if first_user else [])
    rest = body[1:] if first_user and body and body[0] is first_user else body
    if first_user and body and body[0] is not first_user:
        rest = [message for message in body if message is not first_user]
    chunks: list[list[dict]] = []
    current: list[dict] = list(head)
    for message in rest:
        trial = current + [message]
        if sequence_tokens(trial, chars_per_token) <= max_tokens:
            current = trial
            continue
        if len(current) > len(head) or not rest:
            chunks.append(shrink_messages(current, max_tokens, chars_per_token))
        current = shrink_messages(head + [message], max_tokens, chars_per_token)
    if current:
        chunks.append(shrink_messages(current, max_tokens, chars_per_token))
    return [chunk for chunk in chunks if chunk] or [shrink_messages(head or system + body, max_tokens, chars_per_token)]
