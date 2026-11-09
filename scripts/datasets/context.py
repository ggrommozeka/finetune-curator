"""Pack review context for the 9B model, highest priority first.

The budget is the whole review sample. Sections that do not fit are dropped,
except the diff, which is clipped to the remaining tokens.
"""

from __future__ import annotations

import re

from scripts.ingest.opencode_sessions import estimate_tokens

_DEF = re.compile(
    r"^(?P<indent>[ \t]*)(?:async\s+def|def|class|function|func|fn)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_SKIP_CALLS = {"if", "for", "while", "switch", "return", "catch", "print"}


def _clip(text: str, max_tokens: int, chars_per_token: float) -> str:
    if max_tokens <= 0:
        return ""
    budget = max(1, int(max_tokens * chars_per_token))
    if len(text) <= budget:
        return text
    return text[:budget] + "\n… [truncated]"


def function_at(content: str, line: int | None) -> str:
    """Return the function or class that contains ``line``, or a short file head."""
    if not content:
        return ""
    rows = content.splitlines()
    if line is None or line < 1 or len(rows) <= 80:
        return "\n".join(rows[:80])
    starts = []
    for match in _DEF.finditer(content):
        starts.append((content[: match.start()].count("\n") + 1, match.group("name"), len(match.group("indent"))))
    container = None
    for start, name, indent in starts:
        if start <= line:
            container = (start, name, indent)
        else:
            break
    if container is None:
        begin = max(1, line - 20)
        return "\n".join(rows[begin - 1 : line + 20])
    start, _name, indent = container
    end = len(rows)
    for other_start, _other, other_indent in starts:
        if other_start > start and other_indent <= indent:
            end = other_start - 1
            break
    return "\n".join(rows[start - 1 : end])


def called_names(diff: str) -> list[str]:
    names = []
    for name in _CALL.findall(diff or ""):
        if name in _SKIP_CALLS or name in names:
            continue
        names.append(name)
    return names[:8]


def definitions_and_calls(content: str, names: list[str]) -> str:
    if not content or not names:
        return ""
    rows = content.splitlines()
    blocks = []
    for name in names:
        definition = ""
        for match in _DEF.finditer(content):
            if match.group("name") != name:
                continue
            start = content[: match.start()].count("\n")
            definition = "\n".join(rows[start : start + 12])
            break
        sites = []
        for index, row in enumerate(rows, 1):
            if f"{name}(" in row and not row.strip().startswith(("def ", "class ", "function ", "func ", "fn ")):
                sites.append(f"{index}: {row.strip()}")
            if len(sites) >= 4:
                break
        if definition or sites:
            blocks.append(definition + ("\ncall sites:\n" + "\n".join(sites) if sites else ""))
    return "\n\n".join(block for block in blocks if block.strip())


def pack_sections(sections: list[tuple[int, str, str]], max_tokens: int, chars_per_token: float = 3.0) -> str:
    """Add sections in ascending priority number until the budget is spent."""
    ordered = sorted(sections, key=lambda item: item[0])
    used = 0
    parts = []
    for _priority, title, text in ordered:
        body = (text or "").strip()
        if not body:
            continue
        block = f"## {title}\n{body}"
        cost = estimate_tokens(block, chars_per_token)
        remain = max_tokens - used
        if remain <= 0:
            break
        if cost > remain:
            if _priority == 1:
                clipped = _clip(block, remain, chars_per_token)
                if clipped.strip():
                    parts.append(clipped)
            break
        parts.append(block)
        used += cost
    return "\n\n".join(parts)


def review_context(
    *,
    diff: str,
    title: str,
    description: str,
    issues: list[dict],
    base_file: str,
    line_start: int | None,
    conventions: str,
    max_tokens: int,
    chars_per_token: float = 3.0,
) -> str:
    issue_text = "\n\n".join(
        f"#{item.get('iid')} {item.get('title') or ''}\n{item.get('description') or ''}".strip()
        for item in issues or []
        if (item.get("title") or item.get("description"))
    )
    symbols = definitions_and_calls(base_file, called_names(diff))
    changed = function_at(base_file, line_start)
    return pack_sections(
        [
            (1, "Diff", diff),
            (2, "Merge request", f"{title}\n{description}".strip()),
            (3, "Linked issues", issue_text),
            (4, "Changed code at base", changed),
            (5, "Symbols and call sites", symbols),
            (6, "Project conventions", conventions),
        ],
        max_tokens,
        chars_per_token,
    )
