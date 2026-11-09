"""Deduplicate and merge similar harness rules without calling a model."""

from __future__ import annotations

import hashlib
import re

CATEGORIES = [
    "tool-use",
    "looping",
    "verification",
    "editing",
    "planning",
    "communication",
    "context",
    "safety",
    "other",
]

# Keys are already space-normalized (lowercase, hyphens and underscores as spaces).
_SYNONYMS = {
    "tool use": "tool-use",
    "tools": "tool-use",
    "tool": "tool-use",
    "tooluse": "tool-use",
    "loop": "looping",
    "loops": "looping",
    "looping": "looping",
    "retry": "looping",
    "verification": "verification",
    "verify": "verification",
    "testing": "verification",
    "editing": "editing",
    "edit": "editing",
    "edits": "editing",
    "planning": "planning",
    "plan": "planning",
    "communication": "communication",
    "style": "communication",
    "context": "context",
    "safety": "safety",
    "security": "safety",
    "other": "other",
    "инструменты": "tool-use",
    "циклы": "looping",
    "цикл": "looping",
    "проверка": "verification",
    "правки": "editing",
    "планирование": "planning",
    "общение": "communication",
    "контекст": "context",
    "безопасность": "safety",
    "другое": "other",
}


def category_rank(category: str) -> int:
    try:
        return CATEGORIES.index(category)
    except ValueError:
        return len(CATEGORIES)


def category_heading(slug: str) -> str:
    label = (slug or "other").replace("-", " ").replace("_", " ").strip()
    if not label:
        return "Other"
    return label[0].upper() + label[1:]


def slug_category(value: str) -> str:
    raw = (value or "").strip().lower().replace("_", " ").replace("-", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    if raw in _SYNONYMS:
        return _SYNONYMS[raw]
    compact = raw.replace(" ", "")
    if compact in _SYNONYMS:
        return _SYNONYMS[compact]
    slug = re.sub(r"[^\w\s-]", "", raw, flags=re.UNICODE)
    slug = re.sub(r"\s+", "-", slug).strip("-")
    return slug or "other"


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    norm = normalize_text(text)
    return set(norm.split()) if norm else set()


def similarity_scores(a: str, b: str) -> tuple[float, float]:
    """Return (jaccard, containment) on word tokens. Containment uses the smaller set."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0, 0.0
    inter = len(ta & tb)
    jaccard = inter / len(ta | tb)
    containment = inter / min(len(ta), len(tb))
    return jaccard, containment


def should_merge(a: dict, b: dict, threshold: float = 0.6) -> bool:
    jaccard, containment = similarity_scores(a.get("text") or "", b.get("text") or "")
    smaller = min(len(_tokens(a.get("text") or "")), len(_tokens(b.get("text") or "")))
    same = (a.get("category") or "other") == (b.get("category") or "other")
    if same and (jaccard >= threshold or (smaller >= 4 and containment >= 0.85)):
        return True
    # Near-duplicates that the model filed under different categories.
    if smaller >= 5 and containment >= 0.92:
        return True
    return False


def choose_text(a: str, b: str) -> str:
    """Prefer a concise instruction when the other wording is much longer."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 40 and len(long) > len(short) * 2 and len(short) <= 400:
        return short
    return long


def _uniq(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        item = (item or "").strip()
        if item and item not in out:
            out.append(item)
    return out


def combine(a: dict, b: dict) -> dict:
    text = choose_text(a.get("text") or "", b.get("text") or "")
    winner = a if text == (a.get("text") or "").strip() else b
    other = b if winner is a else a
    category = winner.get("category") or "other"
    if category == "other" and other.get("category") not in (None, "", "other"):
        category = other["category"]
    session_ids = sorted(
        {str(s) for s in (a.get("session_ids") or []) if str(s).strip()}
        | {str(s) for s in (b.get("session_ids") or []) if str(s).strip()}
    )
    evidence = " … ".join(_uniq([a.get("evidence") or "", b.get("evidence") or ""]))[:500]
    rationale = " ".join(_uniq([a.get("rationale") or "", b.get("rationale") or ""]))[:800]
    return {
        "text": text,
        "category": category,
        "rationale": rationale,
        "session_ids": session_ids,
        "evidence": evidence,
    }


def make_rule_id(category: str, text: str) -> str:
    digest = hashlib.sha1(f"{category}\n{normalize_text(text)}".encode()).hexdigest()[:10]
    return f"r_{digest}"


def _cluster(rules: list[dict], threshold: float) -> list[dict]:
    clusters: list[dict] = []
    for rule in rules:
        placed = False
        for i, acc in enumerate(clusters):
            if should_merge(acc, rule, threshold):
                clusters[i] = combine(acc, rule)
                placed = True
                break
        if not placed:
            clusters.append(dict(rule))
    return clusters


def _same_cluster(a: dict, b: dict) -> bool:
    return (
        a.get("text") == b.get("text")
        and a.get("category") == b.get("category")
        and list(a.get("session_ids") or []) == list(b.get("session_ids") or [])
        and a.get("evidence") == b.get("evidence")
    )


def merge_rules(rules: list[dict], threshold: float = 0.6) -> list[dict]:
    """Merge similar rules, union their source sessions, and assign stable ids."""
    prepared: list[dict] = []
    for rule in rules:
        text = re.sub(r"\s+", " ", str(rule.get("text") or "")).strip()
        if not text:
            continue
        prepared.append(
            {
                "text": text,
                "category": slug_category(str(rule.get("category") or "other")),
                "rationale": str(rule.get("rationale") or "").strip(),
                "session_ids": [str(s) for s in (rule.get("session_ids") or []) if str(s).strip()],
                "evidence": str(rule.get("evidence") or "").strip(),
            }
        )
    prepared.sort(key=lambda r: (r["category"], normalize_text(r["text"]), r["text"]))

    clusters = prepared
    for _ in range(6):
        nxt = _cluster(clusters, threshold)
        if len(nxt) == len(clusters) and all(_same_cluster(a, b) for a, b in zip(nxt, clusters)):
            clusters = nxt
            break
        clusters = nxt

    finished: list[dict] = []
    for rule in clusters:
        rule = dict(rule)
        rule["session_ids"] = sorted({str(s) for s in rule.get("session_ids") or []})
        rule["id"] = make_rule_id(rule["category"], rule["text"])
        finished.append(rule)

    by_id: dict[str, dict] = {}
    for rule in finished:
        current = rule
        for _ in range(6):
            existing = by_id.get(current["id"])
            if existing is None:
                by_id[current["id"]] = current
                break
            by_id.pop(current["id"], None)
            current = combine(existing, current)
            current["session_ids"] = sorted(set(current["session_ids"]))
            current["id"] = make_rule_id(current["category"], current["text"])
        else:
            by_id[current["id"]] = current

    out = list(by_id.values())
    out.sort(key=lambda r: (category_rank(r["category"]), r["id"]))
    return out
