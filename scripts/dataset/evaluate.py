"""Held-out review precision/recall and style similarity.

The OpenAI-compatible client generates a review or a rewrite. A second call,
or a lexical fallback, judges how close that is to the real comment or the
real rewritten code. Tests pass both functions in, so nothing calls the network.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from scripts.rules.parsing import extract_json, strip_thinking

_WORD = re.compile(r"[A-Za-z0-9_]+")


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text or "") if len(word) > 2}


def lexical_match(gold: str, prediction: str) -> bool:
    gold_words = _words(gold)
    pred_words = _words(prediction)
    if not gold_words or not pred_words:
        return gold.strip().lower() in prediction.strip().lower()
    overlap = len(gold_words & pred_words) / len(gold_words)
    return overlap >= 0.5


def style_ratio(gold: str, prediction: str) -> float:
    if not gold and not prediction:
        return 1.0
    return round(SequenceMatcher(None, gold or "", prediction or "").ratio(), 3)


def _user_text(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def _generate(complete_fn, messages: list[dict]) -> str:
    raw = complete_fn(messages)
    return strip_thinking(str(raw or "")).strip()


def _judge_review(judge_fn, gold: str, prediction: str) -> dict:
    if judge_fn is None:
        matched = lexical_match(gold, prediction)
        return {"matched": matched, "spurious": 0 if matched or not prediction.strip() else 1}
    raw = judge_fn(
        [
            {
                "role": "user",
                "content": (
                    "Compare a predicted review with the real review comment. "
                    'Return JSON {"matched": true or false, "spurious": 0 or 1}.\n'
                    f"Real: {gold}\nPredicted: {prediction}"
                ),
            }
        ]
    )
    try:
        payload = extract_json(strip_thinking(str(raw or "")))
    except Exception:
        matched = lexical_match(gold, prediction)
        return {"matched": matched, "spurious": 0 if matched else 1}
    return {"matched": bool(payload.get("matched")), "spurious": int(payload.get("spurious") or 0)}


def review_metrics(rows: list[dict], complete_fn, judge_fn=None) -> dict:
    """Precision and recall of predicted comments against real review comments."""
    gold_total = 0
    matched = 0
    spurious = 0
    details = []
    for row in rows:
        if row.get("task") != "review" or row.get("kind") == "review-negative":
            continue
        gold = ""
        for message in reversed(row.get("messages") or []):
            if message.get("role") == "assistant":
                gold = message.get("content") or ""
                break
        if not gold or gold.strip() == "No issues.":
            continue
        prompt = [message for message in row.get("messages") or [] if message.get("role") != "assistant"]
        prediction = _generate(complete_fn, prompt)
        verdict = _judge_review(judge_fn, gold, prediction)
        gold_total += 1
        if verdict["matched"]:
            matched += 1
        spurious += int(verdict.get("spurious") or 0)
        details.append({"id": row.get("id"), "matched": verdict["matched"]})
    precision = round(matched / (matched + spurious), 3) if matched + spurious else 0.0
    recall = round(matched / gold_total, 3) if gold_total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "gold": gold_total,
        "matched": matched,
        "details": details,
    }


def style_metrics(rows: list[dict], complete_fn, judge_fn=None) -> dict:
    """Similarity of a generated rewrite to the real after-code."""
    scores = []
    for row in rows:
        if row.get("task") != "style":
            continue
        messages = row.get("messages") or []
        gold = ""
        prompt = []
        for message in messages:
            if message.get("role") == "assistant":
                gold = message.get("content") or ""
            else:
                prompt.append(message)
        if not gold:
            continue
        prediction = _generate(complete_fn, prompt)
        score = style_ratio(gold, prediction)
        if judge_fn is not None:
            raw = judge_fn(
                [
                    {
                        "role": "user",
                        "content": (
                            "Score how well the rewrite matches the reference, from 0 to 1. "
                            'Return JSON {"score": 0.0}.\n'
                            f"Reference:\n{gold}\nRewrite:\n{prediction}"
                        ),
                    }
                ]
            )
            try:
                score = float(extract_json(strip_thinking(str(raw or ""))).get("score"))
            except Exception:
                score = style_ratio(gold, prediction)
        scores.append(score)
    mean = round(sum(scores) / len(scores), 3) if scores else 0.0
    return {"mean_similarity": mean, "samples": len(scores)}


def false_positive_rate(rows: list[dict], complete_fn) -> dict:
    """Share of clean hunks where the reviewer still writes a comment."""
    negatives = [row for row in rows if row.get("kind") == "review-negative"]
    flagged = 0
    for row in negatives:
        prompt = [message for message in row.get("messages") or [] if message.get("role") != "assistant"]
        prediction = _generate(complete_fn, prompt)
        if "no issues" not in prediction.lower():
            flagged += 1
    rate = round(flagged / len(negatives), 3) if negatives else 0.0
    return {"false_positive_rate": rate, "samples": len(negatives), "flagged": flagged}


def author_metrics(dpo_rows: list[dict], sft_rows: list[dict], complete_fn) -> dict:
    """Model A: how often a completion matches the uncorrected step, and how often its code draws a review comment."""
    failures = 0
    graded = 0
    for row in dpo_rows:
        if row.get("task") not in {"agentic", "style"}:
            continue
        chosen = ((row.get("chosen") or [{}])[-1].get("content") or "")
        rejected = ((row.get("rejected") or [{}])[-1].get("content") or "")
        if not chosen or not rejected:
            continue
        prediction = _generate(complete_fn, row.get("prompt") or [])
        graded += 1
        if style_ratio(rejected, prediction) > style_ratio(chosen, prediction):
            failures += 1
    flagged = 0
    reviewed = 0
    for row in sft_rows:
        if row.get("task") not in {"agentic", "style"}:
            continue
        code = ""
        for message in reversed(row.get("messages") or []):
            if message.get("role") == "assistant":
                code = message.get("content") or ""
                break
        if not code.strip():
            continue
        prediction = _generate(
            complete_fn,
            [{"role": "user", "content": "Review this code. If it is fine, reply exactly: No issues.\n\n" + code[:4000]}],
        )
        reviewed += 1
        if "no issues" not in prediction.lower():
            flagged += 1
    return {
        "user_correction_rate": round(failures / graded, 3) if graded else 0.0,
        "review_comment_rate": round(flagged / reviewed, 3) if reviewed else 0.0,
        "correction_samples": graded,
        "reviewed_samples": reviewed,
    }


def load_jsonl(path) -> list[dict]:
    rows = []
    text = path.read_text(encoding="utf-8") if hasattr(path, "read_text") else ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
