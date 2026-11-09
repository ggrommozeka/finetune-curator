"""Proposal and rule-decision files under data/."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.rules.merge import slug_category

_STATUSES = {"approved", "rejected", "pending"}


def empty_rule_decisions() -> dict:
    return {"statuses": {}, "edits": {}}


def load_proposals(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "rules": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"version": 1, "rules": data}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    data.setdefault("rules", [])
    if not isinstance(data["rules"], list):
        raise ValueError(f"{path} field 'rules' must be a list")
    return data


def save_proposals(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_rule_decisions(path: Path) -> dict:
    if not path.exists():
        return empty_rule_decisions()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return empty_rule_decisions()
    if not isinstance(data, dict):
        return empty_rule_decisions()
    statuses_in = data.get("statuses") or {}
    edits_in = data.get("edits") or {}
    statuses: dict[str, str] = {}
    if isinstance(statuses_in, dict):
        for key, value in statuses_in.items():
            status = str(value).lower()
            if status in _STATUSES:
                statuses[str(key)] = status
    edits: dict[str, dict] = {}
    if isinstance(edits_in, dict):
        for key, value in edits_in.items():
            if not isinstance(value, dict):
                continue
            edit: dict[str, str] = {}
            for field in ("text", "category", "rationale"):
                if field in value and value[field] is not None:
                    edit[field] = str(value[field])
            edits[str(key)] = edit
    return {"statuses": statuses, "edits": edits}


def save_rule_decisions(path: Path, decisions: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "statuses": decisions.get("statuses") or {},
        "edits": decisions.get("edits") or {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def effective_rules(proposals: dict, decisions: dict | None) -> list[dict]:
    """Join proposals with review status and text edits. Ids stay stable."""
    decisions = decisions or empty_rule_decisions()
    statuses = decisions.get("statuses") or {}
    edits = decisions.get("edits") or {}
    view: list[dict] = []
    for rule in proposals.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        rid = str(rule.get("id") or "")
        if not rid:
            continue
        edit = edits.get(rid) or {}
        edited = str(edit["text"]).strip() if "text" in edit else ""
        text = edited or str(rule.get("text") or "").strip()
        if "category" in edit:
            category = slug_category(edit["category"])
        else:
            category = slug_category(str(rule.get("category") or "other"))
        if "rationale" in edit:
            rationale = edit["rationale"]
        else:
            rationale = str(rule.get("rationale") or "")
        session_ids = [str(s) for s in (rule.get("session_ids") or []) if str(s).strip()]
        view.append(
            {
                "id": rid,
                "text": text,
                "category": category,
                "rationale": rationale.strip(),
                "session_ids": session_ids,
                "evidence": str(rule.get("evidence") or "").strip(),
                "status": statuses.get(rid, "pending"),
            }
        )
    return view


def approved_rules(proposals: dict, decisions: dict | None) -> list[dict]:
    return [
        rule
        for rule in effective_rules(proposals, decisions)
        if rule["status"] == "approved" and rule["text"]
    ]


def public_rule(rule: dict) -> dict:
    return {
        "id": rule["id"],
        "text": rule["text"],
        "category": rule["category"],
        "rationale": rule.get("rationale") or "",
        "session_ids": list(rule.get("session_ids") or []),
        "evidence": rule.get("evidence") or "",
    }
