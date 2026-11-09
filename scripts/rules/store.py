"""Proposal and rule-decision files under data/."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.rules.merge import slug_category

_STATUSES = {"approved", "rejected", "pending"}


def empty_rule_decisions() -> dict:
    return {"statuses": {}, "edits": {}}


def _raw_items(data: dict) -> list:
    if isinstance(data.get("proposals"), list):
        return data["proposals"]
    rules = data.get("rules")
    return rules if isinstance(rules, list) else []


def normalize_proposal(item: dict) -> dict:
    """Shared proposal shape. Older files that only have rules are kind=rule."""
    kind = str(item.get("kind") or "rule").lower()
    if kind not in {"rule", "skill"}:
        kind = "rule"
    session_ids = item.get("session_ids") or []
    if isinstance(session_ids, str):
        session_ids = [session_ids]
    steps = item.get("steps") or []
    examples = item.get("examples") or []
    return {
        "id": str(item.get("id") or ""),
        "kind": kind,
        "text": str(item.get("text") or "").strip(),
        "category": slug_category(str(item.get("category") or "other")),
        "title": str(item.get("title") or "").strip(),
        "description": str(item.get("description") or "").strip(),
        "rationale": str(item.get("rationale") or "").strip(),
        "session_ids": [str(sid) for sid in session_ids if str(sid).strip()],
        "evidence": str(item.get("evidence") or "").strip(),
        "steps": [str(step) for step in steps if str(step).strip()],
        "examples": [str(example) for example in examples if str(example).strip()],
        "conflicts": [str(conflict) for conflict in (item.get("conflicts") or [])],
        "mode": str(item.get("mode") or ""),
        "cluster_id": str(item.get("cluster_id") or ""),
    }


def load_proposals(path: Path) -> dict:
    if not path.exists():
        return {"version": 2, "proposals": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"proposals": data}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    data["proposals"] = [normalize_proposal(item) for item in _raw_items(data) if isinstance(item, dict) and item.get("id")]
    data["rules"] = [item for item in data["proposals"] if item["kind"] == "rule"]
    data.setdefault("version", 2)
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
            for field in ("text", "category", "rationale", "kind", "title", "description"):
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


def effective_proposals(proposals: dict, decisions: dict | None) -> list[dict]:
    """Join proposals with review status and text edits. Ids stay stable.

    This is the shared model for per-session suggestions and later corpus mining.
    ``kind`` is ``rule`` or ``skill``.
    """
    decisions = decisions or empty_rule_decisions()
    statuses = decisions.get("statuses") or {}
    edits = decisions.get("edits") or {}
    view: list[dict] = []
    for proposal in proposals.get("proposals") or proposals.get("rules") or []:
        if not isinstance(proposal, dict):
            continue
        base = normalize_proposal(proposal)
        rid = base["id"]
        if not rid:
            continue
        edit = edits.get(rid) or {}
        if "text" in edit and str(edit["text"]).strip():
            base["text"] = str(edit["text"]).strip()
        if "category" in edit:
            base["category"] = slug_category(edit["category"])
        if "rationale" in edit:
            base["rationale"] = str(edit["rationale"]).strip()
        if "title" in edit:
            base["title"] = str(edit["title"]).strip()
        if "description" in edit:
            base["description"] = str(edit["description"]).strip()
        if str(edit.get("kind") or "").lower() in {"rule", "skill"}:
            base["kind"] = str(edit["kind"]).lower()
        base["status"] = statuses.get(rid, "pending")
        view.append(base)
    return view


def effective_rules(proposals: dict, decisions: dict | None) -> list[dict]:
    """Rule-kind proposals. Skills stay in ``effective_proposals``."""
    return [item for item in effective_proposals(proposals, decisions) if item["kind"] == "rule"]


def approved_proposals(proposals: dict, decisions: dict | None, kind: str | None = None) -> list[dict]:
    found = [
        item
        for item in effective_proposals(proposals, decisions)
        if item["status"] == "approved" and item["text"]
    ]
    if kind:
        found = [item for item in found if item["kind"] == kind]
    return found


def approved_rules(proposals: dict, decisions: dict | None) -> list[dict]:
    return approved_proposals(proposals, decisions, kind="rule")


def public_rule(rule: dict) -> dict:
    return {
        "id": rule["id"],
        "kind": rule.get("kind") or "rule",
        "text": rule["text"],
        "category": rule["category"],
        "rationale": rule.get("rationale") or "",
        "session_ids": list(rule.get("session_ids") or []),
        "evidence": rule.get("evidence") or "",
    }


def upsert_proposals(path: Path, new_items: list[dict], *, mode: str | None = None, drop_modes: set[str] | None = None) -> dict:
    """Merge proposals into the shared file by id. Decisions are left untouched.

    ``drop_modes`` removes earlier proposals (for example per-session drafts)
    when a corpus run replaces them.
    """
    document = load_proposals(path) if path.exists() else {"version": 2, "proposals": []}
    drop_modes = drop_modes or set()
    by_id = {
        item["id"]: item
        for item in document.get("proposals") or []
        if item.get("mode") not in drop_modes
    }
    for item in new_items:
        proposal = normalize_proposal(item)
        if mode:
            proposal["mode"] = mode
        if not proposal["id"]:
            continue
        by_id[proposal["id"]] = proposal
    merged = {
        "version": 2,
        "mode": mode or document.get("mode"),
        "proposals": list(by_id.values()),
    }
    for key in ("model", "base_url", "context_budget", "source_session_ids", "batches", "errors"):
        if key in document:
            merged[key] = document[key]
    save_proposals(path, merged)
    return load_proposals(path)
