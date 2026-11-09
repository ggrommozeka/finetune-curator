"""Turn a recurring cluster into a system-prompt rule or an OpenCode skill."""

from __future__ import annotations

from pathlib import Path

from scripts.patterns.store import PatternStore
from scripts.rules.merge import similarity_scores
from scripts.rules.parsing import ModelOutputError, extract_json
from scripts.rules.store import upsert_proposals

RULE_TASK_SPREAD = 3

RECOMMEND_SYSTEM = """You recommend one harness change for a recurring coding-agent failure pattern.
Return ONLY JSON:
{"title":"...","description":"when to use this","text":"imperative rule or skill instructions","steps":["..."],"examples":["short quote"],"rationale":"..."}
If the pattern is specific to one task or tool, write skill steps. If it is general, write one system-prompt rule in text.
Do not copy secrets.
"""


def _success_conflict(store: PatternStore, error_type: str, session_ids: list[str]) -> str | None:
    own = set(session_ids)
    for row in store.annotated_rows():
        if row["global_id"] in own:
            continue
        annotation = row["annotation"]
        if annotation.get("outcome") == "success" and annotation.get("error_type") == error_type and error_type not in {"", "none"}:
            return f"error type {error_type} also appears in successful session {row['global_id']}"
    return None


def _rule_conflicts(text: str, existing_rules: list[str]) -> list[str]:
    found = []
    for rule in existing_rules:
        _jaccard, cover = similarity_scores(text, rule)
        if cover >= 0.6:
            found.append(f"overlaps existing rule: {rule[:120]}")
    return found


def recommend_cluster(cluster: dict, complete_fn, *, existing_rules: list[str], success_note: str | None) -> dict:
    spread = cluster.get("task_types") or 0
    kind = "rule" if spread >= RULE_TASK_SPREAD else "skill"
    user = (
        f"Pattern {cluster['id']} label={cluster.get('label')} "
        f"frequency={cluster.get('frequency')} task_types={cluster.get('task_type_names')} "
        f"projects={cluster.get('project_names')} machines={cluster.get('machine_names')} "
        f"fail_rate={cluster.get('fail_rate')}\n"
        "Choose a general system-prompt rule only when the pattern cuts across many task types. "
        "Otherwise draft a task-specific skill.\n/no_think"
    )
    messages = [
        {"role": "system", "content": RECOMMEND_SYSTEM},
        {"role": "user", "content": user},
    ]
    payload = {}
    last_error: Exception | None = None
    for _attempt in range(3):
        raw = complete_fn(messages)
        try:
            parsed = extract_json(raw)
            if isinstance(parsed, dict):
                payload = parsed
                break
        except ModelOutputError as exc:
            last_error = exc
            messages = messages + [
                {"role": "assistant", "content": (raw or "")[:2000]},
                {"role": "user", "content": "Return only the JSON object.\n/no_think"},
            ]
    else:
        raise ModelOutputError(str(last_error))
    text = str(payload.get("text") or payload.get("description") or cluster.get("label") or "").strip()
    conflicts = _rule_conflicts(text, existing_rules)
    if success_note:
        conflicts.append(success_note)
    return {
        "id": "p_" + cluster["id"],
        "cluster_id": cluster["id"],
        "kind": kind,
        "mode": "corpus",
        "text": text,
        "title": str(payload.get("title") or cluster.get("label") or "pattern").strip(),
        "description": str(payload.get("description") or "").strip(),
        "steps": [str(step) for step in (payload.get("steps") or [])],
        "examples": [str(example) for example in (payload.get("examples") or [])],
        "rationale": str(payload.get("rationale") or "").strip(),
        "category": "looping" if cluster.get("error_type") == "loop" else "other",
        "session_ids": cluster.get("session_ids") or [],
        "evidence": str(payload.get("description") or "").strip(),
        "conflicts": conflicts,
    }


def recommend_pending(
    store: PatternStore,
    complete_fn,
    *,
    proposals_path: Path,
    existing_rules: list[str] | None = None,
) -> dict:
    """Write corpus proposals into the shared proposals file, replacing per-session drafts."""
    existing_rules = existing_rules or []
    written = []
    for cluster in store.clusters(include_noise=False):
        note = _success_conflict(
            store,
            cluster.get("error_type") or cluster.get("label") or "",
            cluster.get("session_ids") or [],
        )
        written.append(
            recommend_cluster(cluster, complete_fn, existing_rules=existing_rules, success_note=note)
        )
    upsert_proposals(proposals_path, written, mode="corpus", drop_modes={"per-session"})
    return {"proposals": len(written)}
