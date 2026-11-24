"""OpenCode trajectories and correction pairs."""

from __future__ import annotations

from scripts.datasets.format import split_messages, to_tool_message
from scripts.datasets.sample import AGENTIC_SYSTEM, NO_ISSUES, Sample, _with_system
from scripts.ingest.opencode_sessions import load_jsonl, load_review_decisions, normalize_messages, session_id, source_model_info

def _assistant_after(messages: list[dict], start: int) -> int | None:
    for index in range(start, len(messages)):
        if messages[index].get("role") == "assistant":
            return index
    return None


def _corrections(messages: list[dict]) -> list[tuple[int, int, int]]:
    """Return (user_index, rejected_assistant, chosen_assistant) for follow-up corrections."""
    first_user = next((index for index, message in enumerate(messages) if message.get("role") == "user"), None)
    found = []
    for index, message in enumerate(messages):
        if message.get("role") != "user" or index == first_user:
            continue
        rejected = None
        for earlier in range(index - 1, -1, -1):
            if messages[earlier].get("role") == "assistant":
                rejected = earlier
                break
        chosen = _assistant_after(messages, index + 1)
        if rejected is None or chosen is None:
            continue
        found.append((index, rejected, chosen))
    return found


def _improved(messages: list[dict], pairs: list[tuple[int, int, int]]) -> list[dict]:
    drop = set()
    for user_index, rejected, _chosen in pairs:
        drop.add(user_index)
        drop.add(rejected)
    return [message for index, message in enumerate(messages) if index not in drop]


def agentic_samples(record: dict, *, machine: str, success: bool) -> list[Sample]:
    raw = [to_tool_message(message) for message in normalize_messages(record)]
    raw = [message for message in raw if message.get("content") or message.get("tool_calls")]
    if not raw:
        return []
    sid = session_id(record)
    pairs = _corrections(raw)
    samples = []
    if success and not pairs:
        samples.append(
            Sample(
                id=f"agentic-{sid}",
                task="agentic",
                source="opencode",
                machine=machine,
                group=f"session:{sid}",
                kind="trajectory",
                messages=_with_system(AGENTIC_SYSTEM, raw),
            )
        )
    if pairs:
        improved = _with_system(AGENTIC_SYSTEM, _improved(raw, pairs))
        primary = pairs[0]
        prompt = _with_system(AGENTIC_SYSTEM, raw[: primary[1]])
        dpo = {"prompt": prompt, "chosen": [raw[primary[2]]], "rejected": [raw[primary[1]]]}
        if any(message.get("role") == "assistant" for message in improved):
            samples.append(
                Sample(
                    id=f"agentic-{sid}-improved",
                    task="agentic",
                    source="opencode",
                    machine=machine,
                    group=f"session:{sid}",
                    kind="correction",
                    messages=improved,
                    dpo=dpo,
                    gold=raw[primary[2]].get("content") or "",
                    extra={"correction": raw[primary[0]].get("content") or ""},
                )
            )
        for nth, (user_index, rejected, chosen) in enumerate(pairs[1:], 2):
            step_prompt = _with_system(AGENTIC_SYSTEM, raw[:rejected])
            samples.append(
                Sample(
                    id=f"agentic-{sid}-dpo-{nth}",
                    task="agentic",
                    source="opencode",
                    machine=machine,
                    group=f"session:{sid}",
                    kind="correction-dpo",
                    messages=step_prompt + [raw[chosen]],
                    dpo={"prompt": step_prompt, "chosen": [raw[chosen]], "rejected": [raw[rejected]]},
                    gold=raw[chosen].get("content") or "",
                    extra={"correction": raw[user_index].get("content") or ""},
                )
            )
    return samples

def _success(record: dict, labels: dict[str, str], min_score: float) -> bool:
    sid = session_id(record)
    label = labels.get(sid, "")
    if label == "keep":
        return True
    if label == "discard":
        return False
    score = record.get("score")
    try:
        return score is not None and score != "" and float(score) >= min_score
    except (TypeError, ValueError):
        return False


def _machine_for(record: dict, machines: dict[str, str]) -> str:
    sid = session_id(record)
    if sid in machines:
        return machines[sid]
    return str(record.get("machine") or record.get("source") or "opencode")


def load_machine_labels(patterns_root) -> dict[str, str]:
    from pathlib import Path

    root = Path(patterns_root) if patterns_root else None
    if root is None or not (root / "store.db").exists():
        return {}
    from scripts.analysis.store import PatternStore

    store = PatternStore(root)
    try:
        labels = {}
        for row in store.sessions():
            sources = row.get("sources") or []
            machine = ",".join(sources) if sources else "opencode"
            for key in (row.get("id"), row.get("session_id"), row.get("global_id")):
                if key:
                    labels[str(key)] = machine
        return labels
    finally:
        store.close()


def from_opencode(path, decisions_path, *, machines: dict[str, str], min_score: float) -> list[Sample]:
    from pathlib import Path

    labels = load_review_decisions(Path(decisions_path))["labels"] if decisions_path else {}
    samples = []
    for record in load_jsonl(Path(path)):
        machine = _machine_for(record, machines)
        info = source_model_info(record)
        produced = agentic_samples(record, machine=machine, success=_success(record, labels, min_score))
        for sample in produced:
            sample.extra["source_model"] = info["label"]
            sample.extra["source_provider"] = info["provider"]
        samples.extend(produced)
    return samples
