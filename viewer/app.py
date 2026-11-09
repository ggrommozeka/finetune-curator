#!/usr/bin/env python3
"""Streamlit reviewer: Keep / Discard agent trajectories, export-ready decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import streamlit as st

DEFAULT_INPUT = "data/raw_scored.jsonl"
DEFAULT_DECISIONS = "data/decisions.json"


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    args, _ = parser.parse_known_args()
    return args


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_decisions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v).lower() for k, v in data.items()}
    except json.JSONDecodeError:
        pass
    return {}


def save_decisions(path: Path, decisions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")


def session_id(rec: dict) -> str:
    return str(rec.get("id") or rec.get("session_id") or f"idx-{hash(json.dumps(rec, sort_keys=True)) % 10**8}")


def normalize_messages(rec: dict) -> list[dict]:
    msgs = rec.get("messages") or []
    if not msgs:
        return []
    if isinstance(msgs[0], str):
        out = []
        for m in msgs:
            try:
                out.append(json.loads(m))
            except json.JSONDecodeError:
                out.append({"role": "unknown", "content": m})
        return out
    return msgs


def main() -> None:
    cli = parse_cli_args()
    st.set_page_config(page_title="LogMiner Curator", layout="wide")
    st.title("LogMiner Curator")
    st.caption("Keep / Discard trajectories · export only approved for Unsloth")

    with st.sidebar:
        st.header("Source")
        input_path = st.text_input("JSONL path", cli.input)
        decisions_path = st.text_input("Decisions path", cli.decisions)
        reload = st.button("Reload files")

    input_p = Path(input_path)
    decisions_p = Path(decisions_path)

    if "rows" not in st.session_state or reload:
        st.session_state.rows = load_jsonl(input_p)
        st.session_state.decisions = load_decisions(decisions_p)
        st.session_state.idx = 0

    rows: list[dict] = st.session_state.rows
    decisions: dict[str, str] = st.session_state.decisions

    if not rows:
        st.warning(
            f"No records in `{input_p}`.\n\n"
            "Run export first:\n"
            "`python -m scripts.export_opencode --output data/raw_scored.jsonl --min-score 0`"
        )
        st.stop()

    with st.sidebar:
        st.header("Filters")
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.05)
        labels = sorted({r.get("label", "unknown") for r in rows})
        label_filter = st.multiselect("Label", labels, default=labels)
        status_filter = st.multiselect(
            "Review status",
            ["undecided", "keep", "discard"],
            default=["undecided", "keep", "discard"],
        )

        def status_of(rec: dict) -> str:
            return decisions.get(session_id(rec), "undecided")

        filtered = []
        for r in rows:
            if float(r.get("score") or 0) < min_score:
                continue
            if r.get("label", "unknown") not in label_filter:
                continue
            if status_of(r) not in status_filter:
                continue
            filtered.append(r)

        st.write(f"**{len(filtered)}** / {len(rows)} sessions")
        n_keep = sum(1 for v in decisions.values() if v == "keep")
        n_disc = sum(1 for v in decisions.values() if v == "discard")
        st.metric("Keep", n_keep)
        st.metric("Discard", n_disc)
        st.metric("Undecided", max(0, len(rows) - n_keep - n_disc))

        if st.button("Save decisions now"):
            save_decisions(decisions_p, decisions)
            st.success(f"Saved → {decisions_p}")

        st.markdown("---")
        st.markdown(
            f"Export approved:\n```bash\npython -m scripts.export_approved \\\n"
            f"  --input {input_path} \\\n"
            f"  --decisions {decisions_path} \\\n"
            f"  --output data/approved.jsonl\n```"
        )

    if not filtered:
        st.info("No sessions match current filters.")
        st.stop()

    # Keep index in range
    idx = int(st.session_state.get("idx", 0))
    idx = max(0, min(idx, len(filtered) - 1))
    st.session_state.idx = idx

    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("← Prev") and idx > 0:
            st.session_state.idx = idx - 1
            st.rerun()
    with col_nav2:
        st.write(f"Session **{idx + 1}** / {len(filtered)}")
    with col_nav3:
        if st.button("Next →") and idx < len(filtered) - 1:
            st.session_state.idx = idx + 1
            st.rerun()

    rec = filtered[idx]
    sid = session_id(rec)
    current = decisions.get(sid, "undecided")

    st.subheader(f"`{sid}`")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Score", rec.get("score", "—"))
    m2.metric("Label", rec.get("label", "—"))
    m3.metric("Status", current)
    metrics = rec.get("metrics") or {}
    m4.metric("Tool calls", metrics.get("tool_calls", rec.get("tool_count", "—")))
    m5.metric("Tokens~", metrics.get("total_tokens", "—"))

    reasons = rec.get("reasons") or []
    if reasons:
        st.caption("reasons: " + " · ".join(str(x) for x in reasons))
    tools = rec.get("tools_used") or []
    if tools:
        st.caption("tools: " + ", ".join(tools))

    k1, k2, k3 = st.columns(3)
    with k1:
        if st.button("✅ Keep", use_container_width=True, type="primary"):
            decisions[sid] = "keep"
            st.session_state.decisions = decisions
            save_decisions(decisions_p, decisions)
            if idx < len(filtered) - 1:
                st.session_state.idx = idx + 1
            st.rerun()
    with k2:
        if st.button("❌ Discard", use_container_width=True):
            decisions[sid] = "discard"
            st.session_state.decisions = decisions
            save_decisions(decisions_p, decisions)
            if idx < len(filtered) - 1:
                st.session_state.idx = idx + 1
            st.rerun()
    with k3:
        if st.button("Clear decision", use_container_width=True):
            decisions.pop(sid, None)
            st.session_state.decisions = decisions
            save_decisions(decisions_p, decisions)
            st.rerun()

    st.markdown("---")
    msgs = normalize_messages(rec)
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        expanded = role == "user" and i < 4
        with st.expander(f"{i + 1}. **{role}**", expanded=expanded):
            if role == "assistant" and m.get("tool_calls"):
                st.markdown("**tool_calls**")
                st.json(m["tool_calls"])
            content = m.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, indent=2)
            if content:
                # Truncate very long tool dumps in UI
                shown = content if len(content) <= 8000 else content[:8000] + "\n… [truncated]"
                st.code(shown, language="markdown")
            if role == "tool":
                st.caption(f"tool_call_id: {m.get('tool_call_id', '')}")


if __name__ == "__main__":
    main()
