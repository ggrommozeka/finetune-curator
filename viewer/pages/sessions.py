"""OpenCode session review: chat view, filters, keep or discard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ingest.opencode_sessions import (
    first_user_preview,
    load_jsonl,
    load_review_decisions,
    normalize_messages,
    session_id,
    source_model_info,
)
from shared import esc, label_pill, queue_session_pos, render_message, save_json, status_pill, widget_value


def render() -> None:
    with st.sidebar:
        st.markdown("### ◈ Сессии")
        input_path = st.text_input(
            "JSONL",
            key="input_path",
            **widget_value("input_path", st.session_state.get("cfg_input", "data/raw_scored.jsonl")),
        )
        decisions_path = st.text_input(
            "Decisions",
            key="decisions_path",
            **widget_value("decisions_path", st.session_state.get("cfg_decisions", "data/decisions.json")),
        )
        reload = st.button("↻ Reload", use_container_width=True)

    input_p = Path(input_path)
    decisions_p = Path(decisions_path)
    if "rows" not in st.session_state or reload:
        st.session_state.rows = load_jsonl(input_p)
        st.session_state.decisions = load_review_decisions(decisions_p)
        queue_session_pos(0)

    rows = st.session_state.rows
    decisions = st.session_state.decisions
    labels = decisions.setdefault("labels", {})
    notes = decisions.setdefault("notes", {})
    if not rows:
        st.markdown(
            f"""<div class="hero"><p class="hero-title">Нет данных</p>
            <p class="hero-sub">Файл <code>{esc(str(input_p))}</code> пуст или не найден.</p></div>""",
            unsafe_allow_html=True,
        )
        st.code("streamlit run viewer/app.py", language="bash")
        return

    with st.sidebar:
        st.markdown("**Фильтры**")
        jump = st.session_state.get("jump_sid")
        if jump:
            st.session_state.min_score = 0.0
            st.session_state.status_filter = ["undecided", "keep", "discard"]
            st.session_state.only_tools = False
            st.session_state.query = ""
            st.session_state.label_filter = sorted({str(row.get("label", "unknown")) for row in rows})
        min_score = st.slider("Min score", min_value=0.0, max_value=1.0, step=0.05, key="min_score", **widget_value("min_score", 0.0))
        all_labels = sorted({str(row.get("label", "unknown")) for row in rows})
        label_kwargs = {} if "label_filter" in st.session_state else {"default": all_labels}
        label_filter = st.multiselect("Label", all_labels, key="label_filter", **label_kwargs)
        status_kwargs = {} if "status_filter" in st.session_state else {"default": ["undecided", "keep", "discard"]}
        status_filter = st.multiselect("Статус", ["undecided", "keep", "discard"], key="status_filter", **status_kwargs)
        only_tools = st.checkbox("Только с tool calls", key="only_tools", **widget_value("only_tools", False))
        query = st.text_input("Поиск в user-тексте", key="query", **widget_value("query", ""))
        model_query = st.text_input("Модель-источник", key="source_model", **widget_value("source_model", ""))

        filtered = []
        for row in rows:
            score = row.get("score")
            if score is not None and score != "" and float(score) < min_score:
                continue
            if str(row.get("label", "unknown")) not in label_filter:
                continue
            if labels.get(session_id(row), "undecided") not in status_filter:
                continue
            messages = normalize_messages(row)
            if only_tools and not any(message.get("tool_calls") for message in messages if message.get("role") == "assistant"):
                continue
            if query and query.lower() not in first_user_preview(messages, 500).lower():
                continue
            if model_query and model_query.lower() not in source_model_info(row)["label"].lower():
                continue
            filtered.append(row)

        kept = sum(1 for value in labels.values() if value == "keep")
        discarded = sum(1 for value in labels.values() if value == "discard")
        c1, c2, c3 = st.columns(3)
        c1.metric("Keep", kept)
        c2.metric("Discard", discarded)
        c3.metric("Todo", max(0, len(rows) - kept - discarded))
        st.caption(f"В фильтре: **{len(filtered)}** / {len(rows)}")

    if not filtered:
        missing = st.session_state.pop("jump_sid", None)
        if missing:
            st.warning(f"Сессия {missing} не найдена в текущем JSONL.")
        st.info("Нет сессий под текущие фильтры.")
        return

    jump = st.session_state.get("jump_sid")
    if jump:
        for position, candidate in enumerate(filtered):
            if session_id(candidate) == jump:
                queue_session_pos(position)
                break
        else:
            st.warning(f"Сессия {jump} не попала в фильтр.")
        st.session_state.jump_sid = None

    idx = max(0, min(int(st.session_state.get("idx", 0)), len(filtered) - 1))
    st.session_state.idx = idx
    record = filtered[idx]
    sid = session_id(record)
    messages = normalize_messages(record)
    current = labels.get(sid, "undecided")
    st.markdown(
        f"""<div class="hero"><p class="hero-title"><code>{esc(sid)}</code></p>
        <p class="hero-sub">{esc(first_user_preview(messages, 140))}</p>
        <div style="margin-top:10px"><span class="pill pill-score">score {esc(str(record.get('score', '—')))}</span>
        {label_pill(str(record.get('label', '—')))} {status_pill(current)}
        <span class="pill pill-score">opencode</span>
        <span class="pill pill-score">{esc(source_model_info(record)["label"] or "model unknown")}</span></div></div>""",
        unsafe_allow_html=True,
    )
    metrics = record.get("metrics") or {}
    columns = st.columns(5)
    columns[0].metric("Messages", metrics.get("total_messages", len(messages)))
    columns[1].metric("Tokens~", metrics.get("total_tokens", "—"))
    columns[2].metric("Tool calls", metrics.get("tool_calls", "—"))
    tools = record.get("tools_used") or []
    columns[3].metric("Tools", len(tools) if tools else "—")
    columns[4].metric("#", f"{idx + 1}/{len(filtered)}")
    if tools:
        st.caption("tools: " + ", ".join(str(tool) for tool in tools))
    if record.get("reasons"):
        st.caption("reasons: " + " · ".join(str(reason) for reason in record["reasons"]))

    nav = st.columns([1, 2, 1])
    if "pending_session_pos" in st.session_state:
        st.session_state.session_pos = int(st.session_state.pop("pending_session_pos"))
    if "session_pos" in st.session_state:
        st.session_state.session_pos = max(1, min(int(st.session_state.session_pos), len(filtered)))
    with nav[0]:
        if st.button("← Prev", use_container_width=True) and idx > 0:
            queue_session_pos(idx - 1)
            st.rerun()
    with nav[1]:
        position = st.slider(
            "Сессия в фильтре",
            min_value=1,
            max_value=len(filtered),
            key="session_pos",
            label_visibility="collapsed",
            **widget_value("session_pos", idx + 1),
        )
        if position - 1 != idx:
            st.session_state.idx = position - 1
            st.rerun()
    with nav[2]:
        if st.button("Next →", use_container_width=True) and idx < len(filtered) - 1:
            queue_session_pos(idx + 1)
            st.rerun()

    st.markdown("#### Диалог")
    for index, message in enumerate(messages, 1):
        render_message(index, message)
    note_val = st.text_area(
        "Заметка к сессии",
        value=notes.get(sid, ""),
        height=80,
        key=f"note_{sid}",
    )
    if note_val != notes.get(sid, ""):
        notes[sid] = note_val
        st.session_state.decisions = decisions

    st.markdown('<div class="action-bar">', unsafe_allow_html=True)
    actions = st.columns(4)
    if actions[0].button("✅ Keep", type="primary", use_container_width=True):
        labels[sid] = "keep"
        notes[sid] = note_val
        save_json(decisions_p, decisions)
        if idx < len(filtered) - 1:
            queue_session_pos(idx + 1)
        st.rerun()
    if actions[1].button("❌ Discard", use_container_width=True):
        labels[sid] = "discard"
        notes[sid] = note_val
        save_json(decisions_p, decisions)
        if idx < len(filtered) - 1:
            queue_session_pos(idx + 1)
        st.rerun()
    if actions[2].button("Clear", use_container_width=True):
        labels.pop(sid, None)
        save_json(decisions_p, decisions)
        st.rerun()
    if actions[3].button("💾 Save", use_container_width=True):
        notes[sid] = note_val
        save_json(decisions_p, decisions)
        st.toast("Saved")
    st.markdown("</div>", unsafe_allow_html=True)


render()
