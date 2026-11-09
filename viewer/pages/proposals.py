"""One approval page for rule and skill proposals from any source."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.catalog import Catalog
from scripts.rules.merge import CATEGORIES
from scripts.rules.sessions import first_user_preview, load_jsonl, load_review_decisions, normalize_messages, session_id
from scripts.rules.store import effective_proposals, load_proposals, load_rule_decisions, save_rule_decisions
from shared import esc, open_evidence, status_pill, widget_value


def render() -> None:
    with st.sidebar:
        st.markdown("### ◈ Правила")
        proposals_path = st.text_input(
            "Proposals",
            key="rule_proposals_path",
            **widget_value("rule_proposals_path", st.session_state.get("cfg_rules", "data/rule_proposals.json")),
        )
        decisions_path = st.text_input(
            "Rule decisions",
            key="rule_decisions_path",
            **widget_value("rule_decisions_path", st.session_state.get("cfg_rule_decisions", "data/rule_decisions.json")),
        )
        input_path = st.text_input(
            "JSONL",
            key="input_path",
            **widget_value("input_path", st.session_state.get("cfg_input", "data/raw_scored.jsonl")),
        )
        reload = st.button("↻ Reload", use_container_width=True)

    proposals_p = Path(proposals_path)
    decisions_p = Path(decisions_path)
    if reload:
        for key in list(st.session_state.keys()):
            if str(key).startswith(("rule_text_", "rule_cat_", "rule_kind_")):
                del st.session_state[key]
    if "rule_doc" not in st.session_state or reload:
        st.session_state.rule_doc = load_proposals(proposals_p) if proposals_p.exists() else {"version": 2, "proposals": []}
        st.session_state.rule_decisions = load_rule_decisions(decisions_p)

    document = st.session_state.rule_doc
    rule_decisions = st.session_state.rule_decisions
    proposals = effective_proposals(document, rule_decisions)
    catalog = Catalog(input_path, st.session_state.get("cfg_gitlab", "data/gitlab"))
    rows = catalog.opencode_records()
    notes = (load_review_decisions(Path(st.session_state.get("cfg_decisions", "data/decisions.json"))).get("notes") or {})
    by_id = {session_id(row): row for row in rows}

    with st.sidebar:
        statuses = st.multiselect(
            "Статус",
            ["pending", "approved", "rejected"],
            key="rule_status_filter",
            **({} if "rule_status_filter" in st.session_state else {"default": ["pending", "approved", "rejected"]}),
        )
        kinds = st.multiselect(
            "Тип",
            ["rule", "skill"],
            key="rule_kind_filter",
            **({} if "rule_kind_filter" in st.session_state else {"default": ["rule", "skill"]}),
        )
        known_sources = sorted({item.get("source") or "opencode" for item in proposals} or {"opencode"})
        sources = st.multiselect(
            "Источник",
            known_sources,
            key="rule_source_filter",
            **({} if "rule_source_filter" in st.session_state else {"default": known_sources}),
        )
        st.caption(f"Предложений: **{len(proposals)}**")

    if not proposals:
        st.markdown(
            """<div class="hero"><p class="hero-title">Нет предложений</p>
            <p class="hero-sub">Per-session propose_rules пишет сюда. Позже сюда же пишут GitLab и pattern mining.</p></div>""",
            unsafe_allow_html=True,
        )
        st.code("python -m scripts.propose_rules --input data/raw_scored.jsonl", language="bash")
        return

    visible = [
        item
        for item in proposals
        if item["status"] in statuses
        and item["kind"] in kinds
        and (item.get("source") or "opencode") in sources
    ]
    st.markdown(
        f"""<div class="hero"><p class="hero-title">Правила и skills</p>
        <p class="hero-sub">{len(visible)} из {len(proposals)}. Один список для всех источников.</p></div>""",
        unsafe_allow_html=True,
    )
    if not visible:
        st.info("Нет предложений под фильтр.")
        return

    for item in visible:
        rid = item["id"]
        text_key, cat_key, kind_key = f"rule_text_{rid}", f"rule_cat_{rid}", f"rule_kind_{rid}"
        options = list(CATEGORIES)
        if item["category"] not in options:
            options = [item["category"], *options]
        if text_key not in st.session_state:
            st.session_state[text_key] = item["text"]
        if kind_key not in st.session_state:
            st.session_state[kind_key] = item["kind"]
        if cat_key not in st.session_state or st.session_state[cat_key] not in options:
            st.session_state[cat_key] = item["category"] if item["category"] in options else options[0]
        source = item.get("source") or "opencode"
        st.markdown(
            f"""<div class="hero" style="margin-bottom:8px"><p class="hero-title"><code>{esc(rid)}</code></p>
            <div style="margin-top:8px">{status_pill(item["status"])}
            <span class="pill pill-score">{esc(source)}</span>
            <span class="pill pill-score">{esc(item["category"])}</span></div></div>""",
            unsafe_allow_html=True,
        )
        kind = st.selectbox("Тип", ["rule", "skill"], key=kind_key)
        text = st.text_area("Текст", key=text_key, height=90)
        category = st.selectbox("Категория", options, key=cat_key)
        if item.get("conflicts"):
            st.caption("Конфликты: " + " · ".join(item["conflicts"]))
        st.text(item.get("rationale") or "—")
        for ref in item.get("session_ids") or []:
            found = catalog.resolve(ref)
            label = ref if found is None else f"{found.source}: {ref}"
            if st.button(label, key=f"open_{rid}_{ref}"):
                if found is None:
                    st.warning(f"Источник для {ref} не найден в JSONL или GitLab store.")
                else:
                    open_evidence(ref)
            preview_row = by_id.get(ref)
            if preview_row is not None:
                st.caption(first_user_preview(normalize_messages(preview_row), 80))
                if notes.get(ref):
                    st.caption("заметка: " + notes[ref][:160])
        buttons = st.columns(3)
        if buttons[0].button("✅ Одобрить", key=f"approve_{rid}", type="primary"):
            _save(rule_decisions, decisions_p, rid, "approved", text, category, kind)
            st.rerun()
        if buttons[1].button("❌ Отклонить", key=f"reject_{rid}"):
            _save(rule_decisions, decisions_p, rid, "rejected", text, category, kind)
            st.rerun()
        if buttons[2].button("Сбросить", key=f"clear_{rid}"):
            rule_decisions.setdefault("statuses", {}).pop(rid, None)
            _save(rule_decisions, decisions_p, rid, rule_decisions["statuses"].get(rid, "pending"), text, category, kind)
            rule_decisions["statuses"].pop(rid, None)
            save_rule_decisions(decisions_p, rule_decisions)
            st.rerun()


def _save(decisions, path, rid, status, text, category, kind) -> None:
    decisions.setdefault("statuses", {})[rid] = status
    edit = {"category": category, "kind": kind if kind in {"rule", "skill"} else "rule"}
    if str(text).strip():
        edit["text"] = str(text).strip()
    decisions.setdefault("edits", {})[rid] = edit
    save_rule_decisions(path, decisions)
    st.session_state.rule_decisions = decisions


render()
