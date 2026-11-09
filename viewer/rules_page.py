"""Review proposed harness rules: approve, reject, edit, jump back to the session."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rules.merge import CATEGORIES  # noqa: E402
from scripts.rules.sessions import (  # noqa: E402
    first_user_preview,
    load_jsonl,
    load_review_decisions,
    normalize_messages,
    session_id,
)
from scripts.rules.store import (  # noqa: E402
    effective_proposals,
    load_proposals,
    load_rule_decisions,
    save_rule_decisions,
)


def _esc(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _widget_value(key: str, default):
    if key in st.session_state:
        return {}
    return {"value": default}


def _status_pill(status: str) -> str:
    css = {
        "approved": "pill-approved",
        "rejected": "pill-rejected",
        "pending": "pill-pending",
    }.get(status, "pill-pending")
    label = {"approved": "approved", "rejected": "rejected", "pending": "pending"}.get(status, status)
    return f'<span class="pill {css}">{label}</span>'


def render_rules_page(cli) -> None:
    with st.sidebar:
        proposals_path = st.text_input(
            "Proposals",
            key="rule_proposals_path",
            **_widget_value("rule_proposals_path", cli.rules),
        )
        rule_decisions_path = st.text_input(
            "Rule decisions",
            key="rule_decisions_path",
            **_widget_value("rule_decisions_path", cli.rule_decisions),
        )
        input_path = st.text_input(
            "JSONL",
            key="input_path",
            **_widget_value("input_path", cli.input),
        )
        decisions_path = st.text_input(
            "Decisions",
            key="decisions_path",
            **_widget_value("decisions_path", cli.decisions),
        )
        reload = st.button("↻ Reload rules", use_container_width=True)

    proposals_p = Path(proposals_path)
    rule_decisions_p = Path(rule_decisions_path)

    if reload:
        for key in list(st.session_state.keys()):
            if str(key).startswith("rule_text_") or str(key).startswith("rule_cat_"):
                del st.session_state[key]

    if "rule_doc" not in st.session_state or reload:
        if proposals_p.exists():
            try:
                st.session_state.rule_doc = load_proposals(proposals_p)
                st.session_state.rule_doc_error = ""
            except (OSError, ValueError) as exc:
                st.session_state.rule_doc = {"version": 1, "rules": []}
                st.session_state.rule_doc_error = str(exc)
        else:
            st.session_state.rule_doc = {"version": 1, "rules": []}
            st.session_state.rule_doc_error = ""
        st.session_state.rule_decisions = load_rule_decisions(rule_decisions_p)

    document: dict = st.session_state.rule_doc
    rule_decisions: dict = st.session_state.rule_decisions
    rows = load_jsonl(Path(input_path))
    review = load_review_decisions(Path(decisions_path))
    notes = review.get("notes") or {}
    by_id = {session_id(rec): rec for rec in rows}
    rules = effective_proposals(document, rule_decisions)

    with st.sidebar:
        st.markdown("---")
        status_kwargs = (
            {}
            if "rule_status_filter" in st.session_state
            else {"default": ["pending", "approved", "rejected"]}
        )
        status_filter = st.multiselect(
            "Статус правил",
            ["pending", "approved", "rejected"],
            key="rule_status_filter",
            **status_kwargs,
        )
        n_ok = sum(1 for rule in rules if rule["status"] == "approved")
        n_no = sum(1 for rule in rules if rule["status"] == "rejected")
        n_todo = sum(1 for rule in rules if rule["status"] == "pending")
        c1, c2, c3 = st.columns(3)
        c1.metric("Yes", n_ok)
        c2.metric("No", n_no)
        c3.metric("Todo", n_todo)
        st.caption(f"Правил: **{len(rules)}**")
        st.markdown("---")
        st.markdown(
            f"""```bash
python -m scripts.export_rules \\
  --proposals {proposals_path} \\
  --decisions {rule_decisions_path} \\
  --markdown data/approved_rules.md \\
  --json data/approved_rules.json
```"""
        )

    if st.session_state.get("rule_doc_error"):
        st.error(st.session_state.rule_doc_error)

    if not proposals_p.exists() or not (document.get("rules") or []):
        st.markdown(
            """
            <div class="hero">
              <p class="hero-title">Нет предложенных правил</p>
              <p class="hero-sub">Сначала прогоните propose_rules по размеченным сессиям.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(
            "python -m scripts.propose_rules \\\n"
            f"  --input {input_path} \\\n"
            f"  --decisions {decisions_path} \\\n"
            f"  --output {proposals_path}",
            language="bash",
        )
        return

    model = document.get("model") or "—"
    batches = document.get("batches", "—")
    budget = document.get("context_budget", "—")
    st.markdown(
        f"""
        <div class="hero">
          <p class="hero-title">Правила для system prompt</p>
          <p class="hero-sub">model { _esc(str(model)) } · batches {batches} · budget {budget}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for error in document.get("errors") or []:
        st.warning(error)

    visible = [rule for rule in rules if rule["status"] in status_filter]
    if not visible:
        st.info("Нет правил под текущий фильтр.")
        return

    for rule in visible:
        rid = rule["id"]
        text_key = f"rule_text_{rid}"
        cat_key = f"rule_cat_{rid}"
        options = list(CATEGORIES)
        if rule["category"] not in options:
            options = [rule["category"], *options]
        if text_key not in st.session_state:
            st.session_state[text_key] = rule["text"]
        if cat_key not in st.session_state or st.session_state[cat_key] not in options:
            st.session_state[cat_key] = rule["category"] if rule["category"] in options else options[0]

        st.markdown(
            f"""
            <div class="hero" style="margin-bottom:8px">
              <p class="hero-title"><code>{_esc(rid)}</code></p>
              <div style="margin-top:8px">{_status_pill(rule["status"])}
                <span class="pill pill-score">{_esc(rule["category"])}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        kind_key = f"rule_kind_{rid}"
        if kind_key not in st.session_state:
            st.session_state[kind_key] = rule.get("kind") or "rule"
        kind = st.selectbox("Тип", ["rule", "skill"], key=kind_key)
        text = st.text_area("Текст", key=text_key, height=90)
        category = st.selectbox("Категория", options, key=cat_key)
        st.markdown("**Почему**")
        st.text(rule["rationale"] or "—")
        st.markdown("**Фрагмент**")
        st.text(rule["evidence"] or "—")

        sids = rule["session_ids"] or []
        if sids:
            st.markdown("**Сессии-источники**")
            columns = st.columns(min(4, len(sids)))
            for index, sid in enumerate(sids):
                with columns[index % len(columns)]:
                    if st.button(sid, key=f"open_{rid}_{sid}", use_container_width=True):
                        st.session_state.pending_page = "Сессии"
                        st.session_state.jump_sid = sid
                        st.rerun()
                    rec = by_id.get(sid)
                    if rec is None:
                        st.caption("нет в JSONL")
                    else:
                        preview = first_user_preview(normalize_messages(rec), 80)
                        note = (notes.get(sid) or "").strip()
                        st.caption(preview)
                        if note:
                            st.caption("заметка: " + note[:160])

        b1, b2, b3 = st.columns(3)
        if b1.button("✅ Одобрить", key=f"approve_{rid}", type="primary", use_container_width=True):
            if not str(text).strip():
                st.warning("Пустой текст правила.")
            else:
                _store_status(rule_decisions, rule_decisions_p, rid, "approved", text, category, kind)
                st.rerun()
        if b2.button("❌ Отклонить", key=f"reject_{rid}", use_container_width=True):
            _store_status(rule_decisions, rule_decisions_p, rid, "rejected", text, category, kind)
            st.rerun()
        if b3.button("Сбросить", key=f"clear_{rid}", use_container_width=True):
            rule_decisions.setdefault("statuses", {}).pop(rid, None)
            _store_edit(rule_decisions, rid, text, category, kind)
            save_rule_decisions(rule_decisions_p, rule_decisions)
            st.session_state.rule_decisions = rule_decisions
            st.rerun()
        st.markdown("---")


def _store_edit(decisions: dict, rid: str, text: str, category: str, kind: str) -> None:
    edit = {"category": category, "kind": kind if kind in {"rule", "skill"} else "rule"}
    cleaned = str(text).strip()
    if cleaned:
        edit["text"] = cleaned
    decisions.setdefault("edits", {})[rid] = edit


def _store_status(decisions: dict, path: Path, rid: str, status: str, text: str, category: str, kind: str) -> None:
    decisions.setdefault("statuses", {})[rid] = status
    _store_edit(decisions, rid, text, category, kind)
    save_rule_decisions(path, decisions)
    st.session_state.rule_decisions = decisions
