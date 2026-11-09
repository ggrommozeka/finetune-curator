"""Export training data, rules, and skills from the shared stores."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_approved import main as export_sessions
from scripts.export_gitlab_mrr import main as export_gitlab
from scripts.export_rules import main as export_rules
from shared import widget_value


def render() -> None:
    st.markdown(
        """<div class="hero"><p class="hero-title">Экспорт</p>
        <p class="hero-sub">Один запуск приложения. Датасет сессий, правила и skills пишутся из тех же файлов, что и страницы.</p></div>""",
        unsafe_allow_html=True,
    )
    input_path = st.text_input("JSONL", key="input_path", **widget_value("input_path", st.session_state.get("cfg_input")))
    decisions = st.text_input("Decisions", key="decisions_path", **widget_value("decisions_path", st.session_state.get("cfg_decisions")))
    approved = st.text_input("Approved JSONL", value="data/approved.jsonl")
    proposals = st.text_input("Proposals", key="rule_proposals_path", **widget_value("rule_proposals_path", st.session_state.get("cfg_rules")))
    rule_decisions = st.text_input(
        "Rule decisions",
        key="rule_decisions_path",
        **widget_value("rule_decisions_path", st.session_state.get("cfg_rule_decisions")),
    )
    markdown = st.text_input("Rules Markdown", value="data/approved_rules.md")
    payload = st.text_input("Rules JSON", value="data/approved_rules.json")
    skills = st.text_input("Skills dir", value="skills")
    gitlab_store = st.text_input("GitLab store", value=st.session_state.get("cfg_gitlab", "data/gitlab"))
    gitlab_sft = st.text_input("GitLab SFT", value="data/gitlab_sft.jsonl")
    gitlab_dpo = st.text_input("GitLab DPO", value="data/gitlab_dpo.jsonl")
    if st.button("Экспорт сессий", type="primary"):
        try:
            export_sessions(["--input", input_path, "--decisions", decisions, "--output", approved])
        except SystemExit as exc:
            st.error(str(exc))
        else:
            st.success(approved)
    if st.button("Экспорт правил и skills"):
        try:
            export_rules(
                [
                    "--proposals",
                    proposals,
                    "--decisions",
                    rule_decisions,
                    "--markdown",
                    markdown,
                    "--json",
                    payload,
                    "--skills-dir",
                    skills,
                ]
            )
        except SystemExit as exc:
            st.error(str(exc))
        else:
            st.success(f"{markdown} · {payload} · {skills}")
    if st.button("Экспорт GitLab (kept)"):
        try:
            export_gitlab(["export", "--store", gitlab_store, "--sft", gitlab_sft, "--dpo", gitlab_dpo, "--status", "keep"])
        except SystemExit as exc:
            st.error(str(exc) or "export failed")
        else:
            st.success(f"{gitlab_sft} · {gitlab_dpo}")


render()
