"""Export training data, rules, and skills from the shared stores."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_dataset import main as build_finetune
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
    st.markdown("#### Датасеты для обучения")
    st.caption(
        "agentic и style — Qwen3 27B, шаблон qwen3, до 32k токенов. "
        "review — отдельный датасет для модели ~9B."
    )
    dataset_task = st.selectbox("Задача", ["all", "agentic", "style", "review"], key="finetune_task")
    chat_template = st.selectbox("Chat template", ["qwen3", "chatml", "llama3", "gemma"], key="finetune_template")
    output_dir = st.text_input("Output dir", value=f"data/finetune/{dataset_task}", key=f"finetune_dir_{dataset_task}")
    token_default = 8192 if dataset_task == "review" else 32000
    max_tokens = st.number_input(
        "Макс. токенов",
        min_value=256,
        value=token_default,
        step=256,
        key=f"finetune_tokens_{dataset_task}",
    )
    context_lines = st.number_input(
        "Строк контекста вокруг комментария",
        min_value=1,
        value=20,
        step=1,
        key="finetune_context_lines",
    )
    agentic = style = review = 0.0
    if dataset_task == "all":
        agentic = st.slider("Доля agentic", 0.0, 1.0, 0.5, 0.05)
        style = st.slider("Доля style", 0.0, 1.0, 0.25, 0.05)
        review = st.slider("Доля review", 0.0, 1.0, 0.25, 0.05)
    eval_ratio = st.slider("Held-out", 0.0, 0.5, 0.1, 0.05)
    anonymize = st.checkbox("Анонимизировать датасет", value=False)
    keep_thinking = st.checkbox("Оставить блоки <think>", value=False)
    if st.button("Собрать датасет"):
        argv = [
            "--input",
            input_path,
            "--decisions",
            decisions,
            "--gitlab-store",
            gitlab_store,
            "--patterns-store",
            st.session_state.get("cfg_patterns", "data/patterns"),
            "--output-dir",
            output_dir,
            "--task",
            dataset_task,
            "--chat-template",
            chat_template,
            "--max-tokens",
            str(int(max_tokens)),
            "--review-context-lines",
            str(int(context_lines)),
            "--eval-ratio",
            str(eval_ratio),
        ]
        if dataset_task == "all":
            argv.extend(["--agentic", str(agentic), "--style", str(style), "--review", str(review)])
        if anonymize:
            argv.append("--anonymize")
        argv.extend(["--thinking", "keep" if keep_thinking else "strip"])
        try:
            build_finetune(argv)
        except SystemExit as exc:
            st.error(str(exc) or "build failed")
        else:
            st.success(output_dir)


render()
