"""Browse mined clusters. Approval happens on the shared proposals page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.patterns.store import PatternStore  # noqa: E402
from scripts.rules.store import load_proposals, load_rule_decisions, effective_proposals  # noqa: E402


def _widget_value(key: str, default):
    if key in st.session_state:
        return {}
    return {"value": default}


def render_patterns_page(cli) -> None:
    with st.sidebar:
        store_path = st.text_input(
            "Patterns store",
            key="patterns_store",
            **_widget_value("patterns_store", "data/patterns"),
        )
        proposals_path = st.text_input(
            "Proposals",
            key="rule_proposals_path",
            **_widget_value("rule_proposals_path", cli.rules),
        )
        decisions_path = st.text_input(
            "Rule decisions",
            key="rule_decisions_path",
            **_widget_value("rule_decisions_path", cli.rule_decisions),
        )
        machine = st.text_input("Машина", key="pat_machine", **_widget_value("pat_machine", ""))
        project = st.text_input("Проект", key="pat_project", **_widget_value("pat_project", ""))
        task = st.text_input("Тип задачи", key="pat_task", **_widget_value("pat_task", ""))
        st.caption("Одобрение правил и skills — на странице **Правила**. Здесь только кластеры.")

    root = Path(store_path)
    if not (root / "store.db").exists():
        st.info("Сначала import, annotate, cluster, recommend.")
        st.code(
            "python -m scripts.mine_patterns import --source laptop-a --input data/a.jsonl\n"
            "python -m scripts.mine_patterns annotate\n"
            "python -m scripts.mine_patterns cluster --min-size 3\n"
            "python -m scripts.mine_patterns recommend",
            language="bash",
        )
        return

    store = PatternStore(root)
    try:
        clusters = [cluster for cluster in store.clusters() if _matches(cluster, machine, project, task)]
        document = load_proposals(Path(proposals_path))
        decisions = load_rule_decisions(Path(decisions_path))
        by_cluster = {}
        for proposal in effective_proposals(document, decisions):
            if proposal.get("cluster_id"):
                by_cluster[proposal["cluster_id"]] = proposal
        st.markdown(
            f"""
            <div class="hero">
              <p class="hero-title">Паттерны</p>
              <p class="hero-sub">{len(clusters)} кластеров. Одобрение — на странице Правила.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not clusters:
            st.info("Нет кластеров под фильтр. Шум (1–2 сессии) скрыт. Для отладки: cluster --min-size 1.")
            return
        labels = {
            cluster["id"]: (
                f"{cluster.get('label')} · n={cluster.get('frequency')} · "
                f"fail {cluster.get('fail_rate')} · tasks {cluster.get('task_types')}"
            )
            for cluster in clusters
        }
        current = st.selectbox("Кластер", list(labels), format_func=lambda key: labels[key])
        cluster = next(item for item in clusters if item["id"] == current)
        proposal = by_cluster.get(current)
        st.caption(
            f"машины: {', '.join(cluster.get('machine_names') or []) or '—'} · "
            f"проекты: {', '.join(cluster.get('project_names') or []) or '—'} · "
            f"задачи: {', '.join(cluster.get('task_type_names') or []) or '—'}"
        )
        if proposal:
            st.markdown(f"**Предложение {proposal['id']} · {proposal['kind']} · {proposal['status']}**")
            st.text(proposal.get("text") or "")
            if st.button("Открыть на странице Правила"):
                st.session_state.pending_page = "Правила"
                st.rerun()
        else:
            st.info("Для этого кластера ещё нет предложения в общем файле proposals.")
        st.markdown("**Сессии**")
        for gid in cluster.get("session_ids") or []:
            annotation = store.annotation(gid) or {}
            st.text(
                f"{gid}  {annotation.get('outcome') or ''}  {annotation.get('error_type') or ''}  "
                f"{annotation.get('user_correction') or annotation.get('what_worked') or ''}"
            )
    finally:
        store.close()


def _matches(cluster: dict, machine: str, project: str, task: str) -> bool:
    if machine and machine.lower() not in " ".join(cluster.get("machine_names") or []).lower():
        return False
    if project and project.lower() not in " ".join(cluster.get("project_names") or []).lower():
        return False
    if task and task.lower() not in " ".join(cluster.get("task_type_names") or []).lower():
        return False
    return True
