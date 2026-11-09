"""Review mined patterns and approve each one as a rule or a skill."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.patterns.store import PatternStore  # noqa: E402


def _widget_value(key: str, default):
    if key in st.session_state:
        return {}
    return {"value": default}


def render_patterns_page(cli) -> None:
    del cli
    with st.sidebar:
        store_path = st.text_input(
            "Patterns store",
            key="patterns_store",
            **_widget_value("patterns_store", "data/patterns"),
        )
        machine = st.text_input("Машина", key="pat_machine", **_widget_value("pat_machine", ""))
        project = st.text_input("Проект", key="pat_project", **_widget_value("pat_project", ""))
        task = st.text_input("Тип задачи", key="pat_task", **_widget_value("pat_task", ""))

    root = Path(store_path)
    if not (root / "store.db").exists():
        st.markdown(
            """
            <div class="hero">
              <p class="hero-title">Нет разобранных сессий</p>
              <p class="hero-sub">Импортируйте истории с нескольких машин, затем аннотируйте и кластеризуйте.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(
            "python -m scripts.mine_patterns import --source laptop-a --input data/a.jsonl\n"
            "python -m scripts.mine_patterns annotate\n"
            "python -m scripts.mine_patterns cluster\n"
            "python -m scripts.mine_patterns recommend",
            language="bash",
        )
        return

    store = PatternStore(root)
    try:
        clusters = [
            cluster
            for cluster in store.clusters(include_noise=False)
            if _matches(cluster, machine, project, task)
        ]
        proposals = {row["cluster_id"]: row for row in store.proposals()}
        st.markdown(
            f"""
            <div class="hero">
              <p class="hero-title">Паттерны</p>
              <p class="hero-sub">{len(clusters)} кластеров после отсечения шума</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not clusters:
            st.info("Нет кластеров под фильтр. Шум (1–2 сессии) скрыт.")
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
        _render_cluster(store, cluster, proposals.get(current))
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


def _render_cluster(store: PatternStore, cluster: dict, proposal: dict | None) -> None:
    st.caption(
        f"машины: {', '.join(cluster.get('machine_names') or []) or '—'} · "
        f"проекты: {', '.join(cluster.get('project_names') or []) or '—'} · "
        f"задачи: {', '.join(cluster.get('task_type_names') or []) or '—'}"
    )
    st.markdown("**Сессии**")
    for gid in cluster.get("session_ids") or []:
        annotation = store.annotation(gid) or {}
        st.text(
            f"{gid}  {annotation.get('outcome') or ''}  {annotation.get('error_type') or ''}  "
            f"{annotation.get('user_correction') or annotation.get('what_worked') or ''}"
        )
    if proposal is None:
        st.info("Нет предложения. Запустите recommend.")
        return
    payload = proposal["payload"]
    st.markdown(f"**Предложение: {proposal['kind']} · {proposal['status']}**")
    if proposal.get("conflicts"):
        st.warning("\n".join(proposal["conflicts"]))
    text_key = f"pat_text_{proposal['id']}"
    if text_key not in st.session_state:
        st.session_state[text_key] = payload.get("text") or ""
    text = st.text_area("Текст", key=text_key, height=140)
    kind = st.selectbox("Тип", ["rule", "skill"], index=0 if proposal["kind"] == "rule" else 1, key=f"pat_kind_{proposal['id']}")
    c1, c2, c3 = st.columns(3)
    if c1.button("Одобрить", key=f"pat_ok_{proposal['id']}", type="primary"):
        store.set_proposal_status(proposal["id"], "approved", {"text": text, "kind": kind})
        st.rerun()
    if c2.button("Отклонить", key=f"pat_no_{proposal['id']}"):
        store.set_proposal_status(proposal["id"], "rejected", {"text": text, "kind": kind})
        st.rerun()
    if c3.button("Сбросить", key=f"pat_reset_{proposal['id']}"):
        store.set_proposal_status(proposal["id"], "pending", {"text": text})
        st.rerun()
