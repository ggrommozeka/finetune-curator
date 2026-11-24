"""Before/after metrics for adopted rule and skill versions, and rollback."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.monitor import report, rollback
from scripts.analysis.store import PatternStore
from shared import widget_value


def render() -> None:
    st.markdown(
        """<div class="hero"><p class="hero-title">Мониторинг</p>
        <p class="hero-sub">Сравнение сессий до и после версии правил или skills. Откат восстанавливает файлы экспорта.</p></div>""",
        unsafe_allow_html=True,
    )
    store_path = st.text_input(
        "Patterns store",
        key="patterns_store",
        **widget_value("patterns_store", st.session_state.get("cfg_patterns", "data/patterns")),
    )
    decisions = st.text_input(
        "Decisions",
        key="decisions_path",
        **widget_value("decisions_path", st.session_state.get("cfg_decisions", "data/decisions.json")),
    )
    error_type = st.text_input("Тип ошибки для recurrence", value="")
    root = Path(store_path)
    if not (root / "store.db").exists():
        st.info("Сначала `python -m scripts.mine_patterns import`, затем `export` с `--version`.")
        return
    store = PatternStore(root)
    try:
        versions = store.versions()
        rows = report(
            store,
            decisions_path=Path(decisions) if decisions else None,
            error_type=error_type or None,
        )
        if not versions:
            st.info("Версий ещё нет. Экспорт на странице Экспорт или `mine_patterns export` создаёт снимок.")
            return
        for row in rows:
            st.markdown(f"**{row['version']}** · с {row['adopted_at']}")
            left, right = st.columns(2)
            left.json(row["before"])
            right.json(row["after"])
        chosen = st.selectbox("Откатить к", [row["id"] for row in versions])
        if st.button("Откатить"):
            restored = rollback(store, chosen)
            st.success("Восстановлено: " + ", ".join(restored))
    finally:
        store.close()


render()
