"""Browse synced GitLab MRs and review examples produced offline."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gitlab.link import DEFAULT_FIX_PATTERN  # noqa: E402
from scripts.gitlab.store import GitlabStore  # noqa: E402
from scripts.gitlab.sync import process_store  # noqa: E402


def _widget_value(key: str, default):
    if key in st.session_state:
        return {}
    return {"value": default}


def render_gitlab_page(cli) -> None:
    del cli
    with st.sidebar:
        store_path = st.text_input("GitLab store", key="gitlab_store", **_widget_value("gitlab_store", "data/gitlab"))
        project = st.text_input("Проект", key="gl_project", **_widget_value("gl_project", ""))
        author = st.text_input("Автор MR", key="gl_author", **_widget_value("gl_author", ""))
        reviewer = st.text_input("Ревьюер", key="gl_reviewer", **_widget_value("gl_reviewer", ""))
        since = st.text_input("С даты", key="gl_since", **_widget_value("gl_since", ""))
        until = st.text_input("По дату", key="gl_until", **_widget_value("gl_until", ""))
        fixes_only = st.checkbox("Только с MRR fixes", key="gl_fixes", **_widget_value("gl_fixes", False))
        min_comments = st.number_input("Мин. комментариев", min_value=0, step=1, key="gl_min_comments")
        fix_pattern = st.text_input(
            "Паттерн фикса",
            key="gl_pattern",
            **_widget_value("gl_pattern", DEFAULT_FIX_PATTERN),
        )
        anonymize = st.checkbox("Анонимизировать примеры", key="gl_anon", **_widget_value("gl_anon", False))
        anonymize_urls = st.checkbox("И URL", key="gl_anon_url", **_widget_value("gl_anon_url", False))
        process_reviewer = st.text_input(
            "Обрабатывать только этого ревьюера",
            key="gl_process_reviewer",
            **_widget_value("gl_process_reviewer", ""),
        )

    root = Path(store_path)
    if not (root / "store.db").exists():
        st.markdown(
            """
            <div class="hero">
              <p class="hero-title">Нет локального хранилища</p>
              <p class="hero-sub">Сначала синхронизируйте GitLab. Дальше всё работает офлайн.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(
            "export GITLAB_TOKEN=...\n"
            "python -m scripts.export_gitlab_mrr sync \\\n"
            "  --base-url https://gitlab.company.internal \\\n"
            "  --group my-group \\\n"
            "  --since 2025-01-01 --until 2026-01-01",
            language="bash",
        )
        return

    store = GitlabStore(root)
    try:
        rows = store.list_mrs(
            project=project,
            author=author,
            reviewer=reviewer,
            since=since,
            until=until,
            fixes_only=fixes_only,
            min_comments=int(min_comments or 0),
            fix_pattern=fix_pattern or DEFAULT_FIX_PATTERN,
        )
        labels = {
            row["ref"]: f"{row['ref']}  {row['title']}  ({row['comment_count']} notes, {row['fix_count']} fixes)"
            for row in rows
        }
        with st.sidebar:
            st.caption(f"MR в фильтре: **{len(rows)}**")
            chosen = st.multiselect("Пакет", list(labels), format_func=lambda ref: labels[ref])
            if st.button("Обработать выбранные", use_container_width=True):
                targets = chosen or []
                if not targets:
                    st.warning("Выберите хотя бы один MR.")
                else:
                    stats = process_store(
                        store,
                        targets,
                        reviewer=process_reviewer or None,
                        fix_pattern=fix_pattern or DEFAULT_FIX_PATTERN,
                        anonymize=anonymize,
                        anonymize_urls=anonymize_urls,
                    )
                    st.success(f"Примеров: {stats['examples']}")
                    st.rerun()

        if not rows:
            st.info("Нет MR под текущие фильтры.")
            return

        current = st.selectbox("Merge request", list(labels), format_func=lambda ref: labels[ref])
        summary = next(row for row in rows if row["ref"] == current)
        _render_mr(
            store,
            summary,
            process_reviewer or None,
            fix_pattern or DEFAULT_FIX_PATTERN,
            anonymize,
            anonymize_urls,
        )
    finally:
        store.close()


def _render_mr(store: GitlabStore, summary: dict, reviewer: str | None, fix_pattern: str, anonymize: bool, anonymize_urls: bool) -> None:
    project_id = summary["project_id"]
    iid = summary["iid"]
    bundle = store.load_bundle(project_id, iid)
    if bundle is None:
        st.error("MR пропал из хранилища.")
        return
    mr = bundle["mr"]
    st.markdown(
        f"""
        <div class="hero">
          <p class="hero-title">!{iid} {_esc(summary['title'])}</p>
          <p class="hero-sub">{_esc(summary['author_username'])} · { _esc(summary['merged_at'] or '—') }
          · {summary['comment_count']} comments · {summary['fix_count']} MRR fixes</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if summary.get("web_url"):
        st.caption(summary["web_url"])
    description = (mr.get("description") or "").strip()
    if description:
        st.markdown("**Описание**")
        st.text(description[:4000])

    st.markdown("**Коммиты**")
    from scripts.gitlab.link import is_fix_message

    for commit in bundle["commits"]:
        message = (commit.get("message") or commit.get("title") or "").strip().splitlines()[0] if (commit.get("message") or commit.get("title")) else commit.get("id", "")
        mark = "FIX" if is_fix_message(commit.get("message") or commit.get("title") or "", fix_pattern) else "commit"
        st.caption(f"{mark}  {commit.get('id', '')[:10]}  {message}")

    st.markdown("**Комментарии**")
    from scripts.gitlab.link import flatten_notes

    for note in flatten_notes(bundle["discussions"], reviewer):
        author = (note.get("author") or {}).get("username") or ""
        st.text(f"{author}: {note.get('body') or ''}")

    if bundle["diffs_by_sha"]:
        with st.expander("Diffs фикс-коммитов и остальные diff"):
            for sha, diffs in bundle["diffs_by_sha"].items():
                st.caption(sha[:12])
                for diff in diffs:
                    st.caption(diff.get("new_path") or diff.get("old_path") or "")
                    st.code(diff.get("diff") or "", language="diff")

    if st.button("Обработать этот MR", key=f"proc_{project_id}_{iid}"):
        process_store(
            store,
            [summary["ref"]],
            reviewer=reviewer,
            fix_pattern=fix_pattern,
            anonymize=anonymize,
            anonymize_urls=anonymize_urls,
        )
        st.rerun()

    examples = store.examples_for(project_id, iid)
    if not examples:
        st.info("Примеров ещё нет. Нажмите «Обработать».")
        return
    st.markdown(f"**Примеры ({len(examples)})**")
    for item in examples:
        _render_example(store, item)


def _render_example(store: GitlabStore, item: dict) -> None:
    record = item["record"]
    decision = item["decision"]
    gitlab = record.get("gitlab") or {}
    example_id = item["id"]
    status = decision.get("status") or "undecided"
    st.markdown(f"`{example_id}` · {status} · {gitlab.get('link') or '—'} · {gitlab.get('confidence')}")
    user_key = f"gl_user_{example_id}"
    asst_key = f"gl_asst_{example_id}"
    messages = record.get("messages") or []
    if user_key not in st.session_state:
        st.session_state[user_key] = decision.get("user_text") or (messages[0].get("content") if messages else "")
    if asst_key not in st.session_state:
        st.session_state[asst_key] = decision.get("assistant_text") or (
            messages[1].get("content") if len(messages) > 1 else ""
        )
    user_text = st.text_area("Вход", key=user_key, height=140)
    assistant_text = st.text_area("Выход", key=asst_key, height=140)
    left, right = st.columns(2)
    left.markdown("**Before**")
    left.code(gitlab.get("before") or "—", language="text")
    right.markdown("**After**")
    right.code(decision.get("assistant_text") or gitlab.get("after") or "—", language="text")
    b1, b2, b3 = st.columns(3)
    if b1.button("Keep", key=f"keep_{example_id}", type="primary"):
        store.set_decision(example_id, "keep", user_text=user_text, assistant_text=assistant_text)
        st.rerun()
    if b2.button("Discard", key=f"discard_{example_id}"):
        store.set_decision(example_id, "discard", user_text=user_text, assistant_text=assistant_text)
        st.rerun()
    if b3.button("Сбросить", key=f"reset_{example_id}"):
        store.set_decision(example_id, "undecided", user_text="", assistant_text="")
        st.rerun()
    st.markdown("---")


def _esc(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
