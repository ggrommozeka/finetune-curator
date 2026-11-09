"""Browse synced GitLab MRs and review examples produced offline."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gitlab.link import DEFAULT_FIX_PATTERN  # noqa: E402
from scripts.gitlab.store import GitlabStore  # noqa: E402
from scripts.gitlab.sync import process_store  # noqa: E402
from shared import esc as _esc  # noqa: E402
from shared import widget_value as _widget_value  # noqa: E402


def render() -> None:
    with st.sidebar:
        st.markdown("### ◈ GitLab")
        store_path = st.text_input(
            "GitLab store",
            key="gitlab_store",
            **_widget_value("gitlab_store", st.session_state.get("cfg_gitlab", "data/gitlab")),
        )
        project = st.text_input("Проект", key="gl_project", **_widget_value("gl_project", ""))
        author = st.text_input("Автор MR", key="gl_author", **_widget_value("gl_author", ""))
        reviewer = st.text_input("Ревьюер", key="gl_reviewer", **_widget_value("gl_reviewer", ""))
        since = st.text_input("С даты", key="gl_since", **_widget_value("gl_since", ""))
        until = st.text_input("По дату", key="gl_until", **_widget_value("gl_until", ""))
        fixes_only = st.checkbox("Только с MRR в заголовке", key="gl_fixes", **_widget_value("gl_fixes", False))
        team_leads = st.text_input(
            "Алиасы одного тимлида (username, имя, email)",
            key="gl_leads",
            **_widget_value("gl_leads", ""),
        )
        min_comments = st.number_input("Мин. комментариев", min_value=0, step=1, key="gl_min_comments")
        fix_pattern = st.text_input(
            "Регулярное выражение MRR",
            key="gl_pattern",
            **_widget_value("gl_pattern", DEFAULT_FIX_PATTERN),
        )
        anonymize = st.checkbox("Анонимизировать примеры", key="gl_anon", **_widget_value("gl_anon", False))
        anonymize_urls = st.checkbox("И URL", key="gl_anon_url", **_widget_value("gl_anon_url", False))
        st.caption("Обработка сохраняет комментарии всех ревьюеров. Поле «Ревьюер» только фильтрует список.")

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
                        fix_pattern=fix_pattern or DEFAULT_FIX_PATTERN,
                        anonymize=anonymize,
                        anonymize_urls=anonymize_urls,
                        signal_config=_signal_config(team_leads, fix_pattern),
                    )
                    st.success(f"Примеров: {stats['examples']}")
                    st.rerun()

        if not rows:
            st.info("Нет MR под текущие фильтры.")
            return

        overall = store.link_stats()
        if overall["total"]["comments"]:
            _render_stats(overall, "Все MR в хранилище")

        focus = st.session_state.pop("gitlab_focus", None)
        if focus and focus.get("project_id") and focus.get("mr_iid"):
            focused = f"{int(focus['project_id'])}!{int(focus['mr_iid'])}"
            if focused in labels:
                st.session_state.gitlab_mr = focused
        current = st.selectbox(
            "Merge request",
            list(labels),
            format_func=lambda ref: labels[ref],
            key="gitlab_mr",
        )
        summary = next(row for row in rows if row["ref"] == current)
        _render_mr(
            store,
            summary,
            reviewer or None,
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
          · {summary['comment_count']} comments · {summary.get('lead_count', 0)} team-lead · {summary['fix_count']} MRR titles</p>
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

    _render_timeline(bundle, _signal_config(st.session_state.get("gl_leads", ""), fix_pattern))

    traces = [
        trace
        for trace in store.traces_for(project_id, iid)
        if _reviewer_matches(trace.get("reviewer") or trace.get("author_username") or "", reviewer)
    ]
    if traces:
        _render_stats(store.link_stats(project_id, iid), "Этот MR")
        st.markdown("**Цепочка: комментарий → якорь → фикс → код**")
        for trace in traces:
            st.caption(f"ревьюер {trace.get('reviewer') or trace.get('author_username') or '—'}")
            _render_trace(store, summary, bundle, trace, fix_pattern, anonymize, anonymize_urls)
    else:
        st.markdown("**Комментарии**")
        from scripts.gitlab.link import flatten_notes

        for note in flatten_notes(bundle["discussions"], None):
            author = (note.get("author") or {}).get("username") or ""
            if not _reviewer_matches(author, reviewer):
                continue
            st.text(f"{author}: {note.get('body') or ''}")

    if st.button("Обработать этот MR", key=f"proc_{project_id}_{iid}"):
        process_store(
            store,
            [summary["ref"]],
            fix_pattern=fix_pattern,
            anonymize=anonymize,
            anonymize_urls=anonymize_urls,
            signal_config=_signal_config(st.session_state.get("gl_leads", ""), fix_pattern),
        )
        st.rerun()

    examples = []
    for item in store.examples_for(project_id, iid):
        record = item.get("record") or {}
        if record.get("source") == "gitlab_polish":
            examples.append(item)
            continue
        if _reviewer_matches((record.get("gitlab") or {}).get("reviewer") or "", reviewer):
            examples.append(item)
    if not examples:
        st.info("Примеров ещё нет. Нажмите «Обработать».")
        return
    st.markdown(f"**Примеры ({len(examples)})**")
    for item in examples:
        _render_example(store, item)


def _reviewer_matches(name: str, reviewer: str | None) -> bool:
    if not reviewer:
        return True
    return reviewer.lower() in (name or "").lower()


def _signal_config(team_leads: str, pattern: str):
    from scripts.gitlab.signals import load_signal_config

    leads = [part.strip() for part in (team_leads or "").split(",") if part.strip()]
    return load_signal_config(team_leads=leads, mrr_pattern=pattern or DEFAULT_FIX_PATTERN)


def _render_timeline(bundle: dict, config) -> None:
    from scripts.gitlab.classify import classify_commits
    from scripts.gitlab.link import parse_hunks
    from scripts.gitlab.trace import iter_threads

    roles = {row["sha"]: row for row in classify_commits(bundle["commits"], bundle["mr"], bundle["discussions"], config)}
    events = []
    for commit in bundle["commits"]:
        sha = str(commit.get("id") or "")
        role = roles.get(sha) or {}
        hunks = []
        for diff in bundle["diffs_by_sha"].get(sha) or []:
            path = diff.get("new_path") or diff.get("old_path") or ""
            for hunk in parse_hunks(diff.get("diff") or ""):
                hunks.append({"file": path, "before": hunk.before, "after": hunk.after})
        events.append(
            {
                "at": commit.get("created_at") or "",
                "type": "commit",
                "sha": sha,
                "role": role.get("role") or "other",
                "mrr": bool(role.get("mrr")),
                "title": (commit.get("title") or commit.get("message") or "").splitlines()[0],
                "hunks": hunks,
            }
        )
    for thread in iter_threads(bundle["discussions"], None):
        note = thread["root"]
        events.append(
            {
                "at": note.get("created_at") or "",
                "type": "comment",
                "author": (note.get("author") or {}).get("username") or "",
                "body": note.get("body") or "",
                "file": (note.get("position") or {}).get("new_path") or "",
            }
        )
    events.sort(key=lambda item: item["at"])
    st.markdown("**Хронология**")
    counts = {}
    for role in roles.values():
        counts[role["role"]] = counts.get(role["role"], 0) + 1
    if counts:
        st.caption(
            "Коммиты: "
            + ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
        )
    for event in events:
        if event["type"] == "comment":
            where = f" · {event['file']}" if event["file"] else ""
            st.markdown(f"`comment` **{_esc(event['author'])}**{where}")
            st.text(event["body"])
            continue
        lead = event["role"].startswith("team_lead")
        mark = "MRR" if event["mrr"] else event["role"]
        title = f"{'◆ ' if lead else ''}{mark}  {event['sha'][:10]}  {event['title']}"
        if lead:
            st.markdown(
                f"<div class='hero' style='padding:10px 14px'><p class='hero-title'>{_esc(title)}</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"`{_esc(mark)}` {event['sha'][:10]}  {_esc(event['title'])}")
        for hunk in event["hunks"]:
            left, right = st.columns(2)
            left.markdown(f"**До** `{hunk['file']}`")
            left.code(hunk["before"] or "—", language="text")
            right.markdown("**После**")
            right.code(hunk["after"] or "—", language="text")


def _render_stats(stats: dict, title: str) -> None:
    total = stats.get("total") or {}
    comments = total.get("comments") or 0
    linked = total.get("linked") or 0
    share = total.get("with_fix_share") or 0
    st.markdown(f"**{title}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Комментарии", comments)
    c2.metric("С фиксом", linked)
    c3.metric("Доля", f"{share:.0%}")
    c4.metric("Не закрыты", total.get("not_addressed") or 0)
    if stats.get("by_reviewer"):
        st.caption(
            "По ревьюерам: "
            + ", ".join(
                f"{name} {row['linked']}/{row['comments']}" for name, row in stats["by_reviewer"].items()
            )
        )
    if stats.get("by_file"):
        st.caption(
            "По файлам: "
            + ", ".join(f"{name} {row['linked']}/{row['comments']}" for name, row in stats["by_file"].items())
        )


def _render_trace(store, summary, bundle, trace, fix_pattern, anonymize, anonymize_urls) -> None:
    from scripts.gitlab.link import parse_hunks

    anchor = trace.get("anchor") or {}
    note_id = str(trace.get("note_id"))
    st.markdown(
        f"`{trace.get('author_username') or '—'}` ({trace.get('author_role') or 'commenter'}) · "
        f"{trace.get('status')} · {trace.get('confidence')}"
        + (" · вручную" if trace.get("manual") else "")
    )
    st.text(trace.get("body") or "")
    if trace.get("resolved"):
        st.caption(f"Тред resolved{(' · ' + trace['resolved_by']) if trace.get('resolved_by') else ''}")
    for reply in trace.get("replies") or []:
        st.caption(f"↳ {reply.get('author_username') or ''}: {reply.get('body') or ''}")
    file_label = anchor.get("file") or "(нет файла)"
    lines = anchor.get("new_line_start") or anchor.get("old_line_start")
    shas = " ".join(
        part
        for part in (
            f"base {anchor.get('base_sha', '')[:8]}" if anchor.get("base_sha") else "",
            f"head {anchor.get('head_sha', '')[:8]}" if anchor.get("head_sha") else "",
        )
        if part
    )
    st.caption(f"{file_label} · line {lines or '—'} · {shas}")
    if anchor.get("snippet"):
        st.markdown("**Якорь**")
        st.code(anchor["snippet"], language="text")
    commits = trace.get("changed_by") or []
    if commits:
        st.caption(
            "Коммиты, сдвинувшие или изменившие строки: "
            + ", ".join(f"{item['sha'][:8]}{'*' if item.get('preferred') else ''}" for item in commits)
        )
    left, right = st.columns(2)
    left.markdown("**До**")
    left.code(trace.get("before") or "—", language="text")
    right.markdown("**После**")
    right.code(trace.get("after") or "—", language="text")
    if trace.get("diff"):
        with st.expander("Diff якоря"):
            st.code(trace["diff"], language="diff")

    commit_ids = [str(commit.get("id") or "") for commit in bundle["commits"] if commit.get("id")]
    if commit_ids:
        picked = st.selectbox("Коммит связи", commit_ids, key=f"ov_sha_{note_id}")
        hunks = []
        for diff in bundle["diffs_by_sha"].get(picked) or []:
            hunks.extend(parse_hunks(diff.get("diff") or ""))
        labels = [f"{index}: {(hunk.text.splitlines() or ['hunk'])[0][:80]}" for index, hunk in enumerate(hunks)] or ["0"]
        hunk_label = st.selectbox("Ханк", labels, key=f"ov_hunk_{note_id}")
        hunk_index = labels.index(hunk_label) if hunk_label in labels else 0
        b1, b2 = st.columns(2)
        if b1.button("Поставить эту связь", key=f"ov_set_{note_id}"):
            store.set_link_override(
                summary["project_id"],
                note_id,
                "linked",
                commit_sha=picked,
                hunk_index=hunk_index,
            )
            process_store(
                store,
                [summary["ref"]],
                fix_pattern=fix_pattern,
                anonymize=anonymize,
                anonymize_urls=anonymize_urls,
                signal_config=_signal_config(st.session_state.get("gl_leads", ""), fix_pattern),
            )
            st.rerun()
        if b2.button("Не исправлено", key=f"ov_none_{note_id}"):
            store.set_link_override(summary["project_id"], note_id, "not_addressed")
            process_store(
                store,
                [summary["ref"]],
                fix_pattern=fix_pattern,
                anonymize=anonymize,
                anonymize_urls=anonymize_urls,
                signal_config=_signal_config(st.session_state.get("gl_leads", ""), fix_pattern),
            )
            st.rerun()
    st.markdown("---")


def _render_example(store: GitlabStore, item: dict) -> None:
    record = item["record"]
    decision = item["decision"]
    gitlab = record.get("gitlab") or {}
    example_id = item["id"]
    status = decision.get("status") or "undecided"
    st.markdown(
        f"`{example_id}` · source {record.get('source') or 'gitlab'} · {status} · "
        f"{gitlab.get('link') or '—'} · {gitlab.get('confidence')}"
    )
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


render()
