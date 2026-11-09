#!/usr/bin/env python3
"""finetune-curator — polished trajectory reviewer for LogMiner / OpenCode JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import streamlit as st

DEFAULT_INPUT = "data/raw_scored.jsonl"
DEFAULT_DECISIONS = "data/decisions.json"

st.set_page_config(
    page_title="finetune-curator",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
  [data-testid="stSidebar"] { background: #0f1419; }
  [data-testid="stSidebar"] * { color: #e7ecf3; }
  h1, h2, h3 { letter-spacing: -0.02em; }

  .msg {
    border-radius: 14px;
    padding: 12px 16px;
    margin: 10px 0;
    border: 1px solid transparent;
    line-height: 1.45;
    font-size: 0.92rem;
  }
  .msg-user {
    background: linear-gradient(135deg, #1a2332 0%, #162032 100%);
    border-color: #2a3f5f;
    margin-right: 12%;
  }
  .msg-assistant {
    background: linear-gradient(135deg, #141a14 0%, #182018 100%);
    border-color: #2d4a2d;
    margin-left: 4%;
  }
  .msg-tool {
    background: #12151a;
    border-color: #2a2f38;
    margin-left: 8%;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.82rem;
  }
  .msg-system {
    background: #1a1520;
    border-color: #3a2a4a;
    margin: 8px 8%;
    opacity: 0.9;
  }
  .msg-role {
    font-size: 0.72rem;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
    opacity: 0.75;
  }
  .msg-user .msg-role { color: #7db4ff; }
  .msg-assistant .msg-role { color: #8fd98f; }
  .msg-tool .msg-role { color: #e0b84a; }
  .msg-system .msg-role { color: #c49bff; }
  .msg-body {
    white-space: pre-wrap;
    word-break: break-word;
    color: #e8edf5;
  }
  .msg-meta {
    margin-top: 8px;
    font-size: 0.75rem;
    opacity: 0.55;
  }
  .tool-chip {
    display: inline-block;
    background: #2a2410;
    color: #e0b84a;
    border: 1px solid #4a3f18;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px 4px 2px 0;
  }

  .hero {
    background: linear-gradient(120deg, #121820 0%, #0f1a14 50%, #12141c 100%);
    border: 1px solid #243040;
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 14px;
  }
  .hero-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f0f4fa;
    margin: 0;
  }
  .hero-sub {
    color: #8b9bb0;
    font-size: 0.88rem;
    margin-top: 4px;
  }
  .pill {
    display: inline-block;
    border-radius: 999px;
    padding: 3px 11px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
  }
  .pill-good { background: #16301c; color: #7ddea0; border: 1px solid #2a5a38; }
  .pill-maybe { background: #302810; color: #e0c060; border: 1px solid #5a4a18; }
  .pill-bad { background: #301818; color: #f0a0a0; border: 1px solid #5a2828; }
  .pill-keep { background: #12301c; color: #6fdf9a; border: 1px solid #2a6a40; }
  .pill-discard { background: #301418; color: #f09090; border: 1px solid #6a2830; }
  .pill-undecided { background: #1a1e28; color: #a0aec0; border: 1px solid #303848; }
  .pill-score { background: #142038; color: #90b8f0; border: 1px solid #284868; }

  .action-bar {
    position: sticky;
    bottom: 0;
    background: rgba(12, 16, 22, 0.92);
    backdrop-filter: blur(10px);
    border-top: 1px solid #2a3444;
    padding: 12px 8px 4px;
    margin-top: 18px;
    z-index: 20;
  }

  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    args, _ = parser.parse_known_args()
    return args


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_decisions(path: Path) -> dict:
    if not path.exists():
        return {"labels": {}, "notes": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"labels": {}, "notes": {}}
    if data and all(isinstance(v, str) for v in data.values()):
        return {"labels": {str(k): v.lower() for k, v in data.items()}, "notes": {}}
    labels = {str(k): str(v).lower() for k, v in (data.get("labels") or {}).items()}
    notes = {str(k): str(v) for k, v in (data.get("notes") or {}).items()}
    return {"labels": labels, "notes": notes}


def save_decisions(path: Path, decisions: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")


def session_id(rec: dict) -> str:
    return str(
        rec.get("id")
        or rec.get("session_id")
        or f"idx-{abs(hash(json.dumps(rec, sort_keys=True, default=str))) % 10**8}"
    )


def normalize_messages(rec: dict) -> list[dict]:
    msgs = rec.get("messages") or []
    if not msgs:
        return []
    if isinstance(msgs[0], str):
        out = []
        for m in msgs:
            try:
                out.append(json.loads(m))
            except json.JSONDecodeError:
                out.append({"role": "unknown", "content": m})
        return out
    return msgs


def first_user_preview(msgs: list[dict], n: int = 90) -> str:
    for m in msgs:
        if m.get("role") == "user":
            c = m.get("content") or ""
            if not isinstance(c, str):
                c = json.dumps(c, ensure_ascii=False)
            c = re.sub(r"\s+", " ", c).strip()
            return (c[:n] + "…") if len(c) > n else c
    return "—"


def content_to_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def label_pill(label: str) -> str:
    lab = (label or "unknown").lower()
    cls = {"good": "pill-good", "maybe": "pill-maybe", "bad": "pill-bad"}.get(lab, "pill-undecided")
    return f'<span class="pill {cls}">{lab}</span>'


def status_pill(status: str) -> str:
    s = (status or "undecided").lower()
    cls = {
        "keep": "pill-keep",
        "discard": "pill-discard",
        "undecided": "pill-undecided",
    }.get(s, "pill-undecided")
    return f'<span class="pill {cls}">{s}</span>'


def _esc(s: str) -> str:
    return (
        s.replace("&", "&")
        .replace("<", "<")
        .replace(">", ">")
        .replace('"', """)
    )


def render_message(i: int, m: dict) -> None:
    role = (m.get("role") or "?").lower()
    css = {
        "user": "msg-user",
        "assistant": "msg-assistant",
        "tool": "msg-tool",
        "system": "msg-system",
    }.get(role, "msg-assistant")

    body = content_to_str(m.get("content"))
    shown = body if len(body) <= 6000 else body[:6000] + "\n… [truncated]"

    tool_bits = ""
    if role == "assistant" and m.get("tool_calls"):
        names = []
        for tc in m["tool_calls"]:
            if isinstance(tc, dict):
                fn = (tc.get("function") or {})
                names.append(fn.get("name") or tc.get("name") or "tool")
        chips = "".join(f'<span class="tool-chip">{n}</span>' for n in names)
        tool_bits = f"<div style='margin:6px 0'>{chips}</div>"
        try:
            tool_bits += (
                "<details><summary style='cursor:pointer;opacity:0.7;font-size:0.8rem'>"
                "tool_calls JSON</summary>"
                f"<pre style='font-size:0.75rem;overflow:auto'>{json.dumps(m['tool_calls'], ensure_ascii=False, indent=2)[:4000]}</pre>"
                "</details>"
            )
        except Exception:
            pass

    meta = ""
    if role == "tool":
        meta = f'<div class="msg-meta">tool_call_id: {m.get("tool_call_id", "")}</div>'

    html = f"""
    <div class="msg {css}">
      <div class="msg-role">{i}. {role}</div>
      {tool_bits}
      <div class="msg-body">{_esc(shown)}</div>
      {meta}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    cli = parse_cli_args()

    with st.sidebar:
        st.markdown("### ◈ finetune-curator")
        st.caption("Разметка траекторий для SFT / Unsloth")
        input_path = st.text_input("JSONL", cli.input)
        decisions_path = st.text_input("Decisions", cli.decisions)
        reload = st.button("↻ Reload", use_container_width=True)

    input_p = Path(input_path)
    decisions_p = Path(decisions_path)

    if "rows" not in st.session_state or reload:
        st.session_state.rows = load_jsonl(input_p)
        st.session_state.decisions = load_decisions(decisions_p)
        st.session_state.idx = 0

    rows: list[dict] = st.session_state.rows
    decisions: dict = st.session_state.decisions
    labels: dict[str, str] = decisions.setdefault("labels", {})
    notes: dict[str, str] = decisions.setdefault("notes", {})

    if not rows:
        st.markdown(
            f"""
            <div class="hero">
              <p class="hero-title">Нет данных</p>
              <p class="hero-sub">Файл <code>{_esc(str(input_p))}</code> пуст или не найден.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(
            "python -m scripts.export_opencode --output data/raw_scored.jsonl --min-score 0\n"
            "streamlit run viewer/app.py -- --input data/raw_scored.jsonl",
            language="bash",
        )
        st.stop()

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Фильтры**")
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.05)
        all_labels = sorted({str(r.get("label", "unknown")) for r in rows})
        label_filter = st.multiselect("Label", all_labels, default=all_labels)
        status_filter = st.multiselect(
            "Статус",
            ["undecided", "keep", "discard"],
            default=["undecided", "keep", "discard"],
        )
        only_tools = st.checkbox("Только с tool calls", value=False)
        query = st.text_input("Поиск в user-тексте", "")

        def status_of(rec: dict) -> str:
            return labels.get(session_id(rec), "undecided")

        filtered: list[dict] = []
        for r in rows:
            if float(r.get("score") or 0) < min_score:
                continue
            if str(r.get("label", "unknown")) not in label_filter:
                continue
            if status_of(r) not in status_filter:
                continue
            msgs = normalize_messages(r)
            if only_tools:
                has_tools = any(m.get("tool_calls") for m in msgs if m.get("role") == "assistant")
                if not has_tools:
                    continue
            if query:
                preview = first_user_preview(msgs, 500).lower()
                if query.lower() not in preview:
                    continue
            filtered.append(r)

        n_keep = sum(1 for v in labels.values() if v == "keep")
        n_disc = sum(1 for v in labels.values() if v == "discard")
        n_und = max(0, len(rows) - n_keep - n_disc)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Keep", n_keep)
        c2.metric("Discard", n_disc)
        c3.metric("Todo", n_und)
        st.progress(0 if not rows else (n_keep + n_disc) / len(rows))
        st.caption(f"В фильтре: **{len(filtered)}** / {len(rows)}")

        if st.button("💾 Save decisions", use_container_width=True):
            save_decisions(decisions_p, decisions)
            st.success(f"→ {decisions_p}")

        st.markdown("---")
        st.markdown(
            f"""```bash
python -m scripts.export_approved \\
  --input {input_path} \\
  --decisions {decisions_path} \\
  --output data/approved.jsonl
```"""
        )

    if not filtered:
        st.info("Нет сессий под текущие фильтры.")
        st.stop()

    idx = int(st.session_state.get("idx", 0))
    idx = max(0, min(idx, len(filtered) - 1))
    st.session_state.idx = idx

    rec = filtered[idx]
    sid = session_id(rec)
    current = labels.get(sid, "undecided")
    msgs = normalize_messages(rec)
    metrics = rec.get("metrics") or {}

    score = rec.get("score", "—")
    lab = rec.get("label", "—")
    st.markdown(
        f"""
        <div class="hero">
          <p class="hero-title"><code>{_esc(sid)}</code></p>
          <p class="hero-sub">{_esc(first_user_preview(msgs, 140))}</p>
          <div style="margin-top:10px">
            <span class="pill pill-score">score {score}</span>
            {label_pill(str(lab))}
            {status_pill(current)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Messages", metrics.get("total_messages", len(msgs)))
    m2.metric("Tokens~", metrics.get("total_tokens", "—"))
    m3.metric("Tool calls", metrics.get("tool_calls", "—"))
    tools = rec.get("tools_used") or []
    m4.metric("Tools", len(tools) if tools else "—")
    m5.metric("#", f"{idx + 1}/{len(filtered)}")

    if tools:
        st.caption("tools: " + ", ".join(tools))
    reasons = rec.get("reasons") or []
    if reasons:
        st.caption("reasons: " + " · ".join(str(x) for x in reasons))

    n1, n2, n3 = st.columns([1, 2, 1])
    with n1:
        if st.button("← Prev", use_container_width=True) and idx > 0:
            st.session_state.idx = idx - 1
            st.rerun()
    with n2:
        jump = st.slider("Сессия в фильтре", 1, len(filtered), idx + 1, label_visibility="collapsed")
        if jump - 1 != idx:
            st.session_state.idx = jump - 1
            st.rerun()
    with n3:
        if st.button("Next →", use_container_width=True) and idx < len(filtered) - 1:
            st.session_state.idx = idx + 1
            st.rerun()

    st.markdown("#### Диалог")
    for i, m in enumerate(msgs):
        render_message(i + 1, m)

    note_val = st.text_area(
        "Заметка к сессии (сохраняется в decisions)",
        value=notes.get(sid, ""),
        height=80,
        placeholder="Почему keep/discard, баги, что вырезать…",
        key=f"note_{sid}",
    )
    if note_val != notes.get(sid, ""):
        notes[sid] = note_val
        st.session_state.decisions = decisions

    st.markdown('<div class="action-bar">', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns([1.4, 1.4, 1, 1.2])
    with a1:
        if st.button("✅ Keep", type="primary", use_container_width=True):
            labels[sid] = "keep"
            notes[sid] = note_val
            st.session_state.decisions = decisions
            save_decisions(decisions_p, decisions)
            if idx < len(filtered) - 1:
                st.session_state.idx = idx + 1
            st.rerun()
    with a2:
        if st.button("❌ Discard", use_container_width=True):
            labels[sid] = "discard"
            notes[sid] = note_val
            st.session_state.decisions = decisions
            save_decisions(decisions_p, decisions)
            if idx < len(filtered) - 1:
                st.session_state.idx = idx + 1
            st.rerun()
    with a3:
        if st.button("Clear", use_container_width=True):
            labels.pop(sid, None)
            st.session_state.decisions = decisions
            save_decisions(decisions_p, decisions)
            st.rerun()
    with a4:
        if st.button("💾 Save", use_container_width=True):
            notes[sid] = note_val
            st.session_state.decisions = decisions
            save_decisions(decisions_p, decisions)
            st.toast("Saved")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Keep / Discard сразу пишут decisions.json и переходят к следующей сессии.")


if __name__ == "__main__":
    main()
