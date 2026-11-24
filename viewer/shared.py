"""Shared Streamlit helpers for every page of the curator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = "data/raw_scored.jsonl"
DEFAULT_DECISIONS = "data/decisions.json"
DEFAULT_PROPOSALS = "data/rule_proposals.json"
DEFAULT_RULE_DECISIONS = "data/rule_decisions.json"
DEFAULT_GITLAB = "data/gitlab"
DEFAULT_PATTERNS = "data/patterns"

CSS = """
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
  [data-testid="stSidebar"] { background: #0f1419; }
  [data-testid="stSidebar"] * { color: #e7ecf3; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  .msg { border-radius: 14px; padding: 12px 16px; margin: 10px 0; border: 1px solid transparent; line-height: 1.45; font-size: 0.92rem; }
  .msg-user { background: linear-gradient(135deg, #1a2332 0%, #162032 100%); border-color: #2a3f5f; margin-right: 12%; }
  .msg-assistant { background: linear-gradient(135deg, #141a14 0%, #182018 100%); border-color: #2d4a2d; margin-left: 4%; }
  .msg-tool { background: #12151a; border-color: #2a2f38; margin-left: 8%; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }
  .msg-system { background: #1a1520; border-color: #3a2a4a; margin: 8px 8%; opacity: 0.9; }
  .msg-role { font-size: 0.72rem; font-weight: 650; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; opacity: 0.75; }
  .msg-user .msg-role { color: #7db4ff; }
  .msg-assistant .msg-role { color: #8fd98f; }
  .msg-tool .msg-role { color: #e0b84a; }
  .msg-system .msg-role { color: #c49bff; }
  .msg-body { white-space: pre-wrap; word-break: break-word; color: #e8edf5; }
  .msg-meta { margin-top: 8px; font-size: 0.75rem; opacity: 0.55; }
  .tool-chip { display: inline-block; background: #2a2410; color: #e0b84a; border: 1px solid #4a3f18; border-radius: 999px; padding: 2px 10px; font-size: 0.75rem; margin: 2px 4px 2px 0; }
  .hero { background: linear-gradient(120deg, #121820 0%, #0f1a14 50%, #12141c 100%); border: 1px solid #243040; border-radius: 16px; padding: 18px 22px; margin-bottom: 14px; }
  .hero-title { font-size: 1.35rem; font-weight: 700; color: #f0f4fa; margin: 0; }
  .hero-sub { color: #8b9bb0; font-size: 0.88rem; margin-top: 4px; }
  .pill { display: inline-block; border-radius: 999px; padding: 3px 11px; font-size: 0.78rem; font-weight: 600; margin-right: 6px; }
  .pill-good { background: #16301c; color: #7ddea0; border: 1px solid #2a5a38; }
  .pill-maybe { background: #302810; color: #e0c060; border: 1px solid #5a4a18; }
  .pill-bad { background: #301818; color: #f0a0a0; border: 1px solid #5a2828; }
  .pill-keep, .pill-approved { background: #12301c; color: #6fdf9a; border: 1px solid #2a6a40; }
  .pill-discard, .pill-rejected { background: #301418; color: #f09090; border: 1px solid #6a2830; }
  .pill-undecided, .pill-pending { background: #1a1e28; color: #a0aec0; border: 1px solid #303848; }
  .pill-score { background: #142038; color: #90b8f0; border: 1px solid #284868; }
  .action-bar { position: sticky; bottom: 0; background: rgba(12, 16, 22, 0.92); backdrop-filter: blur(10px); border-top: 1px solid #2a3444; padding: 12px 8px 4px; margin-top: 18px; z-index: 20; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
"""


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    parser.add_argument("--rules", default=DEFAULT_PROPOSALS)
    parser.add_argument("--rule-decisions", default=DEFAULT_RULE_DECISIONS)
    parser.add_argument("--gitlab-store", default=DEFAULT_GITLAB)
    parser.add_argument("--patterns-store", default=DEFAULT_PATTERNS)
    args, _ = parser.parse_known_args()
    return args


def install_config() -> argparse.Namespace:
    """Remember CLI paths in session state once, so every page sees the same config."""
    args = parse_cli_args()
    st.session_state.setdefault("cfg_input", args.input)
    st.session_state.setdefault("cfg_decisions", args.decisions)
    st.session_state.setdefault("cfg_rules", args.rules)
    st.session_state.setdefault("cfg_rule_decisions", args.rule_decisions)
    st.session_state.setdefault("cfg_gitlab", args.gitlab_store)
    st.session_state.setdefault("cfg_patterns", args.patterns_store)
    return args


def widget_value(key: str, default):
    if key in st.session_state:
        return {}
    return {"value": default}


def esc(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def content_to_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def label_pill(label: str) -> str:
    lab = (label or "unknown").lower()
    css = {"good": "pill-good", "maybe": "pill-maybe", "bad": "pill-bad"}.get(lab, "pill-undecided")
    return f'<span class="pill {css}">{esc(lab)}</span>'


def status_pill(status: str) -> str:
    name = (status or "undecided").lower()
    css = {
        "keep": "pill-keep",
        "discard": "pill-discard",
        "approved": "pill-approved",
        "rejected": "pill-rejected",
        "pending": "pill-pending",
    }.get(name, "pill-undecided")
    return f'<span class="pill {css}">{esc(name)}</span>'


def render_message(index: int, message: dict) -> None:
    role = (message.get("role") or "?").lower()
    css = {
        "user": "msg-user",
        "assistant": "msg-assistant",
        "tool": "msg-tool",
        "system": "msg-system",
    }.get(role, "msg-assistant")
    body = content_to_str(message.get("content"))
    shown = body if len(body) <= 6000 else body[:6000] + "\n… [truncated]"
    tool_bits = ""
    if role == "assistant" and message.get("tool_calls"):
        names = []
        for call in message["tool_calls"]:
            if isinstance(call, dict):
                fn = call.get("function") or {}
                names.append(fn.get("name") or call.get("name") or "tool")
        chips = "".join(f'<span class="tool-chip">{esc(name)}</span>' for name in names)
        tool_bits = f"<div style='margin:6px 0'>{chips}</div>"
    meta = ""
    if role == "tool":
        meta = f'<div class="msg-meta">tool_call_id: {esc(str(message.get("tool_call_id", "")))}</div>'
    st.markdown(
        f'<div class="msg {css}"><div class="msg-role">{index}. {esc(role)}</div>{tool_bits}'
        f'<div class="msg-body">{esc(shown)}</div>{meta}</div>',
        unsafe_allow_html=True,
    )


def queue_session_pos(index: int) -> None:
    st.session_state.idx = index
    st.session_state.pending_session_pos = index + 1


def open_evidence(ref: str) -> None:
    """Switch to the OpenCode session or the GitLab MR that this id points at."""
    from scripts.ingest.catalog import Catalog

    item = Catalog(
        st.session_state.get("input_path") or st.session_state.get("cfg_input", DEFAULT_INPUT),
        st.session_state.get("cfg_gitlab", DEFAULT_GITLAB),
    ).resolve(ref)
    if item is None:
        st.warning(f"Нет источника для {ref}")
        return
    if item.source == "gitlab":
        st.session_state.gitlab_focus = item.locator
        st.switch_page("pages/gitlab.py")
        return
    st.session_state.jump_sid = item.locator.get("session_id") or item.item_id
    st.switch_page("pages/sessions.py")
