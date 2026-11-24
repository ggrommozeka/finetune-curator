"""Before/after review card for a GitLab SFT example."""

from __future__ import annotations

import streamlit as st


def render_gitlab_panel(rec: dict) -> None:
    gitlab = rec.get("gitlab") or {}
    if not gitlab:
        return
    link = gitlab.get("link") or "—"
    confidence = gitlab.get("confidence")
    confidence_label = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—"
    file_label = gitlab.get("file") or "(merge request)"
    st.markdown("#### Review")
    st.caption(f"{file_label} · {link} · confidence {confidence_label}")
    comment = gitlab.get("comment") or ""
    if comment:
        st.text(comment)
    web_url = gitlab.get("web_url")
    if web_url:
        st.caption(str(web_url))
    before, after = st.columns(2)
    before.markdown("**Before**")
    before.code(gitlab.get("before") or "—", language="text")
    after.markdown("**After**")
    after.code(gitlab.get("after") or "—", language="text")
    diff = gitlab.get("diff") or ""
    if diff:
        with st.expander("Diff"):
            st.code(diff, language="diff")
