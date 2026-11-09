#!/usr/bin/env python3
"""One Streamlit app for sessions, GitLab reviews, proposals, export, and monitoring.

Launch:
    streamlit run viewer/app.py
"""

from __future__ import annotations

import streamlit as st

from shared import CSS, install_config

st.set_page_config(
    page_title="finetune-curator",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)
install_config()

pages = [
    st.Page("pages/sessions.py", title="Сессии", default=True),
    st.Page("pages/gitlab.py", title="GitLab"),
    st.Page("pages/proposals.py", title="Правила"),
    st.Page("pages/export_page.py", title="Экспорт"),
    st.Page("pages/monitoring.py", title="Мониторинг"),
]
navigation = st.navigation(pages)
navigation.run()
