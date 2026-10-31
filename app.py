import streamlit as st

from postprime_generator import (
    MACRO_MODE,
    MICRO_MODE,
    render_postprime_generator,
)

st.set_page_config(
    page_title="PostPrime自動投稿生成",
    page_icon="📈",
    layout="wide",
)

selected_mode = st.sidebar.selectbox(
    "投稿モード",
    options=(MACRO_MODE, MICRO_MODE),
    key="postprime-app-mode",
)
render_postprime_generator(selected_mode)
