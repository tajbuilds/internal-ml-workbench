import streamlit as st

from app.core.state import init_state
from app.pages.main_pages import (
    auto_ml_page,
    data_profiling_page,
    data_upload_page,
    developer_page,
    evaluation_page,
    home_page,
    model_download_page,
    render_sidebar,
)


def main() -> None:
    st.set_page_config(page_title="AutoML Workbench", layout="wide")
    st.title("AutoML Workbench")

    init_state()
    selected = render_sidebar()

    if selected == "Home":
        home_page()
    elif selected == "Upload Data":
        data_upload_page()
    elif selected == "Data Profiling":
        data_profiling_page()
    elif selected == "Auto ML":
        auto_ml_page()
    elif selected == "Evaluation":
        evaluation_page()
    elif selected == "Model Download":
        model_download_page()
    else:
        developer_page()
