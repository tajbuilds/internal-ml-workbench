import streamlit as st

from app.core.state import init_state
from app.ui.header import render_header
from app.ui.sidebar import render_sidebar
from app.ui.steps import (
    render_eda_step,
    render_export_step,
    render_modelling_step,
    render_upload_step,
    render_validation_step,
)
from app.ui.theme import apply_theme

STEP_RENDERERS = {
    "Upload": render_upload_step,
    "EDA": render_eda_step,
    "Modelling": render_modelling_step,
    "Validation": render_validation_step,
    "Export": render_export_step,
}


def main() -> None:
    st.set_page_config(page_title="AutoML Workbench", layout="wide")

    init_state()
    apply_theme()

    selected_step = render_sidebar()

    render_header(
        st.session_state.df,
        st.session_state.target_col,
        st.session_state.task_type,
    )

    renderer = STEP_RENDERERS.get(selected_step, render_upload_step)
    renderer()


if __name__ == "__main__":
    main()
