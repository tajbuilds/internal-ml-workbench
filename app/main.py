import streamlit as st

from app.core.state import init_state, load_active_dataset
from app.ui.header import render_header
from app.ui.steps import (
    render_datasets_step,
    render_eda_step,
    render_export_step,
    render_modelling_step,
    render_validation_step,
    render_workspace_step,
)
from app.ui.theme import apply_theme
from app.ui.top_nav import render_top_nav, render_workspace_context

STEP_RENDERERS = {
    "Workspace": render_workspace_step,
    "Datasets": render_datasets_step,
    "EDA": render_eda_step,
    "Modelling": render_modelling_step,
    "Validation": render_validation_step,
    "Export": render_export_step,
}


def _active_dataset_name() -> str | None:
    active_id = st.session_state.get("active_dataset_id")
    for item in st.session_state.get("datasets", []):
        if item.get("id") == active_id:
            return item.get("name")
    return None


def main() -> None:
    st.set_page_config(
        page_title="AutoML Workbench",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_state()
    apply_theme()

    render_workspace_context()
    selected_step = render_top_nav()

    if st.session_state.get("active_dataset_id"):
        try:
            load_active_dataset()
        except Exception as exc:
            st.error(f"Failed to load active dataset: {exc}")

    render_header(
        st.session_state.df,
        st.session_state.target_col,
        st.session_state.task_type,
        _active_dataset_name(),
    )

    renderer = STEP_RENDERERS.get(selected_step, render_workspace_step)
    renderer()


if __name__ == "__main__":
    main()
