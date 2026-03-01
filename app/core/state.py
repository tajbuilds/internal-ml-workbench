import pandas as pd
import streamlit as st

from app.core.config import APP_DATA_DIR, DATA_FILE

STATE_DEFAULTS = {
    "df": None,
    "target_col": None,
    "task_type": None,
    "eda_report_path": None,
    "artifacts": None,
    "artifact_paths": None,
}


def init_state() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for key, default in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.df is None and DATA_FILE.exists():
        st.session_state.df = pd.read_csv(DATA_FILE, index_col=None)


def reset_downstream() -> None:
    st.session_state.eda_report_path = None
    st.session_state.artifacts = None
    st.session_state.artifact_paths = None


def clear_all_state() -> None:
    st.session_state.df = None
    st.session_state.target_col = None
    st.session_state.task_type = None
    reset_downstream()


def get_df() -> pd.DataFrame | None:
    return st.session_state.df
