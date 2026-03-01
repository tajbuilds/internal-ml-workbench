import pandas as pd
import streamlit as st

from app.core.config import APP_DATA_DIR, DATA_FILE


def init_state() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if "df" not in st.session_state:
        st.session_state.df = None
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = None

    if st.session_state.df is None and DATA_FILE.exists():
        st.session_state.df = pd.read_csv(DATA_FILE, index_col=None)


def reset_training_outputs() -> None:
    st.session_state.artifacts = None


def get_df() -> pd.DataFrame | None:
    return st.session_state.df
