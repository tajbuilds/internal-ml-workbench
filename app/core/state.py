import streamlit as st

from app.core.config import APP_DATA_DIR
from app.core.workbench import (
    list_datasets,
    load_dataset_by_id,
    migrate_legacy_dataset_if_present,
)

STATE_DEFAULTS = {
    "df": None,
    "target_col": None,
    "task_type": None,
    "eda_report_path": None,
    "selected_eda_report_path": None,
    "artifacts": None,
    "artifact_paths": None,
    "datasets": [],
    "active_dataset_id": None,
    "run_dataset_id": None,
    "dataset_settings": {},
}


def init_state() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for key, default in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    legacy_id = migrate_legacy_dataset_if_present()
    st.session_state.datasets = list_datasets()

    if st.session_state.active_dataset_id is None:
        if st.session_state.datasets:
            st.session_state.active_dataset_id = st.session_state.datasets[0]["id"]
        elif legacy_id:
            st.session_state.active_dataset_id = legacy_id

    active_id = st.session_state.active_dataset_id
    if active_id:
        try:
            st.session_state.df = load_dataset_by_id(active_id)
        except Exception:
            st.session_state.df = None


def load_active_dataset() -> None:
    active_id = st.session_state.active_dataset_id
    if not active_id:
        st.session_state.df = None
        return

    st.session_state.df = load_dataset_by_id(active_id)
    settings = st.session_state.dataset_settings.get(active_id, {})
    st.session_state.target_col = settings.get("target_col")
    st.session_state.task_type = settings.get("task_type")


def persist_active_settings() -> None:
    active_id = st.session_state.active_dataset_id
    if not active_id:
        return

    st.session_state.dataset_settings[active_id] = {
        "target_col": st.session_state.target_col,
        "task_type": st.session_state.task_type,
    }


def refresh_datasets() -> None:
    st.session_state.datasets = list_datasets()


def reset_downstream() -> None:
    st.session_state.eda_report_path = None
    st.session_state.selected_eda_report_path = None
    st.session_state.artifacts = None
    st.session_state.artifact_paths = None
    st.session_state.run_dataset_id = None


def clear_all_state() -> None:
    st.session_state.df = None
    st.session_state.target_col = None
    st.session_state.task_type = None
    st.session_state.active_dataset_id = None
    st.session_state.datasets = []
    st.session_state.dataset_settings = {}
    reset_downstream()

