import pandas as pd
import streamlit as st

from app.core.ml import detect_task_type
from app.core.workbench import dataset_kpis


def render_header(
    df: pd.DataFrame | None,
    target_col: str | None,
    task_type: str | None,
    active_dataset_name: str | None,
) -> None:
    loaded = df is not None
    rows_cols = "0 x 0"
    if df is not None:
        rows_cols = f"{len(df):,} x {len(df.columns):,}"

    selected_target = target_col if target_col else "-"
    selected_task = task_type if task_type else "-"
    dataset_name = active_dataset_name if active_dataset_name else "-"

    st.markdown(
        """
        <div class="imw-header">
            <h2>AutoML Dashboard</h2>
            <p>Workspace -> Datasets -> Preparation -> EDA -> Modelling -> Validation -> Export</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dataset Loaded", "Yes" if loaded else "No")
    c2.metric("Active Dataset", dataset_name)
    c3.metric("Rows x Cols", rows_cols)
    c4.metric("Selected Target", selected_target)
    c5.metric("Task", selected_task)



def render_data_kpis(df: pd.DataFrame | None, target_col: str | None) -> None:
    if df is None:
        return

    if target_col and target_col in df.columns:
        inferred_target_type = str(df[target_col].dtype)
    else:
        inferred_target_type = "-"

    values = dataset_kpis(df, target_col)
    if inferred_target_type != "-":
        values["target_type"] = inferred_target_type

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", values["rows"])
    c2.metric("Cols", values["cols"])
    c3.metric("Missing %", values["missing_pct"])
    c4.metric("Duplicates", values["duplicates"])
    c5.metric("Target Type", values["target_type"])



def resolve_task_choice(df: pd.DataFrame, target_col: str, current_task: str | None) -> str:
    auto_task = detect_task_type(df, target_col)
    if current_task in {"classification", "regression"}:
        return current_task
    return auto_task
