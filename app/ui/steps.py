from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from app.core.ml import detect_task_type
from app.core.state import clear_all_state, reset_downstream
from app.core.workbench import (
    clear_persisted_dataframe,
    generate_eda_report,
    persist_training_artifacts,
    run_training,
    save_uploaded_dataframe,
)
from app.ui.header import render_data_kpis, resolve_task_choice


def _require_dataset_and_target() -> tuple[pd.DataFrame | None, str | None, bool]:
    df = st.session_state.df
    target_col = st.session_state.target_col
    ready = df is not None and bool(target_col)
    return df, target_col, ready


def render_upload_step() -> None:
    st.subheader("1) Upload")
    st.caption("Upload a CSV, choose a target, and persist source data to /data.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")

    col_a, col_b = st.columns([1, 1])
    if col_a.button("Save Uploaded Dataset", type="primary", disabled=uploaded is None):
        try:
            with st.spinner("Saving dataset to /data...", show_time=True):
                df = save_uploaded_dataframe(uploaded)
            st.session_state.df = df
            st.session_state.target_col = None
            st.session_state.task_type = None
            reset_downstream()
            st.success("Dataset saved to /data/sourcedata.csv")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

    if col_b.button("Clear Dataset", type="secondary"):
        clear_all_state()
        clear_persisted_dataframe()
        st.success("Dataset and downstream outputs cleared.")

    df = st.session_state.df
    if df is None:
        st.info("No dataset loaded yet.")
        return

    target = st.selectbox(
        "Select target column",
        options=list(df.columns),
        index=list(df.columns).index(st.session_state.target_col)
        if st.session_state.target_col in df.columns
        else len(df.columns) - 1,
        key="target_select",
    )
    st.session_state.target_col = target

    auto_task = detect_task_type(df, target)
    current_task = resolve_task_choice(df, target, st.session_state.task_type)
    task_index = 0 if current_task == "classification" else 1
    selected_task = st.radio(
        "Task type",
        ["classification", "regression"],
        index=task_index,
        horizontal=True,
        help=f"Auto-detected from target: {auto_task}",
    )
    st.session_state.task_type = selected_task

    render_data_kpis(df, target)
    st.markdown("### Preview")
    st.dataframe(df.head(200), use_container_width=True)



def render_eda_step() -> None:
    st.subheader("2) EDA")
    st.caption("Generate an HTML profile report and save it under /data/reports.")

    df, _target_col, ready = _require_dataset_and_target()
    if not ready:
        st.warning("Upload a dataset and select a target in step 1 before running EDA.")

    run_eda = st.button("Generate EDA Report", type="primary", disabled=not ready)
    if run_eda and df is not None:
        try:
            with st.status("Generating EDA report...", expanded=True) as status:
                status.write("Profiling dataframe structure and distributions")
                report_path = generate_eda_report(df)
                status.write(f"Saved report to: {report_path}")
                status.update(label="EDA report complete", state="complete")
            st.session_state.eda_report_path = str(report_path)
            st.success(f"EDA report saved: {report_path}")
        except Exception as exc:
            st.error(f"EDA generation failed: {exc}")

    report_path = st.session_state.eda_report_path
    if not report_path:
        st.info("No report generated yet.")
        return

    path_obj = Path(report_path)
    if not path_obj.exists():
        st.error("Saved EDA report path no longer exists on disk.")
        return

    st.markdown(f"**Report file**: `{path_obj}`")
    report_html = path_obj.read_text(encoding="utf-8")
    st.components.v1.html(report_html, height=900, scrolling=True)



def render_modelling_step() -> None:
    st.subheader("3) Modelling")
    st.caption("Run cross-validated model comparison and keep outputs in /data/artifacts.")

    df, target_col, ready = _require_dataset_and_target()
    if not ready:
        st.warning("Upload a dataset and select a target in step 1 before training.")

    train_clicked = st.button("Train and Compare", type="primary", disabled=not ready)
    if train_clicked and df is not None and target_col is not None:
        try:
            task_type = st.session_state.task_type or detect_task_type(df, target_col)
            st.session_state.task_type = task_type
            with st.status("Training models...", expanded=True) as status:
                status.write(f"Task type: {task_type}")
                status.write("Running cross-validation and model comparison")
                artifacts = run_training(df, target_col, task_type)
                status.write("Persisting artifacts to /data/artifacts")
                artifact_paths = persist_training_artifacts(artifacts)
                status.update(label="Training complete", state="complete")
            st.session_state.artifacts = artifacts
            st.session_state.artifact_paths = {k: str(v) for k, v in artifact_paths.items()}
            st.success(f"Training complete ({task_type}).")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    artifacts = st.session_state.artifacts
    if artifacts is None:
        st.info("No model run yet.")
        return

    st.markdown("### Best Model")
    st.write(f"**Selected model:** {artifacts.best_model_name}")
    st.markdown("### Setup")
    st.dataframe(artifacts.setup_df, use_container_width=True)
    st.markdown("### Leaderboard")
    st.dataframe(artifacts.leaderboard_df, use_container_width=True)



def render_validation_step() -> None:
    st.subheader("4) Validation")
    st.caption("Inspect holdout metrics and diagnostics for the selected model.")

    artifacts = st.session_state.artifacts
    if artifacts is None:
        st.warning("Run step 3 (Modelling) first.")
        return

    st.markdown("### Holdout Metrics")
    st.dataframe(artifacts.evaluation_df, use_container_width=True)

    payload = artifacts.evaluation_payload
    if artifacts.task_type == "classification":
        y_true = payload["y_true"]
        y_pred = payload["y_pred"]
        labels = payload["labels"]

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Classification Report")
        st.code(payload["classification_report"])
    else:
        y_true = payload["y_true"]
        y_pred = payload["y_pred"]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_true, y_pred, alpha=0.7)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        ax.set_title("Predicted vs Actual")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)
        plt.close(fig)



def _download_file_button(label: str, path: Path, mime: str) -> None:
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        use_container_width=True,
    )



def render_export_step() -> None:
    st.subheader("5) Export")
    st.caption("Download persisted artifacts from /data/artifacts.")

    artifacts = st.session_state.artifacts
    artifact_paths = st.session_state.artifact_paths
    if artifacts is None or not artifact_paths:
        st.warning("Run step 3 (Modelling) first.")
        return

    model_path = Path(artifact_paths["model"])
    leaderboard_path = Path(artifact_paths["leaderboard"])
    setup_path = Path(artifact_paths["setup"])
    evaluation_path = Path(artifact_paths["evaluation"])

    missing_files = [
        str(p)
        for p in [model_path, leaderboard_path, setup_path, evaluation_path]
        if not p.exists()
    ]
    if missing_files:
        st.error(f"Missing artifact files: {', '.join(missing_files)}")
        return

    c1, c2 = st.columns(2)
    with c1:
        _download_file_button("Download Model (.pkl)", model_path, "application/octet-stream")
        _download_file_button("Download Leaderboard (.csv)", leaderboard_path, "text/csv")
    with c2:
        _download_file_button("Download Setup (.csv)", setup_path, "text/csv")
        _download_file_button("Download Evaluation (.csv)", evaluation_path, "text/csv")

    st.markdown("### Saved Artifact Paths")
    st.code(
        "\n".join(
            [
                str(model_path),
                str(leaderboard_path),
                str(setup_path),
                str(evaluation_path),
            ]
        )
    )
