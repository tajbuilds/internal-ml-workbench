from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix

from app.core.eda_report import generate_selected_eda_report
from app.core.ml import detect_task_type
from app.core.state import (
    load_active_dataset,
    persist_active_settings,
    refresh_datasets,
    reset_downstream,
)
from app.core.workbench import (
    delete_dataset,
    generate_eda_report,
    get_dataset_meta,
    persist_training_artifacts,
    run_training,
    save_uploaded_dataset,
)
from app.ui.header import render_data_kpis, resolve_task_choice


def _require_dataset_and_target() -> tuple[pd.DataFrame | None, str | None, str | None, bool]:
    df = st.session_state.df
    target_col = st.session_state.target_col
    dataset_id = st.session_state.active_dataset_id
    ready = df is not None and bool(target_col) and bool(dataset_id)
    return df, target_col, dataset_id, ready


def _sync_settings_to_active_dataset() -> None:
    persist_active_settings()


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number, "bool"]).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _numeric_cols(df)]


def _render_quality_summary(df: pd.DataFrame) -> None:
    st.markdown("### Data Quality Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")

    st.markdown("#### Schema")
    schema_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "null_count": [int(df[c].isna().sum()) for c in df.columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
    st.dataframe(schema_df, use_container_width=True)


def _render_missingness(df: pd.DataFrame, top_n: int) -> None:
    st.markdown("### Missingness")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values detected.")
        return

    missing = missing.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(missing.index.astype(str), missing.values)
    ax.set_title("Missing values by column")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)


def _render_distributions(df: pd.DataFrame, max_numeric: int, bins: int) -> None:
    st.markdown("### Numeric Distributions")
    numeric = _numeric_cols(df)
    if not numeric:
        st.info("No numeric columns available.")
        return

    cols = numeric[:max_numeric]
    n = len(cols)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, max(3, 2.4 * n)))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        series = df[col].dropna()
        ax.hist(series, bins=bins)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_outliers(df: pd.DataFrame, max_numeric: int) -> None:
    st.markdown("### Outlier Inspection (Boxplots)")
    numeric = _numeric_cols(df)
    if not numeric:
        st.info("No numeric columns available.")
        return

    cols = numeric[:max_numeric]
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(cols)), 4.5))
    data = [df[c].dropna().values for c in cols]
    ax.boxplot(data, labels=cols, showfliers=True)
    ax.set_title("Boxplots by numeric feature")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)


def _render_categorical_counts(df: pd.DataFrame, max_cols: int, max_levels: int) -> None:
    st.markdown("### Categorical Distributions")
    cat_cols = _categorical_cols(df)
    if not cat_cols:
        st.info("No categorical columns available.")
        return

    cols = cat_cols[:max_cols]
    for col in cols:
        vc = df[col].astype(str).value_counts(dropna=False).head(max_levels)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.bar(vc.index, vc.values)
        ax.set_title(f"Top categories: {col}")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close(fig)


def _render_correlations(df: pd.DataFrame, method: str) -> None:
    st.markdown("### Correlation Heatmap")
    numeric = _numeric_cols(df)
    if len(numeric) < 2:
        st.info("Need at least 2 numeric columns for correlation.")
        return

    corr = df[numeric].corr(method=method)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title(f"{method.title()} correlation")
    fig.colorbar(im, ax=ax)

    if len(corr.columns) <= 12:
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    st.pyplot(fig)
    plt.close(fig)


def _render_pairwise(df: pd.DataFrame, max_numeric: int, sample_rows: int) -> None:
    st.markdown("### Pairwise Relationships")
    numeric = _numeric_cols(df)
    if len(numeric) < 2:
        st.info("Need at least 2 numeric columns for pairwise view.")
        return

    cols = numeric[:max_numeric]
    sample_df = df[cols].dropna().head(sample_rows)
    if sample_df.empty:
        st.info("Not enough non-null numeric rows for pairwise plot.")
        return

    size = min(14, 2.2 * len(cols))
    fig = plt.figure(figsize=(size, size))
    scatter_matrix(sample_df, alpha=0.55, diagonal="hist", figsize=fig.get_size_inches())
    st.pyplot(fig)
    plt.close(fig)


def _render_target_analysis(df: pd.DataFrame, target_col: str, max_numeric: int) -> None:
    st.markdown("### Target Deep Dive")
    if target_col not in df.columns:
        st.info("Select a valid target in Workspace first.")
        return

    target = df[target_col]
    task = detect_task_type(df, target_col)
    st.caption(f"Auto-detected task: {task}")

    if task == "classification":
        vc = target.astype(str).value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(vc.index, vc.values)
        ax.set_title(f"Target class distribution: {target_col}")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(target.dropna(), bins=25)
        ax.set_title(f"Target distribution: {target_col}")
        st.pyplot(fig)
        plt.close(fig)

    numeric = [c for c in _numeric_cols(df) if c != target_col][:max_numeric]
    if not numeric:
        return

    fig, axes = plt.subplots(
        nrows=len(numeric),
        ncols=1,
        figsize=(10, max(3, 2.2 * len(numeric))),
    )
    if len(numeric) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric):
        x = df[col]
        y = target
        ax.scatter(x, y, alpha=0.45)
        ax.set_title(f"{col} vs {target_col}")
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_custom_plot(df: pd.DataFrame) -> None:
    st.markdown("### Custom Feature Plot")
    cols = list(df.columns)
    if len(cols) < 1:
        st.info("Dataset has no columns.")
        return

    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("X", cols, key="viz_x")
    y_options = ["<none>"] + cols
    y_col = c2.selectbox("Y", y_options, key="viz_y")
    plot_type = c3.selectbox("Plot", ["line", "scatter", "bar"], key="viz_plot_type")

    fig, ax = plt.subplots(figsize=(8.5, 4))
    if y_col == "<none>":
        vc = df[x_col].astype(str).value_counts().head(20)
        ax.bar(vc.index, vc.values)
        ax.set_title(f"Top values for {x_col}")
        ax.tick_params(axis="x", rotation=45)
    else:
        temp = df[[x_col, y_col]].dropna().head(2000)
        if plot_type == "line":
            ax.plot(temp[x_col], temp[y_col])
        elif plot_type == "bar":
            grouped = temp.groupby(x_col, dropna=False)[y_col].mean().head(30)
            ax.bar(grouped.index.astype(str), grouped.values)
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.scatter(temp[x_col], temp[y_col], alpha=0.5)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{plot_type.title()} plot: {x_col} vs {y_col}")

    st.pyplot(fig)
    plt.close(fig)


def render_workspace_step() -> None:
    st.subheader("Workspace")
    st.caption("Upload and manage the active dataset context for downstream steps.")

    upload_name = st.text_input("Dataset name (optional)", placeholder="customer-churn")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")

    col_a, col_b = st.columns([1, 1])
    if col_a.button("Save Uploaded Dataset", type="primary", disabled=uploaded is None):
        try:
            with st.spinner("Saving dataset to /data/datasets...", show_time=True):
                df, dataset_id = save_uploaded_dataset(uploaded, upload_name or None)
            refresh_datasets()
            st.session_state.active_dataset_id = dataset_id
            st.session_state.df = df
            st.session_state.target_col = None
            st.session_state.task_type = None
            reset_downstream()
            st.success("Dataset saved and set as active.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

    reload_disabled = not st.session_state.active_dataset_id
    if col_b.button("Reload Active Dataset", type="secondary", disabled=reload_disabled):
        try:
            load_active_dataset()
            reset_downstream()
            st.success("Active dataset reloaded.")
        except Exception as exc:
            st.error(f"Reload failed: {exc}")

    df = st.session_state.df
    if df is None:
        st.info("No active dataset loaded yet.")
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
    _sync_settings_to_active_dataset()

    render_data_kpis(df, target)
    st.markdown("### Preview")
    st.dataframe(df.head(200), use_container_width=True)


def render_datasets_step() -> None:
    st.subheader("Datasets")
    st.caption("Browse all stored datasets and switch active context.")

    datasets = st.session_state.get("datasets", [])
    if not datasets:
        st.info("No stored datasets yet. Upload one in Workspace.")
        return

    rows: list[dict[str, str]] = []
    for d in datasets:
        rows.append(
            {
                "id": d["id"],
                "name": d["name"],
                "shape": f"{d['rows']} x {d['cols']}",
                "updated": d.get("updated_at", "-"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    ids = [d["id"] for d in datasets]
    labels = [f"{d['name']} ({d['rows']}x{d['cols']})" for d in datasets]
    selected_label = st.selectbox("Dataset", labels, key="dataset_admin_select")
    selected_id = ids[labels.index(selected_label)]

    c1, c2 = st.columns([1, 1])
    if c1.button("Set As Active", type="primary"):
        st.session_state.active_dataset_id = selected_id
        load_active_dataset()
        reset_downstream()
        st.success("Active dataset changed.")

    if c2.button("Delete Dataset", type="secondary"):
        if selected_id == st.session_state.get("run_dataset_id"):
            st.warning("This dataset has the latest model run in session. Run another model first.")
        else:
            delete_dataset(selected_id)
            refresh_datasets()
            if st.session_state.active_dataset_id == selected_id:
                st.session_state.active_dataset_id = None
                st.session_state.df = None
                st.session_state.target_col = None
                st.session_state.task_type = None
                reset_downstream()
                if st.session_state.datasets:
                    st.session_state.active_dataset_id = st.session_state.datasets[0]["id"]
                    load_active_dataset()
            st.success("Dataset deleted.")


def render_eda_step() -> None:
    st.subheader("EDA & Visualization Workbench")
    st.caption("Run selected analyses from one page, plus optional profile report export.")

    df, target_col, dataset_id, ready = _require_dataset_and_target()
    if not ready:
        st.warning("Set an active dataset and target in Workspace before running EDA.")
        return

    assert df is not None
    assert target_col is not None
    assert dataset_id is not None

    module_options = [
        "Data Quality Summary",
        "Missingness",
        "Numeric Distributions",
        "Outlier Boxplots",
        "Categorical Distributions",
        "Correlation Heatmap",
        "Pairwise Relationships",
        "Target Deep Dive",
        "Custom Feature Plot",
    ]

    with st.expander("Analysis Options", expanded=True):
        selected_modules = st.multiselect(
            "Select analyses to run",
            options=module_options,
            default=module_options[:7],
        )

        c1, c2, c3 = st.columns(3)
        corr_method = c1.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
        max_numeric = c2.slider("Max numeric columns", min_value=2, max_value=12, value=6)
        bins = c3.slider("Histogram bins", min_value=10, max_value=80, value=30)

        c4, c5 = st.columns(2)
        max_categories = c4.slider(
            "Top categories per feature",
            min_value=5,
            max_value=40,
            value=15,
        )
        sample_rows = c5.slider("Pairwise sample rows", min_value=100, max_value=3000, value=1000)

    if not selected_modules:
        st.info("Select at least one analysis module.")
        return

    with st.status("Running selected analyses...", expanded=False) as status:
        if "Data Quality Summary" in selected_modules:
            status.write("Data quality summary")
            _render_quality_summary(df)

        if "Missingness" in selected_modules:
            status.write("Missingness")
            _render_missingness(df, top_n=max_numeric * 2)

        if "Numeric Distributions" in selected_modules:
            status.write("Numeric distributions")
            _render_distributions(df, max_numeric=max_numeric, bins=bins)

        if "Outlier Boxplots" in selected_modules:
            status.write("Outlier boxplots")
            _render_outliers(df, max_numeric=max_numeric)

        if "Categorical Distributions" in selected_modules:
            status.write("Categorical distributions")
            _render_categorical_counts(
                df,
                max_cols=max(2, max_numeric // 2),
                max_levels=max_categories,
            )

        if "Correlation Heatmap" in selected_modules:
            status.write("Correlation heatmap")
            _render_correlations(df, method=corr_method)

        if "Pairwise Relationships" in selected_modules:
            status.write("Pairwise relationships")
            _render_pairwise(df, max_numeric=min(max_numeric, 6), sample_rows=sample_rows)

        if "Target Deep Dive" in selected_modules:
            status.write("Target deep dive")
            _render_target_analysis(df, target_col=target_col, max_numeric=max(2, max_numeric // 2))

        if "Custom Feature Plot" in selected_modules:
            status.write("Custom feature plot")
            _render_custom_plot(df)

        status.update(label="Analyses complete", state="complete")

    try:
        selected_report_path = generate_selected_eda_report(
            df=df,
            dataset_id=dataset_id,
            target_col=target_col,
            selected_modules=selected_modules,
            options={
                "corr_method": corr_method,
                "max_numeric": max_numeric,
                "bins": bins,
                "max_categories": max_categories,
                "sample_rows": sample_rows,
                "top_missing": max_numeric * 2,
                "viz_x": st.session_state.get("viz_x"),
                "viz_y": st.session_state.get("viz_y"),
                "viz_plot_type": st.session_state.get("viz_plot_type", "scatter"),
            },
        )
        st.session_state.selected_eda_report_path = str(selected_report_path)
        st.success(f"Selected analyses report saved: {selected_report_path}")
    except Exception as exc:
        st.error(f"Could not build selected analyses report: {exc}")

    selected_report = st.session_state.selected_eda_report_path
    if selected_report:
        selected_path = Path(selected_report)
        if selected_path.exists():
            with st.expander("Open selected analyses report", expanded=False):
                st.components.v1.html(
                    selected_path.read_text(encoding="utf-8"),
                    height=900,
                    scrolling=True,
                )

    st.markdown("---")
    st.markdown("### Profile Report (Optional)")
    st.caption("Generate full HTML profile report under /data/reports/<dataset_id>.")

    run_eda_report = st.button("Generate Full Profile Report", type="primary")
    if run_eda_report:
        try:
            with st.status("Generating full profile report...", expanded=True) as status:
                status.write("This can take time on larger datasets")
                report_path = generate_eda_report(df, dataset_id)
                status.write(f"Saved report to: {report_path}")
                status.update(label="Profile report complete", state="complete")
            st.session_state.eda_report_path = str(report_path)
            st.success(f"EDA report saved: {report_path}")
        except Exception as exc:
            st.error(f"EDA report generation failed: {exc}")

    report_path = st.session_state.eda_report_path
    if report_path:
        path_obj = Path(report_path)
        if path_obj.exists():
            st.markdown(f"**Latest report file**: `{path_obj}`")
            with st.expander("Open latest report", expanded=False):
                report_html = path_obj.read_text(encoding="utf-8")
                st.caption("Inline preview below. If it appears blank, use download fallback.")
                st.components.v1.html(report_html, height=1000, scrolling=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Download latest profile report (.html)",
                        data=report_html.encode("utf-8"),
                        file_name=path_obj.name,
                        mime="text/html",
                        use_container_width=True,
                    )
                with c2:
                    selected_report = st.session_state.selected_eda_report_path
                    if selected_report and Path(selected_report).exists():
                        st.download_button(
                            "Download selected analyses report (.html)",
                            data=Path(selected_report).read_bytes(),
                            file_name=Path(selected_report).name,
                            mime="text/html",
                            use_container_width=True,
                        )


def render_modelling_step() -> None:
    st.subheader("Modelling")
    st.caption("Run cross-validated model comparison and persist outputs per dataset.")

    df, target_col, dataset_id, ready = _require_dataset_and_target()
    if not ready:
        st.warning("Set an active dataset and target in Workspace before training.")

    train_clicked = st.button("Train and Compare", type="primary", disabled=not ready)
    if train_clicked and df is not None and target_col is not None and dataset_id is not None:
        try:
            task_type = st.session_state.task_type or detect_task_type(df, target_col)
            st.session_state.task_type = task_type
            _sync_settings_to_active_dataset()
            with st.status("Training models...", expanded=True) as status:
                status.write(f"Task type: {task_type}")
                status.write("Running cross-validation and model comparison")
                artifacts = run_training(df, target_col, task_type)
                status.write(f"Persisting artifacts to /data/artifacts/{dataset_id}")
                artifact_paths = persist_training_artifacts(artifacts, dataset_id)
                status.update(label="Training complete", state="complete")
            st.session_state.artifacts = artifacts
            st.session_state.artifact_paths = {k: str(v) for k, v in artifact_paths.items()}
            st.session_state.run_dataset_id = dataset_id
            st.success(f"Training complete ({task_type}).")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    artifacts = st.session_state.artifacts
    if artifacts is None:
        st.info("No model run yet.")
        return

    if st.session_state.get("run_dataset_id") != st.session_state.get("active_dataset_id"):
        st.warning(
            "Latest model run belongs to a different dataset. "
            "Train again for current active dataset."
        )

    st.markdown("### Best Model")
    st.write(f"**Selected model:** {artifacts.best_model_name}")
    st.markdown("### Setup")
    st.dataframe(artifacts.setup_df, use_container_width=True)
    st.markdown("### Leaderboard")
    st.dataframe(artifacts.leaderboard_df, use_container_width=True)


def render_validation_step() -> None:
    st.subheader("Validation")
    st.caption("Inspect holdout metrics and diagnostics for the selected model run.")

    artifacts = st.session_state.artifacts
    if artifacts is None:
        st.warning("Run Modelling first.")
        return

    run_dataset_id = st.session_state.get("run_dataset_id")
    meta = get_dataset_meta(run_dataset_id) if run_dataset_id else None
    if meta:
        st.caption(f"Showing validation for dataset: {meta['name']}")

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
    st.subheader("Export")
    st.caption("Download persisted artifacts for the latest model run.")

    artifacts = st.session_state.artifacts
    artifact_paths = st.session_state.artifact_paths
    if artifacts is None or not artifact_paths:
        st.warning("Run Modelling first.")
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




