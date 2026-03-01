import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix
from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport

from app.core.config import DATA_FILE, DEVELOPER_IMAGE_URL, resolve_hero_image
from app.core.ml import detect_task_type, train_and_compare
from app.core.state import get_df, reset_training_outputs
from app.utils.theme import apply_theme, flow_hint, hero, kpi_card


@st.cache_data(show_spinner=False)
def build_profile_html(df: pd.DataFrame) -> str:
    report = ProfileReport(df, minimal=True, explorative=True)
    return report.to_html()


def _render_hero_image() -> None:
    local_img = resolve_hero_image()
    if local_img is not None:
        st.image(str(local_img), use_container_width=True)
    else:
        st.info("Place a hero image in app/assets as hero.jpg, hero.png, hero.jpeg, or hero.webp")


def home_page() -> None:
    hero(
        "AutoML Workbench",
        "A self-hosted lab for quick tabular model baselining, evaluation, and export.",
    )
    flow_hint()
    _render_hero_image()

    df = get_df()
    artifacts = st.session_state.artifacts

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Dataset", "Loaded" if df is not None else "Not loaded")
    with c2:
        if df is None:
            kpi_card("Rows x Cols", "0 x 0")
        else:
            kpi_card("Rows x Cols", f"{len(df):,} x {len(df.columns):,}")
    with c3:
        status = "Ready" if artifacts is not None else "Pending"
        kpi_card("Model", status)

    st.markdown("### Quick Actions")
    st.info(
        "Use the sidebar flow to upload data, profile it, train models, "
        "and export the best pipeline."
    )


def data_page() -> None:
    hero("Data Intake", "Upload a CSV, validate shape, and persist it for repeatable runs.")
    flow_hint()

    data = st.file_uploader("Choose CSV", type=["csv"])

    col1, col2 = st.columns([1, 1])
    persist_to_disk = col1.checkbox("Persist dataset to disk", value=True)
    clear_data = col2.button("Clear dataset", type="secondary")

    if clear_data:
        st.session_state.df = None
        reset_training_outputs()
        if DATA_FILE.exists():
            DATA_FILE.unlink(missing_ok=True)
        st.success("Dataset cleared.")

    if data is not None:
        try:
            df = pd.read_csv(data, index_col=None)
            st.session_state.df = df
            reset_training_outputs()
            if persist_to_disk:
                df.to_csv(DATA_FILE, index=False)
            st.success("Dataset loaded.")
        except Exception as exc:
            st.error(f"Could not parse CSV: {exc}")
            return

    df = get_df()
    if df is None:
        st.warning("No dataset loaded yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

    st.markdown("### Preview")
    st.dataframe(df.head(200), use_container_width=True)

    st.markdown("### Schema")
    schema_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "null_count": [int(df[c].isna().sum()) for c in df.columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
    st.dataframe(schema_df, use_container_width=True)


def profiling_page() -> None:
    hero(
        "Profiling",
        "Generate an exploratory profile report to inspect data quality and feature behavior.",
    )
    flow_hint()

    df = get_df()
    if df is None:
        st.warning("Load a dataset first on the Data page.")
        return

    if st.button("Generate Profile", type="primary"):
        with st.spinner("Building profile report..."):
            profile_html = build_profile_html(df)
        st.components.v1.html(profile_html, height=900, scrolling=True)


def training_page() -> None:
    hero("Model Training", "Run cross-validated model comparison and select a best baseline.")
    flow_hint()

    df = get_df()
    if df is None:
        st.warning("Load a dataset first on the Data page.")
        return

    target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
    auto_task = detect_task_type(df, target)
    task_type = st.radio(
        "Task type",
        options=["classification", "regression"],
        index=0 if auto_task == "classification" else 1,
        horizontal=True,
        help=f"Auto-detected: {auto_task}",
    )

    if st.button("Train and Compare", type="primary"):
        try:
            with st.spinner("Training and cross-validating models..."):
                st.session_state.artifacts = train_and_compare(df, target, task_type)
            st.success(f"Training complete ({task_type}).")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    artifacts = st.session_state.artifacts
    if artifacts is not None:
        st.markdown("### Run Summary")
        st.dataframe(artifacts.setup_df, use_container_width=True)

        st.markdown("### Leaderboard")
        st.dataframe(artifacts.leaderboard_df, use_container_width=True)


def evaluation_page() -> None:
    hero("Evaluation", "Inspect holdout performance before exporting the selected model.")
    flow_hint()

    artifacts = st.session_state.artifacts
    if artifacts is None:
        st.warning("Train models first on the Training page.")
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


def export_page() -> None:
    hero("Export", "Download the fitted model pipeline for reuse in downstream systems.")
    flow_hint()

    artifacts = st.session_state.artifacts
    if artifacts is None:
        st.warning("Train models first on the Training page.")
        return

    st.download_button(
        "Download Model (.pkl)",
        data=artifacts.model_bytes,
        file_name="auto_trained_model.pkl",
        mime="application/octet-stream",
    )
    st.caption("The exported file is a fitted scikit-learn pipeline.")


def developer_page() -> None:
    hero("About", "Project lineage and maintainer context.")
    st.image(DEVELOPER_IMAGE_URL, use_container_width=True)
    st.write(
        "This app was originally developed by Elvis Darko and modernized for "
        "a stable local-first workflow."
    )


def render_sidebar() -> str:
    apply_theme()

    css_style = {
        "icon": {"color": "#d7f4ec", "font-size": "18px"},
        "container": {"padding": "0!important"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "left",
            "margin": "4px 0px",
            "--hover-color": "#1f4e46",
            "border-radius": "8px",
        },
        "nav-link-selected": {"background-color": "#0b8f77", "color": "white"},
    }

    with st.sidebar:
        st.markdown("## ML Workbench")
        st.caption("Self-hosted tabular modeling workspace")
        return option_menu(
            menu_title=None,
            options=[
                "Home",
                "Data",
                "Profiling",
                "Training",
                "Evaluation",
                "Export",
                "About",
            ],
            icons=[
                "house",
                "database",
                "bar-chart-line",
                "cpu",
                "clipboard-data",
                "download",
                "person",
            ],
            styles=css_style,
        )



