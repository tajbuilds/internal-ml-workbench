import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix
from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport

from workbench.constants import APP_IMAGE_URL, DATA_FILE, DEVELOPER_IMAGE_URL
from workbench.ml import detect_task_type, train_and_compare
from workbench.state import get_df, reset_training_outputs


@st.cache_data(show_spinner=False)
def build_profile_html(df: pd.DataFrame) -> str:
    report = ProfileReport(df, minimal=True, explorative=True)
    return report.to_html()


def home_page() -> None:
    st.image(APP_IMAGE_URL, use_container_width=True)
    st.markdown(
        "Upload a CSV, inspect an automated data profile, compare scikit-learn "
        "models, review holdout metrics, then download the best pipeline."
    )


def data_upload_page() -> None:
    st.subheader("Upload Data")
    st.write("Upload a CSV dataset to start profiling and model training.")
    data = st.file_uploader("Choose a CSV file", type=["csv"])

    col1, col2 = st.columns([1, 1])
    persist_to_disk = col1.checkbox("Persist as sourcedata.csv", value=True)
    clear_data = col2.button("Clear loaded dataset", type="secondary")

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
    if df is not None:
        st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
        st.dataframe(df.head(200), use_container_width=True)
    else:
        st.info("No dataset loaded yet.")


def data_profiling_page() -> None:
    st.subheader("Automated Data Profiling")
    df = get_df()
    if df is None:
        st.warning("Load a dataset first on the Upload Data page.")
        return

    if st.button("Generate profile", type="primary"):
        with st.spinner("Building profile report..."):
            profile_html = build_profile_html(df)
        st.components.v1.html(profile_html, height=900, scrolling=True)


def auto_ml_page() -> None:
    st.subheader("Train Models (scikit-learn)")
    df = get_df()
    if df is None:
        st.warning("Load a dataset first on the Upload Data page.")
        return

    target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
    auto_task = detect_task_type(df, target)
    task_type = st.radio(
        "Task type",
        options=["classification", "regression"],
        index=0 if auto_task == "classification" else 1,
        horizontal=True,
        help=f"Auto-detected as: {auto_task}",
    )

    if st.button("Train and compare", type="primary"):
        try:
            with st.spinner("Training and cross-validating models..."):
                st.session_state.artifacts = train_and_compare(df, target, task_type)
            st.success(f"Training complete ({task_type}).")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    artifacts = st.session_state.artifacts
    if artifacts is not None:
        st.markdown("**Training summary**")
        st.dataframe(artifacts.setup_df, use_container_width=True)

        st.markdown("**Model leaderboard (cross-validation mean scores)**")
        st.dataframe(artifacts.leaderboard_df, use_container_width=True)


def evaluation_page() -> None:
    st.subheader("Evaluation Report")
    artifacts = st.session_state.artifacts
    if artifacts is None:
        st.warning("Train models first on the Auto ML page.")
        return

    st.markdown("**Holdout metrics**")
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

        st.markdown("**Classification report**")
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


def model_download_page() -> None:
    st.subheader("Download Best Model")
    artifacts = st.session_state.artifacts
    if artifacts is None:
        st.warning("Train models first on the Auto ML page.")
        return

    st.download_button(
        "Download model (.pkl)",
        data=artifacts.model_bytes,
        file_name="auto_trained_model.pkl",
        mime="application/octet-stream",
    )
    st.caption("The downloaded file is a fitted scikit-learn pipeline.")


def developer_page() -> None:
    st.subheader("Developer")
    st.image(DEVELOPER_IMAGE_URL, use_container_width=True)
    st.write(
        "This app was originally developed by Elvis Darko and modernized for "
        "a stable local-first workflow."
    )


def render_sidebar() -> str:
    css_style = {
        "icon": {"color": "white"},
        "nav-link": {"--hover-color": "grey"},
        "nav-link-selected": {"background-color": "#FF4C1B"},
    }
    with st.sidebar:
        st.image(APP_IMAGE_URL, use_container_width=True)
        st.info(
            "Build and download a scikit-learn model with Streamlit and "
            "ydata-profiling."
        )
        return option_menu(
            menu_title=None,
            options=[
                "Home",
                "Upload Data",
                "Data Profiling",
                "Auto ML",
                "Evaluation",
                "Model Download",
                "Developer",
            ],
            icons=[
                "house",
                "cloud-upload",
                "clipboard-data",
                "cpu",
                "bar-chart",
                "download",
                "people",
            ],
            styles=css_style,
        )
