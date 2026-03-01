import streamlit as st

try:
    from streamlit_option_menu import option_menu
except ModuleNotFoundError:  # pragma: no cover
    option_menu = None

STEP_OPTIONS = ["Workspace", "Datasets", "EDA", "Modelling", "Validation", "Export"]
STEP_ICONS = ["house", "database", "bar-chart", "cpu", "clipboard-check", "download"]


def _menu_styles() -> dict:
    return {
        "icon": {"color": "var(--primary-color)", "font-size": "16px"},
        "container": {
            "padding": "0!important",
            "background-color": "transparent",
            "border": "1px solid color-mix(in srgb, var(--text-color) 16%, transparent)",
            "border-radius": "10px",
            "margin-bottom": "0.75rem",
        },
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "0px",
            "padding": "0.55rem 0.75rem",
            "border-radius": "8px",
            "--hover-color": "color-mix(in srgb, var(--primary-color) 14%, transparent)",
            "color": "var(--text-color)",
        },
        "nav-link-selected": {
            "background-color": "var(--primary-color)",
            "color": "white",
        },
    }


def render_workspace_context() -> None:
    datasets = st.session_state.get("datasets", [])

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Workspace Context**")
    with c2:
        if datasets:
            labels = [f"{d['name']} ({d['rows']}x{d['cols']})" for d in datasets]
            ids = [d["id"] for d in datasets]
            current_id = st.session_state.get("active_dataset_id")
            idx = ids.index(current_id) if current_id in ids else 0
            selected_label = st.selectbox(
                "Active dataset",
                labels,
                index=idx,
                label_visibility="collapsed",
                key="top_active_dataset",
            )
            st.session_state.active_dataset_id = ids[labels.index(selected_label)]
        else:
            st.info("No datasets yet. Upload one in Workspace.")


def render_top_nav() -> str:
    if option_menu is None:
        return st.radio("Section", STEP_OPTIONS, index=0, horizontal=True)

    return option_menu(
        menu_title=None,
        options=STEP_OPTIONS,
        icons=STEP_ICONS,
        orientation="horizontal",
        styles=_menu_styles(),
        default_index=0,
    )
