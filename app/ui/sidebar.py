import streamlit as st

try:
    from streamlit_option_menu import option_menu
except ModuleNotFoundError:  # pragma: no cover
    option_menu = None

STEP_OPTIONS = ["Upload", "EDA", "Modelling", "Validation", "Export"]
STEP_ICONS = ["upload", "bar-chart", "cpu", "clipboard-check", "download"]


def _menu_styles() -> dict:
    return {
        "icon": {"color": "var(--primary-color)", "font-size": "18px"},
        "container": {"padding": "0!important"},
        "nav-link": {
            "font-size": "15px",
            "text-align": "left",
            "margin": "3px 0px",
            "border-radius": "8px",
            "--hover-color": "transparent",
            "color": "var(--text-color)",
        },
        "nav-link-selected": {
            "background-color": "var(--primary-color)",
            "color": "white",
        },
    }


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## Internal ML Workbench")
        st.caption("Step-based flow")

        if option_menu is None:
            st.warning("streamlit-option-menu not installed; using default selector.")
            return st.radio("Navigate", STEP_OPTIONS, index=0)

        return option_menu(
            menu_title=None,
            options=STEP_OPTIONS,
            icons=STEP_ICONS,
            styles=_menu_styles(),
            default_index=0,
        )
