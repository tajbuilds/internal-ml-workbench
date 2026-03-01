import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --imw-bg: var(--background-color);
            --imw-fg: var(--text-color);
            --imw-card: var(--secondary-background-color);
            --imw-border: color-mix(in srgb, var(--text-color) 16%, transparent);
            --imw-muted: color-mix(in srgb, var(--text-color) 60%, transparent);
            --imw-accent: var(--primary-color);
            --imw-hover-tint: color-mix(in srgb, var(--primary-color) 18%, transparent);
        }

        section[data-testid="stSidebar"],
        button[kind="header"][aria-label*="sidebar"],
        div[data-testid="collapsedControl"] {
            display: none !important;
        }

        .stApp,
        div[data-testid="stAppViewContainer"],
        div[data-testid="stMain"],
        div[data-testid="stMainBlockContainer"],
        section.main,
        section.main > div {
            background: var(--imw-bg) !important;
            color: var(--imw-fg) !important;
        }

        header[data-testid="stHeader"],
        header[data-testid="stAppHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stAppToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {
            background: var(--imw-bg) !important;
            color: var(--imw-fg) !important;
            border: none !important;
        }

        .block-container {
            padding-top: 0.75rem;
            padding-bottom: 2rem;
            max-width: 1240px;
        }

        .imw-header {
            border: 1px solid var(--imw-border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            background: linear-gradient(120deg, var(--imw-card) 0%, var(--imw-bg) 100%);
            margin-bottom: 1rem;
        }

        .imw-header h2 {
            margin: 0;
            font-size: 1.35rem;
            color: var(--imw-fg);
            font-weight: 700;
            letter-spacing: 0.2px;
        }

        .imw-header p {
            margin: 0.28rem 0 0;
            color: var(--imw-muted);
            font-size: 0.92rem;
        }

        div[data-testid="stMetric"] {
            background: var(--imw-card);
            border: 1px solid var(--imw-border);
            border-radius: 12px;
            padding: 0.6rem 0.75rem;
        }

        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] {
            color: var(--imw-muted) !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--imw-fg) !important;
            font-weight: 700;
        }

        .stSelectbox > div > div,
        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stNumberInput input,
        .stDataFrame,
        .stTable,
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div {
            background: var(--imw-card) !important;
            border-color: var(--imw-border) !important;
            color: var(--imw-fg) !important;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: var(--imw-card) !important;
            border: 1px dashed var(--imw-border) !important;
            color: var(--imw-fg) !important;
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--imw-accent) !important;
        }

        div[data-testid="stFileUploaderDropzone"] * {
            color: var(--imw-muted) !important;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            border: 1px solid var(--imw-border);
            background: var(--imw-card);
            color: var(--imw-muted);
        }

        .stTabs [data-baseweb="tab"]:hover {
            border-color: var(--imw-accent);
            color: var(--imw-fg);
        }

        .stAlert {
            border-radius: 10px;
            border: 1px solid var(--imw-border);
            background: color-mix(in srgb, var(--imw-card) 92%, var(--imw-bg));
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
