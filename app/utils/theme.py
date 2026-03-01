import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7f8;
            --surface: #ffffff;
            --ink: #11201f;
            --muted: #5f6f6e;
            --accent: #0b8f77;
            --line: #d9e4e2;
        }

        .stApp {
            background: radial-gradient(
                circle at top right,
                #e8f7f3 0%,
                var(--bg) 36%,
                #eef2f2 100%
            );
            color: var(--ink);
        }

        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, #ffffff 0%, #ecfaf6 65%, #dff6f0 100%);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 8px 30px rgba(16, 38, 34, 0.08);
            margin-bottom: 1rem;
        }

        .hero h2 {
            margin: 0;
            color: var(--ink);
            letter-spacing: 0.3px;
        }

        .hero p {
            margin: 0.45rem 0 0;
            color: var(--muted);
        }

        .kpi {
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 4px 14px rgba(16, 38, 34, 0.05);
            min-height: 92px;
        }

        .kpi .label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.25rem;
        }

        .kpi .value {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 700;
        }

        .process {
            margin: 0.4rem 0 1rem;
            color: var(--muted);
            font-size: 0.9rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2a26 0%, #12332e 100%);
            border-right: 1px solid #224540;
        }

        section[data-testid="stSidebar"] * {
            color: #e6f1ef !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def flow_hint() -> None:
    st.markdown(
        "<div class='process'>1 Upload -> 2 Profile -> 3 Train -> 4 Evaluate -> 5 Export</div>",
        unsafe_allow_html=True,
    )
