import streamlit as st


def render_sidebar_context() -> None:
    with st.sidebar:
        st.markdown("## Internal ML Workbench")
        st.caption("Workspace context")

        datasets = st.session_state.get("datasets", [])
        if datasets:
            labels = [f"{d['name']} ({d['rows']}x{d['cols']})" for d in datasets]
            ids = [d["id"] for d in datasets]
            current_id = st.session_state.get("active_dataset_id")
            idx = ids.index(current_id) if current_id in ids else 0
            selected_label = st.selectbox("Active dataset", labels, index=idx)
            st.session_state.active_dataset_id = ids[labels.index(selected_label)]
        else:
            st.info("No datasets yet. Upload one in Workspace.")
