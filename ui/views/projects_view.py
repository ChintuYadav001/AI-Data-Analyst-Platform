import streamlit as st
import pandas as pd
from core.models import ProjectModel

def render_projects_view():
    st.title("📂 Project Management")
    st.markdown("Preserve, switch, and inspect datasets, cleaning histories, and analytics across sessions.")

    projects = ProjectModel.list_all_projects()

    st.markdown(f"### Saved Projects ({len(projects)})")

    if not projects:
        st.info("No projects created yet. Upload a dataset or load a sample to create your first project.")
        return

    for proj_meta in projects:
        p_id = proj_meta.get("id")
        p_name = proj_meta.get("name", "Untitled")
        created = proj_meta.get("created_at", "")[:19].replace("T", " ")
        cleaning_cnt = len(proj_meta.get("cleaning_history", []))

        is_current = st.session_state.get("current_project_id") == p_id

        with st.container():
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                st.markdown(f"#### 📁 {p_name} {'*(Active)*' if is_current else ''}")
                st.caption(f"Project ID: `{p_id}` | Created: {created}")
            with c2:
                st.write(f"Cleaning operations: **{cleaning_cnt}**")
            with c3:
                if not is_current:
                    is_auth = st.session_state.get("authenticated", False)
                    if not is_auth:
                        st.caption("🔒 Login to load")
                    else:
                        if st.button("Switch to Project", key=f"btn_switch_{p_id}"):
                            p = ProjectModel(p_id, p_name)
                            loaded_df = p.get_cleaned_dataset()
                            if loaded_df is not None:
                                st.session_state["df"] = loaded_df
                                raw_data = p.get_raw_dataset()
                                st.session_state["raw_df"] = raw_data if raw_data is not None else loaded_df.copy()
                                st.session_state["cleaning_log"] = p.get_cleaning_history()
                                st.session_state["current_project_id"] = p_id
                                st.session_state["current_project_name"] = p_name
                                st.session_state["active_nav"] = "Dashboard"
                                st.rerun()
                else:
                    st.success("Currently Active")


            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
