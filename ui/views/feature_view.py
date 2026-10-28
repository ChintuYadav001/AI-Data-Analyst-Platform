import streamlit as st
import pandas as pd
from core.models import FeatureEngineeringModel, ProjectModel

def render_feature_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Please upload a dataset or load a sample dataset first.")
        return

    df = st.session_state["df"]
    st.title("⚡ Automatic Feature Engineering")
    st.markdown("Dataset-aware feature discovery. Review formulas, analytical rationale, and approve new features to augment your model.")

    recommendations = FeatureEngineeringModel.recommend_features(df)

    if not recommendations:
        st.info("No additional domain features recommended based on current columns.")
        return

    st.markdown(f"### Recommended Features ({len(recommendations)} available)")

    for idx, rec in enumerate(recommendations):
        with st.container():
            st.markdown(f"#### 💡 {rec['feature_name']}")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**Formula:** `{rec['formula']}`")
                st.markdown(f"**Why this matters:** {rec['reason']}")
                st.markdown(f"**Expected Value:** {rec['expected_value']}")
            with c2:
                if st.button(f"Approve & Create", key=f"btn_feat_{idx}", type="primary"):
                    with st.spinner("Generating feature..."):
                        new_df, msg = FeatureEngineeringModel.apply_feature(df, rec)
                        st.session_state["df"] = new_df
                        
                        p_id = st.session_state.get("current_project_id")
                        if p_id:
                            p = ProjectModel(p_id, st.session_state.get("current_project_name", "Project"))
                            p.save_cleaned_dataset(new_df, st.session_state.get("cleaning_log", []))

                        st.success(msg)
                        st.rerun()

            st.markdown("<hr style='margin: 14px 0;'>", unsafe_allow_html=True)

    with st.expander("👁️ Inspect Current Features / Columns"):
        st.write(f"Total columns currently: **{len(df.columns)}**")
        st.write(list(df.columns))
        st.dataframe(df.head(5), width="stretch")
