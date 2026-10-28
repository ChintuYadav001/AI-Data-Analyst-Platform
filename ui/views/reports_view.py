import streamlit as st
import os
from core.models import ReportModel, DataProfilerModel, DataQualityModel, EDAModel, AIInsightsModel
from core.config import EXPORTS_DIR

def render_reports_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Please upload a dataset or load a sample dataset first.")
        return

    df = st.session_state["df"]
    project_name = st.session_state.get("current_project_name", "Analytics Project")
    project_goal = st.session_state.get("project_goal", "General Business Intelligence")
    project_goal_explanation = st.session_state.get("project_goal_explanation", "")
    cleaning_log = st.session_state.get("cleaning_log", [])

    st.title("📄 Institutional Analytical Dossier")
    st.markdown("Automated generation of institutional-grade documentation (30+ pages) calibrated to your project objective and analytical brief, ready for executive and board presentation.")

    expl_badge = f"<br/><span style='color: #CBD5E1; font-size: 0.85rem;'><b>Context Brief:</b> <i>\"{project_goal_explanation}\"</i></span>" if project_goal_explanation else ""
    st.markdown(
        f"<div style='background: rgba(168, 85, 247, 0.12); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;'>"
        f"<span style='color: #D8B4FE; font-family: monospace; font-size: 0.8rem; font-weight: 700;'>🎯 CALIBRATED REPORT GOAL:</span> "
        f"<b style='color: #F3E8FF; font-size: 0.95rem;'>{project_goal}</b> &bull; <span style='color: #00F0FF; font-family: monospace;'>[PAGE TARGET: 30+ PAGES]</span>"
        f"{expl_badge}"
        f"</div>",
        unsafe_allow_html=True
    )

    report_cache_key = f"cached_report_{id(df)}_{len(cleaning_log)}_{project_goal}_{hash(project_goal_explanation)}"
    if report_cache_key not in st.session_state:
        with st.spinner("Compiling full analytical report..."):
            profile = DataProfilerModel.profile_dataset(df)
            quality = DataQualityModel.calculate_quality_score(df)
            eda = {
                "numerical": EDAModel.analyze_numerical(df),
                "categorical": EDAModel.analyze_categorical(df),
                "relationships": EDAModel.analyze_relationships(df),
                "time_trends": EDAModel.analyze_time_trends(df)
            }
            insights = AIInsightsModel.generate_offline_insights(
                profile, eda, quality,
                goal=project_goal,
                goal_explanation=project_goal_explanation
            )
            report_md = ReportModel.generate_report_markdown(
                project_name, profile, quality, cleaning_log, eda, [], insights
            )
            st.session_state[report_cache_key] = report_md

    report_md = st.session_state[report_cache_key]

    c1, c2 = st.columns([1, 1])
    with c1:
        # PDF Generation (Institutional 30+ pages) - Requires Authenticated Session
        is_auth = st.session_state.get("authenticated", False)
        if not is_auth:
            st.markdown("""
            <div style="background: rgba(255, 0, 85, 0.08); border: 1px solid rgba(255, 0, 85, 0.35); border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #FF0055; font-weight: 700;">
                    🔒 AUTHENTICATION REQUIRED • GUEST RESTRICTION
                </div>
                <div style="font-size: 0.85rem; color: #E2E8F0; margin-top: 4px;">
                    Full Institutional Dossier generation (30+ Pages PDF) requires an active authenticated account.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔑 Sign In / Sign Up to Unlock 30+ Page PDF", type="secondary", width="stretch"):
                st.session_state["active_nav"] = "Auth"
                st.rerun()
        else:
            pdf_path = str(EXPORTS_DIR / f"{project_name}_institutional_dossier.pdf")
            if st.button("📑 Export Institutional Dossier (30+ Pages PDF)", type="primary"):
                with st.spinner("Rendering multi-chapter institutional dossier (30+ pages)..."):
                    profile = DataProfilerModel.profile_dataset(df)
                    quality = DataQualityModel.calculate_quality_score(df)
                    eda = {
                        "numerical": EDAModel.analyze_numerical(df),
                        "categorical": EDAModel.analyze_categorical(df),
                        "relationships": EDAModel.analyze_relationships(df),
                        "time_trends": EDAModel.analyze_time_trends(df)
                    }
                    insights = AIInsightsModel.generate_offline_insights(
                        profile, eda, quality,
                        goal=project_goal,
                        goal_explanation=project_goal_explanation
                    )
                    ReportModel.export_pdf(
                        pdf_path, project_name, profile, quality, cleaning_log, eda, insights,
                        goal=project_goal,
                        goal_explanation=project_goal_explanation
                    )
                    st.success("✓ Institutional Dossier synthesized successfully (30+ pages).")
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Institutional PDF (30+ Pages)",
                            data=f,
                            file_name=f"{project_name}_Institutional_Dossier_30Pages.pdf",
                            mime="application/pdf"
                        )


    with c2:
        st.download_button(
            label="⬇️ Download Markdown Dossier",
            data=report_md,
            file_name=f"{project_name}_report.md",
            mime="text/markdown"
        )

    st.markdown("---")
    st.markdown(report_md)
