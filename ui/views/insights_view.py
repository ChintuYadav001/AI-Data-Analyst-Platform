import streamlit as st
from core.models import AIInsightsModel, DataProfilerModel, EDAModel, DataQualityModel

def render_insights_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Please upload a dataset or load a sample dataset first.")
        return

    df = st.session_state["df"]
    current_goal = st.session_state.get("project_goal", "General Business Intelligence")
    current_explanation = st.session_state.get("project_goal_explanation", "")

    st.title("🧠 Deep AI Insights Engine")
    st.markdown("Multi-horizon executive intelligence (Descriptive, Diagnostic, Predictive, Prescriptive, and Goal Alignment) with mathematical requirement audit.")

    c_goal1, c_goal2 = st.columns([3, 1])
    with c_goal1:
        st.markdown(f"<div style='background: rgba(168, 85, 247, 0.12); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'>"
                    f"<span style='color: #D8B4FE; font-family: monospace; font-size: 0.8rem; font-weight: 700;'>🎯 CURRENT TARGET OBJECTIVE:</span><br>"
                    f"<b style='color: #F3E8FF; font-size: 1rem;'>{current_goal}</b>"
                    f"</div>", unsafe_allow_html=True)
    with c_goal2:
        new_goal_opt = st.selectbox(
            "Recalibrate Objective",
            options=[
                current_goal,
                "Revenue Maximization & Growth",
                "Customer Churn Prevention & Retention",
                "Cost Reduction & Margin Optimization",
                "Anomaly & Financial Risk Mitigation",
                "Conversion Velocity & Efficiency"
            ] if current_goal not in [
                "Revenue Maximization & Growth",
                "Customer Churn Prevention & Retention",
                "Cost Reduction & Margin Optimization",
                "Anomaly & Financial Risk Mitigation",
                "Conversion Velocity & Efficiency"
            ] else [
                "Revenue Maximization & Growth",
                "Customer Churn Prevention & Retention",
                "Cost Reduction & Margin Optimization",
                "Anomaly & Financial Risk Mitigation",
                "Conversion Velocity & Efficiency"
            ],
            index=0,
            key="insights_recalibrate_goal_sel"
        )
        if new_goal_opt != current_goal:
            st.session_state["project_goal"] = new_goal_opt
            st.rerun()

    with st.expander("📝 Edit Goal Explanation & Context Brief", expanded=False):
        new_expl = st.text_area(
            "Analytical Brief / Specific Hypotheses to Investigate",
            value=current_explanation,
            height=75,
            help="Update your analytical focus, target metrics, or hypotheses. Insights will be re-synthesized based on this context.",
            key="insights_edit_goal_explanation"
        )
        if st.button("Apply New Analytical Brief", key="btn_apply_brief"):
            st.session_state["project_goal_explanation"] = new_expl
            st.rerun()

    st.info("🔒 **Data Privacy & Grounding Notice:** Uploaded datasets remain strictly private. Insights are generated from calculated statistical moments, cohort aggregates, and correlation matrices with 100% mathematical auditability.")

    insights_cache_key = f"cached_insights_{id(df)}_{st.session_state.get('project_goal', 'default')}_{hash(st.session_state.get('project_goal_explanation', ''))}"
    if st.button("🔄 Re-synthesize Latest Insights", type="primary") or insights_cache_key not in st.session_state:
        with st.spinner("Executing deep multi-horizon analysis calibrated to objective & explanation..."):
            profile = DataProfilerModel.profile_dataset(df)
            quality = DataQualityModel.calculate_quality_score(df)
            eda = {
                "numerical": EDAModel.analyze_numerical(df),
                "categorical": EDAModel.analyze_categorical(df),
                "relationships": EDAModel.analyze_relationships(df),
                "time_trends": EDAModel.analyze_time_trends(df)
            }
            insights = AIInsightsModel.generate_insights(
                profile, eda, quality,
                goal=st.session_state.get("project_goal"),
                goal_explanation=st.session_state.get("project_goal_explanation", "")
            )
            st.session_state[insights_cache_key] = insights

    insights = st.session_state.get(insights_cache_key, {})

    st.caption(f"Reasoning Provider: **{insights.get('provider', 'Deep Analytical Synthesis')}**")

    # Multi-horizon tabs including Goal Alignment
    tab_goal, tab_desc, tab_diag, tab_pred, tab_anom, tab_presc, tab_spec = st.tabs([
        "🎯 Goal Alignment & Strategy",
        "📊 Descriptive (What Happened?)",
        "🔍 Diagnostic (Why It Happened)",
        "🚀 Predictive (Trends & Signals)",
        "⚠️ Anomalies & Volatility",
        "💡 Prescriptive Recommendations",
        "📋 Requirement Specification"
    ])

    with tab_goal:
        st.markdown(f"### 🎯 Strategic Objective Alignment: `{current_goal}`")
        if current_explanation:
            st.markdown(
                f"<div style='background: rgba(168, 85, 247, 0.08); border-left: 3px solid #C084FC; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px;'>"
                f"<span style='color: #D8B4FE; font-family: monospace; font-size: 0.75rem; font-weight: 700;'>// DESIGNATED GOAL EXPLANATION & CONTEXT:</span><br>"
                f"<span style='color: #E2E8F0; font-size: 0.92rem;'>\"{current_explanation}\"</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.caption("Custom-calibrated findings and tactical action sequences optimized specifically for this objective and explanation.")
        
        goal_data = insights.get("goal_alignment", {})
        findings = goal_data.get("strategic_findings", [])
        interventions = goal_data.get("tactical_interventions", [])
        hypotheses = goal_data.get("hypothesis_evaluations", [])

        if hypotheses:
            st.markdown("#### 🔬 Empirical Hypothesis Testing & Feature Linkages")
            for h in hypotheses:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.7); border-left: 3px solid #10B981; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                    {h}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        c_find, c_act = st.columns(2)
        with c_find:
            st.markdown("#### 🔭 Strategic Objective Findings")
            for f in findings:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.7); border-left: 3px solid #C084FC; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                    {f}
                </div>
                """, unsafe_allow_html=True)

        with c_act:
            st.markdown("#### ⚡ Tactical Interventions")
            for act in interventions:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.7); border-left: 3px solid #00F0FF; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                    <b>Intervention:</b> {act}
                </div>
                """, unsafe_allow_html=True)

    with tab_desc:
        st.markdown("### 📊 Descriptive Baseline")
        st.caption("Factual breakdown of aggregate volumes, central tendencies, and distributions.")
        for item in insights.get("descriptive", []):
            st.markdown(f"- {item}")

    with tab_diag:
        st.markdown("### 🔍 Diagnostic Root-Cause Analysis")
        st.caption("Evaluates correlation drivers, category concentration ratios, and variance linkages.")
        for item in insights.get("diagnostic", []):
            st.markdown(f"- {item}")

    with tab_pred:
        st.markdown("### 🚀 Predictive Momentum & Trajectory")
        st.caption("Temporal trajectory, growth velocity, and period-over-period acceleration signals.")
        for item in insights.get("predictive", []):
            st.markdown(f"- {item}")

    with tab_anom:
        st.markdown("### ⚠️ Data Integrity & Volatility Alerts")
        st.caption("Identifies extreme tail events, high coefficient of variation, and data sparsity.")
        for item in insights.get("anomalies", []):
            st.warning(item)

    with tab_presc:
        st.markdown("### 💡 Actionable Business Recommendations")
        st.caption("Concrete, prioritized operational interventions supported by verified data.")
        for rec in insights.get("recommendations", []):
            st.success(f"✔ **Strategic Action:** {rec}")

    with tab_spec:
        st.markdown("### 📋 Analytical Requirement Specification & Audit")
        st.caption("Full transparency into data lineage, analytical principles, and anti-hallucination guarantees.")
        
        req_spec = insights.get("requirement_spec", {})
        st.markdown(f"""
        <div style="background: rgba(8, 14, 28, 0.95); border: 1px solid rgba(0, 240, 255, 0.35); border-left: 4px solid #00F0FF; border-radius: 10px; padding: 20px; box-shadow: 0 0 20px rgba(0, 240, 255, 0.08);">
            <p style="color: #00F0FF; font-family: monospace; font-size: 0.85rem; margin-bottom: 8px;">// AUDIT GROUNDING SPECIFICATION</p>
            <p style="color: #E2E8F0;"><b>Analytical Objective:</b> {req_spec.get('analytical_objective', 'Autonomous analytics')}</p>
            <p style="color: #CBD5E1;"><b>Data Lineage:</b> {req_spec.get('data_lineage', 'Verified dataset records')}</p>
            <p style="color: #CBD5E1;"><b>Privacy Guarantee:</b> {req_spec.get('privacy_guarantee', 'Zero raw rows transmitted')}</p>
            <p style="color: #10B981;"><b>Grounding Audit:</b> {req_spec.get('grounding_audit', '100% mathematically verifiable')}</p>
        </div>
        """, unsafe_allow_html=True)
