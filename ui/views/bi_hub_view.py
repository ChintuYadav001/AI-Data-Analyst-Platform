"""
ui.views.bi_hub_view
====================
AI Business Intelligence & Decision Intelligence Platform Hub for Quantum Analyst.
Unites 5 core enterprise decision pillars into one connected cockpit:
1. Executive Command Center
2. AI Business Recommendations & Opportunities
3. What-If Scenario Simulator
4. AI Alerts & Anomaly Monitoring
5. Department-Specific AI Analysts
Core Principle: "Data engines calculate. ML models predict. AI explains. Business intelligence recommends. Users decide."
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import core.bi as bi


def render_bi_hub_view():
    # Verify dataset loaded
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No active dataset found. Please upload a dataset or select a sample dataset first.")
        c1, _ = st.columns([1, 4])
        with c1:
            if st.button("📂 Go to Upload Dataset", type="primary", use_container_width=True):
                st.session_state["active_nav"] = "Upload Dataset"
                st.rerun()
        return

    df: pd.DataFrame = st.session_state["df"]
    proj_name = st.session_state.get("current_project_name", "Active Dataset")

    # Session State defaults for BI
    if "bi_domain" not in st.session_state:
        st.session_state["bi_domain"] = "Auto-Detect"
    if "bi_dismissed_alerts" not in st.session_state:
        st.session_state["bi_dismissed_alerts"] = []
    if "bi_active_tab" not in st.session_state:
        st.session_state["bi_active_tab"] = 0

    # Domain context
    domain_ctx = bi.get_domain_context(df, manual_domain=st.session_state["bi_domain"])
    detected_domain = domain_ctx["domain"]

    # Header Ribbon
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
        <div>
            <div style="font-size: 0.74rem; font-weight: 800; color: #00F0FF; letter-spacing: 0.08em; text-transform: uppercase;">
                // AI BUSINESS DECISION & DECISION INTELLIGENCE PLATFORM
            </div>
            <h2 style="margin: 2px 0 6px 0; color: #FFFFFF; font-size: 1.7rem; font-weight: 800;">
                🌐 Quantum Decision Intelligence Hub
            </h2>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                Turn analytics into business decisions. Executive command center, automated recommendations, What-If simulation, alerts, and departmental analysts.
            </div>
        </div>
        <div style="text-align: right; background: rgba(0, 240, 255, 0.06); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 6px 14px;">
            <div style="font-size: 0.68rem; font-weight: 700; color: #38BDF8;">BUSINESS DOMAIN</div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.92rem;">{detected_domain}</div>
            <div style="font-size: 0.72rem; color: #94A3B8;">{len(df):,} records &bull; {len(domain_ctx['available_dimensions'])} mapped metrics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Domain override & transparency notice
    if domain_ctx.get("disclaimer"):
        st.caption(f"ℹ️ {domain_ctx['disclaimer']}")

    # 5 Major Business Pillars (Tabs)
    bi_tabs = st.tabs([
        "🏛️ Executive Command Center",
        "💡 Business Recommendations",
        "🔀 What-If Scenario Simulator",
        "🚨 AI Alerts & Monitoring",
        "👥 Department AI Analysts",
    ])

    # =========================================================================
    # TAB 1: EXECUTIVE COMMAND CENTER
    # =========================================================================
    with bi_tabs[0]:
        st.subheader("🏛️ Executive Command Center")
        st.caption("High-level executive posture: health scoring, top signals, strategic risks, and insight-to-action workflows.")

        cc_data = bi.generate_executive_command_center(df, dataset_name=proj_name)
        kpi_cards = cc_data.get("kpi_cards", [])
        health = cc_data.get("health", {})
        summary = cc_data.get("summary", {})

        # Top KPI Cards
        cols_kpi = st.columns(len(kpi_cards))
        for idx, k in enumerate(kpi_cards):
            with cols_kpi[idx]:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 9px; padding: 12px; margin-bottom: 12px;">
                    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">{k['title']}</div>
                    <div style="font-size: 1.45rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{k['value']}</div>
                    <div style="font-size: 0.75rem; color: #38BDF8;">{k['subtitle']}</div>
                </div>
                """, unsafe_allow_html=True)

        c_sum1, c_sum2 = st.columns([6, 4])
        with c_sum1:
            st.markdown(f"""
            <div style="background: rgba(0, 240, 255, 0.04); border-left: 4px solid #00F0FF; padding: 14px 16px; border-radius: 6px; margin-bottom: 16px;">
                <div style="font-weight: 800; color: #00F0FF; font-size: 0.85rem; margin-bottom: 4px;">AI EXECUTIVE STATUS ASSESSMENT</div>
                <div style="font-size: 0.9rem; color: #E2E8F0; line-height: 1.5;">{summary.get('status', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            col_sig1, col_sig2 = st.columns(2)
            with col_sig1:
                st.markdown("##### 🟢 Top Positive Signals")
                for sig in summary.get("positive_signals", []):
                    st.markdown(f"- <span style='font-size: 0.85rem; color: #CBD5E1;'>{sig}</span>", unsafe_allow_html=True)
            with col_sig2:
                st.markdown("##### 🔴 Top Business Risks")
                for rsk in summary.get("top_risks", []):
                    st.markdown(f"- <span style='font-size: 0.85rem; color: #CBD5E1;'>{rsk}</span>", unsafe_allow_html=True)

        with c_sum2:
            st.markdown("##### 📊 Dimensional Business Health (0-100)")
            dim_df = pd.DataFrame(list(health.get("dimensions", {}).items()), columns=["Dimension", "Score"])
            fig_health = px.bar(
                dim_df,
                x="Score",
                y="Dimension",
                orientation="h",
                color="Score",
                color_continuous_scale="Viridis",
                range_x=[0, 100],
            )
            fig_health.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font_color="#E2E8F0",
                height=240,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_health, use_container_width=True)

        st.markdown("---")
        st.markdown("#### ⚡ Insight → Action Execution Workflow")
        for wf in cc_data.get("workflow", []):
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #38BDF8; font-weight: 700; font-size: 0.85rem;">INSIGHT: {wf['insight']}</div>
                    <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 2px;"><b>Recommendation:</b> {wf['recommendation']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Download Report Package
        st.markdown("---")
        st.markdown("#### 📑 Executive Business Report Export (16 Sections)")
        c_rep1, c_rep2, c_rep3 = st.columns(3)
        with c_rep1:
            if st.button("📑 Generate & Download PDF Dossier", type="primary", use_container_width=True):
                with st.spinner("Rendering institutional 16-section Executive Business Report..."):
                    # Assemble full BI data package
                    recs = bi.generate_business_recommendations(df, dataset_name=proj_name)
                    alerts = bi.generate_business_alerts(df)
                    bi_pkg = {
                        "dataset_name": proj_name,
                        "health": health,
                        "kpi_cards": kpi_cards,
                        "summary": summary,
                        "recommendations": recs,
                        "alerts": alerts,
                    }
                    pdf_bytes = bi.generate_executive_business_pdf_report(bi_pkg)
                    st.download_button(
                        label="⬇️ Download Executive PDF",
                        data=pdf_bytes,
                        file_name=f"{proj_name}_Executive_Business_Report.pdf",
                        mime="application/pdf",
                        key="btn_dl_bi_pdf",
                    )
        with c_rep2:
            if st.button("🌐 Export Responsive HTML Report", use_container_width=True):
                recs = bi.generate_business_recommendations(df, dataset_name=proj_name)
                alerts = bi.generate_business_alerts(df)
                bi_pkg = {
                    "dataset_name": proj_name,
                    "health": health,
                    "kpi_cards": kpi_cards,
                    "summary": summary,
                    "recommendations": recs,
                    "alerts": alerts,
                }
                html_str = bi.generate_executive_business_html_report(bi_pkg)
                st.download_button(
                    label="⬇️ Download HTML Dossier",
                    data=html_str,
                    file_name=f"{proj_name}_Executive_Report.html",
                    mime="text/html",
                    key="btn_dl_bi_html",
                )
        with c_rep3:
            if st.button("📊 Export Tabular Findings (CSV)", use_container_width=True):
                recs = bi.generate_business_recommendations(df, dataset_name=proj_name)
                alerts = bi.generate_business_alerts(df)
                bi_pkg = {
                    "dataset_name": proj_name,
                    "health": health,
                    "kpi_cards": kpi_cards,
                    "summary": summary,
                    "recommendations": recs,
                    "alerts": alerts,
                }
                csv_str = bi.generate_executive_business_csv_summary(bi_pkg)
                st.download_button(
                    label="⬇️ Download Findings CSV",
                    data=csv_str,
                    file_name=f"{proj_name}_Executive_Summary.csv",
                    mime="text/csv",
                    key="btn_dl_bi_csv",
                )

    # =========================================================================
    # TAB 2: BUSINESS RECOMMENDATIONS & OPPORTUNITIES
    # =========================================================================
    with bi_tabs[1]:
        st.subheader("💡 AI Business Recommendations & Opportunity Engine")
        st.caption("Evidence-based decisions calculated directly from distributions, correlations, and predictive benchmarks.")

        recommendations = bi.generate_business_recommendations(df, dataset_name=proj_name)

        if not recommendations:
            st.info("No critical business recommendations generated for the active dataset.")
        else:
            c_flt1, c_flt2 = st.columns([2, 2])
            with c_flt1:
                p_filter = st.selectbox("Filter by Priority:", options=["All Priorities", "CRITICAL", "HIGH", "MEDIUM", "LOW"], index=0)
            with c_flt2:
                st.caption("Sorted by verified Priority Score (Magnitude, Affected Volume, Risk, Trend).")

            filtered_recs = recommendations
            if p_filter != "All Priorities":
                filtered_recs = [r for r in recommendations if r["priority"] == p_filter]

            for r in filtered_recs:
                sev_color = "#EF4444" if r["priority"] == "CRITICAL" else ("#F59E0B" if r["priority"] == "HIGH" else "#00F0FF")
                with st.container():
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-left: 5px solid {sev_color}; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 0.75rem; font-weight: 800; color: {sev_color}; letter-spacing: 0.05em; text-transform: uppercase;">
                                {r['priority']} PRIORITY &bull; SCORE {r['priority_score']:.0f}/100 &bull; {r['category']}
                            </span>
                            <span style="font-size: 0.72rem; color: #64748B;">{r['timestamp']}</span>
                        </div>
                        <h4 style="margin: 2px 0 6px 0; color: #FFFFFF; font-size: 1.05rem;">{r['title']}</h4>
                        <div style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 6px;"><b>Evidence:</b> {r['evidence']}</div>
                        <div style="font-size: 0.83rem; color: #94A3B8; margin-bottom: 8px;"><b>Possible Driver:</b> {r['reason']}</div>
                        <div style="background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 6px; padding: 8px 12px; font-size: 0.85rem; color: #38BDF8;">
                            👉 <b>Recommended Action:</b> {r['recommended_action']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"🔍 Traceability Audit — Why am I seeing this? ({r['id']})", expanded=False):
                        trace = r.get("traceability", {})
                        st.json(trace)

            # Business Opportunities Section
            st.markdown("---")
            st.markdown("#### 🚀 Business Opportunity Detection (Pareto & Efficiency)")
            opps = bi.detect_business_opportunities(df)
            if opps:
                for opp in opps:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                        <div style="color: #10B981; font-weight: 800; font-size: 0.75rem; text-transform: uppercase;">{opp['type']}</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF; margin: 2px 0;">{opp['title']}</div>
                        <div style="font-size: 0.82rem; color: #CBD5E1;">{opp['evidence']}</div>
                        <div style="font-size: 0.82rem; color: #34D399; margin-top: 4px;"><b>Impact:</b> {opp['potential_impact']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: WHAT-IF SCENARIO SIMULATOR
    # =========================================================================
    with bi_tabs[2]:
        st.subheader("🔀 AI What-If & Scenario Simulator")
        st.caption("Simulate strategic business decisions across price, marketing, churn, sales volume, and costs with transparent assumptions.")

        sim_tab1, sim_tab2 = st.tabs(["Interactive Scenario Simulator", "Multi-Scenario Comparison (A vs B vs C)"])

        with sim_tab1:
            base_metrics = bi.get_dataset_baseline_metrics(df)

            st.markdown("##### 📌 Baseline Operating Scenario")
            b_cols = st.columns(4)
            with b_cols[0]:
                st.metric("Baseline Revenue", f"₹{base_metrics['revenue']:,.2f}")
            with b_cols[1]:
                st.metric("Baseline Net Profit", f"₹{base_metrics['profit']:,.2f}")
            with b_cols[2]:
                st.metric("Operating Margin", f"{base_metrics['margin_pct']:.1f}%")
            with b_cols[3]:
                st.metric("Total Volume / Units", f"{int(base_metrics['volume']):,}")

            st.markdown("---")
            st.markdown("##### 🎛️ Scenario Perturbation Variables")
            sc_col1, sc_col2, sc_col3 = st.columns(3)
            with sc_col1:
                p_price = st.slider("Price Change (%):", min_value=-30.0, max_value=30.0, value=10.0, step=1.0)
                p_mktg = st.slider("Marketing Spend Change (%):", min_value=-30.0, max_value=50.0, value=20.0, step=5.0)
            with sc_col2:
                p_churn = st.slider("Customer Churn Change (%):", min_value=-20.0, max_value=20.0, value=-5.0, step=1.0)
                p_vol = st.slider("Exogenous Volume Shift (%):", min_value=-30.0, max_value=30.0, value=0.0, step=2.0)
            with sc_col3:
                p_cost = st.slider("Cost Escalation (%):", min_value=-20.0, max_value=30.0, value=5.0, step=1.0)
                p_conv = st.slider("Conversion Rate Improvement (%):", min_value=-10.0, max_value=20.0, value=5.0, step=1.0)

            modifications = {
                "price_change_pct": p_price,
                "marketing_spend_pct": p_mktg,
                "churn_change_pct": p_churn,
                "sales_volume_pct": p_vol,
                "costs_pct": p_cost,
                "conversion_pct": p_conv,
            }

            sim_res = bi.simulate_business_scenario(df, modifications=modifications)

            st.markdown("---")
            st.markdown("##### 📊 Simulated Business Outcome")
            deltas = sim_res["deltas"]
            sim = sim_res["simulated"]

            sm_cols = st.columns(4)
            with sm_cols[0]:
                st.metric("Simulated Revenue", f"₹{sim['revenue']:,.2f}", delta=f"{deltas['revenue_pct']:+.1f}%")
            with sm_cols[1]:
                st.metric("Simulated Profit", f"₹{sim['profit']:,.2f}", delta=f"{deltas['profit_pct']:+.1f}%")
            with sm_cols[2]:
                st.metric("Simulated Margin", f"{sim['margin_pct']:.1f}%", delta=f"{deltas['margin_pct']:+.1f}% pts")
            with sm_cols[3]:
                st.metric("Simulated Demand", f"{int(sim['volume']):,}", delta=f"{int(deltas['volume']):+,}")

            # Comparison Table
            st.markdown("###### Detailed Scenario vs Baseline Delta Table")
            df_comp = pd.DataFrame(sim_res["comparison_table"])
            st.dataframe(df_comp, use_container_width=True)

            # AI Strategic Interpretation
            st.markdown(f"""
            <div style="background: rgba(0, 240, 255, 0.05); border-left: 3px solid #00F0FF; padding: 12px 14px; border-radius: 4px; font-size: 0.85rem; color: #E2E8F0; margin-top: 10px;">
                <b>AI STRATEGIC INTERPRETATION:</b> {sim_res['ai_interpretation']}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("ℹ️ Model Assumptions & Limitations", expanded=False):
                for a in sim_res["assumptions_and_limitations"]:
                    st.caption(f"- {a}")

        with sim_tab2:
            st.markdown("##### 📊 Comparative Scenario Analysis (Baseline vs A vs B vs C)")
            multi_comp = bi.compare_business_scenarios(df)
            df_multi = pd.DataFrame(multi_comp["comparison_data"])

            fig_sc = px.bar(
                df_multi,
                x="Scenario",
                y="Profit",
                color="Scenario",
                title="Profit Comparison Across Strategic Scenarios",
                labels={"Profit": "Estimated Net Profit (₹)"},
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
            st.plotly_chart(fig_sc, use_container_width=True)

            st.dataframe(df_multi[["Scenario", "Revenue", "Profit", "Margin %", "Customers", "Volume"]], use_container_width=True)
            st.success(f"💡 Recommendation: {multi_comp['optimal_scenario_recommendation']}")

    # =========================================================================
    # TAB 4: AI ALERTS & MONITORING
    # =========================================================================
    with bi_tabs[3]:
        st.subheader("🚨 AI Business Alerts & Anomaly Monitoring")
        st.caption("Active monitoring of revenue drops, statistical anomalies, margin leakages, and threshold breaches.")

        c_al1, c_al2 = st.columns([2, 2])
        with c_al1:
            sens_choice = st.select_slider("Alert Sensitivity:", options=["Conservative", "Standard", "Sensitive"], value="Standard")
        with c_al2:
            st.caption("Adaptive significance scoring prevents alert spam by isolating statistically meaningful deviations.")

        alerts = bi.generate_business_alerts(df, sensitivity=sens_choice, dismissed_ids=st.session_state["bi_dismissed_alerts"])

        if not alerts:
            st.success("🟢 No active anomalies or threshold breaches detected in the dataset.")
        else:
            st.markdown(f"**Active Alerts ({len(alerts)} items flagged):**")
            for al in alerts:
                badge_bg = "#EF4444" if al["severity"] == "CRITICAL" else ("#F59E0B" if al["severity"] == "HIGH" else "#0284C7")
                with st.container():
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="background: {badge_bg}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 800;">
                                {al['severity']}
                            </span>
                            <span style="font-size: 0.75rem; color: #64748B;">{al['timestamp']} &bull; {al['alert_type']}</span>
                        </div>
                        <h4 style="margin: 4px 0 4px 0; color: #FFFFFF; font-size: 1rem;">{al['title']}</h4>
                        <div style="font-size: 0.83rem; color: #CBD5E1;">
                            <b>Metric:</b> {al['metric']} &bull; <b>Actual:</b> {al['actual_value']} &bull; <b>Expected:</b> {al['expected_value']} &bull; <b>Deviation:</b> <span style="color: #F87171;">{al['deviation']}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin: 4px 0;">{al['possible_explanation']}</div>
                        <div style="font-size: 0.82rem; color: #38BDF8;">👉 <b>Action:</b> {al['recommended_action']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 5: DEPARTMENT AI ANALYSTS
    # =========================================================================
    with bi_tabs[4]:
        st.subheader("👥 Department-Specific AI Analysts")
        st.caption("Role-tailored analytical views and preset inquiries calibrated to specific operational departments.")

        dep_list = list(bi.DEPARTMENT_CONFIGS.keys())
        selected_dep = st.selectbox("Select Department Mode:", options=dep_list, index=0)

        dep_data = bi.get_department_analysis(df, department=selected_dep)
        cfg = dep_data.get("config", {})

        st.markdown(f"""
        <div style="background: rgba(0, 240, 255, 0.04); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 12px 14px; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.3rem;">{cfg.get('icon', '👥')}</span>
                <div>
                    <div style="font-weight: 800; color: #FFFFFF; font-size: 1rem;">{cfg.get('title', selected_dep)} Active</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">{cfg.get('focus', '')}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Department KPI Cards
        dep_kpis = dep_data.get("kpis", [])
        if dep_kpis:
            k_cols = st.columns(len(dep_kpis))
            for i, kp in enumerate(dep_kpis):
                with k_cols[i]:
                    st.metric(label=kp["label"], value=kp["value"], delta=kp["change"], delta_color="off")

        st.markdown("---")
        c_memo1, c_memo2 = st.columns([6, 4])
        with c_memo1:
            st.markdown("##### 📝 Strategic Department Briefing")
            st.write(dep_data.get("briefing_memo", ""))
            for f in dep_data.get("focus_findings", []):
                st.markdown(f"- {f}")

        with c_memo2:
            st.markdown("##### ❓ Preset Inquiries for This Role")
            for q in dep_data.get("preset_questions", []):
                if st.button(f"👉 {q}", key=f"btn_q_{hash(q)}_{selected_dep}", use_container_width=True):
                    st.session_state["copilot_open"] = True
                    st.session_state["copilot_input"] = q
                    st.rerun()
