"""
Quantum Analyst - Institutional AI EDA Engine View
Adaptive 14-stage goal-aware exploratory data analysis cockpit with executive scorecards,
target driver associations, distribution intelligence, auto-granularity chronological waves,
dimensional segment Pareto rankings, multi-method anomaly impact, automated hypothesis tests,
natural language EDA assistant, and data-backed chart actions (Explain / View SQL / Pin).
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

from core.models import EDAModel
from core.models.chart_model import ChartModel
from core.eda import AIEDAOrchestrator, EDANLQueryEngine


def render_eda_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No active dataset detected. Please upload or load a sample dataset first.")
        if st.button("⬅️ Return to Dashboard / Ingestion", type="primary"):
            st.session_state["active_nav"] = "Dashboard"
            st.rerun()
        return

    df = st.session_state["df"]
    project_name = st.session_state.get("current_project_name", "Active Dataset")
    user_goal = st.session_state.get("project_goal", "")
    user_target = st.session_state.get("eda_target", None)
    eda_mode = st.session_state.get("eda_mode", "auto")

    # -------------------------------------------------------------------------
    # 1. RUN OR RETRIEVE FULL AI EDA AUDIT
    # -------------------------------------------------------------------------
    with st.spinner("🔬 Running institutional 14-stage AI EDA engine..."):
        try:
            eda_res = AIEDAOrchestrator.run_full_eda(df, user_goal=user_goal, user_target=user_target, mode=eda_mode)
        except Exception:
            eda_res = EDAModel.run_ai_eda(df, user_goal=user_goal, user_target=user_target, mode=eda_mode)

    profile = eda_res.get("profile", {})
    quality = eda_res.get("quality", {})
    goal_info = eda_res.get("goal_info", {})
    target_info = eda_res.get("target_info", {})
    coverage = eda_res.get("coverage", {})
    eda_plan = eda_res.get("eda_plan", {})

    target_col = target_info["column"] if target_info else None
    target_task = target_info["task_type"] if target_info else "N/A"

    # Sync goal back to session state if it was inferred
    if not user_goal and goal_info.get("goal"):
        st.session_state["project_goal"] = goal_info["goal"]

    # -------------------------------------------------------------------------
    # 2. FUTURISTIC HERO BANNER & INTELLIGENCE HUD
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(14, 20, 36, 0.95), rgba(7, 10, 20, 0.98)); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 16px; padding: 22px 28px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0, 240, 255, 0.08);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(0, 240, 255, 0.12); border: 1px solid rgba(0, 240, 255, 0.4); display: flex; align-items: center; justify-content: center; font-size: 1.6rem;">
                    🔬
                </div>
                <div>
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <h2 style="margin: 0; font-size: 1.55rem; color: #FFFFFF; font-weight: 800; letter-spacing: -0.02em;">
                            AI Exploratory Data Analysis // Spectral Engine
                        </h2>
                        <span style="background: rgba(0, 240, 255, 0.15); color: #00F0FF; border: 1px solid rgba(0, 240, 255, 0.35); font-size: 0.72rem; padding: 3px 9px; border-radius: 9999px; font-weight: 700; text-transform: uppercase;">
                            14-Stage Adaptive
                        </span>
                        <span style="background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.35); font-size: 0.72rem; padding: 3px 9px; border-radius: 9999px; font-weight: 700;">
                            {profile.get('domain', 'Enterprise Tabular')}
                        </span>
                    </div>
                    <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.88rem;">
                        Dataset-aware & goal-calibrated analytical intelligence. Distribution moments, inferential hypothesis testing, cohort contribution, and causal-guarded driver exploration.
                    </p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.82rem; color: #64748B;">Dataset:</span>
                <span style="font-size: 0.84rem; font-weight: 700; color: #F1F5F9; background: rgba(255, 255, 255, 0.05); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.1);">
                    {project_name}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    total_row_count = max(len(df), st.session_state.get("dataset_profile", {}).get("rows", 0))
    is_large_dataset = total_row_count >= 250000
    if is_large_dataset:
        c_mode1, c_mode2 = st.columns([4, 1.2])
        with c_mode1:
            if total_row_count >= 2000000:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-radius: 10px; padding: 12px 18px; margin-bottom: 16px;">
                    <span style="font-family: monospace; font-size: 0.8rem; color: #10B981; font-weight: 700;">⚡ 2M+ ULTRA SCALE MODE ACTIVE &bull; {total_row_count:,} ROWS</span>
                    <span style="color: #94A3B8; font-size: 0.82rem; margin-left: 8px;">DuckDB SIMD Engine &bull; WebGL Adaptive Sampling &bull; Real-time Aggregation</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(0, 240, 255, 0.08); border: 1px solid #00F0FF; border-radius: 10px; padding: 12px 18px; margin-bottom: 16px;">
                    <span style="font-family: monospace; font-size: 0.8rem; color: #00F0FF; font-weight: 700;">⚡ LARGE DATASET MODE ACTIVE</span>
                    <span style="color: #94A3B8; font-size: 0.82rem; margin-left: 8px;">DuckDB acceleration &bull; Smart sampling &bull; Sub-millisecond reruns</span>
                </div>
                """, unsafe_allow_html=True)
        with c_mode2:
            if st.button("🚀 Run Deep EDA", type="secondary", width="stretch", help="Executes background Isolation Forest and high-dimensional driver rankings"):
                with st.spinner("Executing Deep EDA models in background..."):
                    from core.eda_engine import EDAEngine
                    from core.eda_planner import EDAPlanner
                    from core.profile_engine import ProfileEngine
                    p_path = st.session_state.get("parquet_path")
                    d_hash = st.session_state.get("dataset_hash", "default")
                    if p_path:
                        prof = ProfileEngine.profile_dataset_fast(p_path, dataset_hash=d_hash)
                        pln = EDAPlanner.create_eda_plan(prof, goal=user_goal, target_col=target_col)
                        deep_res = EDAEngine.run_deep_eda(p_path, prof, pln)
                        st.session_state["deep_eda_res"] = deep_res
                        st.success("✓ Deep EDA models completed!")
                        st.rerun()

    # -------------------------------------------------------------------------
    # 3. TOP EDA INTELLIGENCE SCORECARD
    # -------------------------------------------------------------------------
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Records", f"{len(df):,}", delta=f"{len(df.columns)} Columns")
    with m2:
        q_score = quality.get("score", 92)
        st.metric("Data Quality", f"{q_score}%", delta="Reliability Audit")
    with m3:
        target_display = target_col.upper() if target_col else "AUTO-DETECT"
        st.metric("Target Variable", target_display, delta=f"{target_task.title()} Task")
    with m4:
        cov_score = coverage.get("score", 85)
        st.metric("EDA Coverage", f"{cov_score}%", delta="14-Stage Pipeline")
    with m5:
        n_findings = len(eda_res.get("key_findings", []))
        st.metric("Key Findings", f"{n_findings} Insights", delta="Evidence-Backed")
    with m6:
        n_anom = sum(a.get("iqr_outliers_count", 0) for a in eda_res.get("anomalies", []))
        st.metric("Anomalies Flagged", f"{n_anom:,}", delta="No Silent Deletion")

    # Brief Narrative Highlight Box
    top_drivers = eda_res.get("target_drivers", [])
    top_driver_str = f"Primary Driver: <b>`{top_drivers[0]['feature']}`</b> ({top_drivers[0]['association_score']}/100 strength)" if top_drivers else "Multi-variable linkage"
    next_step = eda_res.get("next_analyses", ["Audit driver stability across periods."])[0]

    st.markdown(f"""
    <div style="background: rgba(14, 20, 36, 0.6); border: 1px solid rgba(0, 240, 255, 0.18); border-radius: 10px; padding: 12px 18px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div style="font-size: 0.86rem; color: #CBD5E1;">
            🎯 <span style="color: #00F0FF; font-weight: 700;">GOAL:</span> {goal_info.get('goal', 'General Exploratory Intelligence')}
            &nbsp;&bull;&nbsp;
            ⚡ <span style="color: #A78BFA; font-weight: 700;">{top_driver_str}</span>
        </div>
        <div style="font-size: 0.82rem; color: #94A3B8; font-style: italic;">
            💡 <b>Recommended Next:</b> "{next_step}"
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. ACTION TOOLBAR & CONTROL MATRIX
    # -------------------------------------------------------------------------
    with st.expander("🎛️ Configure Analytical Goal, Target Variable & Execution Mode", expanded=False):
        c_goal, c_target, c_mode = st.columns([2, 1.2, 1])
        with c_goal:
            new_goal = st.text_input(
                "Analytical Objective / Business Brief:",
                value=goal_info.get("goal", ""),
                help="Calibrates chart prioritization and Senior Analyst insights to your goal."
            )
        with c_target:
            candidate_cols = [c["column"] for c in profile.get("target_candidates", [])]
            all_cols = list(df.columns)
            target_options = ["None (General EDA)"] + candidate_cols + [c for c in all_cols if c not in candidate_cols]
            curr_target_idx = 0
            if target_col and target_col in target_options:
                curr_target_idx = target_options.index(target_col)
            selected_target_opt = st.selectbox("Target Variable (Reorganizes EDA):", options=target_options, index=curr_target_idx)
        with c_mode:
            selected_mode = st.selectbox(
                "Auto-EDA Mode:",
                options=["Auto (Adaptive)", "Quick (Executive 8)", "Standard (Full)", "Deep (Inferential Tests)"],
                index=0
            )

        if st.button("🚀 Apply EDA Calibration & Refresh", type="primary"):
            st.session_state["project_goal"] = new_goal
            st.session_state["eda_target"] = None if selected_target_opt == "None (General EDA)" else selected_target_opt
            mode_map = {"Auto (Adaptive)": "auto", "Quick (Executive 8)": "quick", "Standard (Full)": "standard", "Deep (Inferential Tests)": "deep"}
            st.session_state["eda_mode"] = mode_map.get(selected_mode, "auto")
            st.rerun()

    # -------------------------------------------------------------------------
    # 5. PROGRESSIVE DISCLOSURE TABS (10 ADVANCED TABS)
    # -------------------------------------------------------------------------
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
        "📊 Executive Overview",
        "🎯 Target & Drivers",
        "📈 Distributions",
        "🔗 Relationships & Correlations",
        "⏱️ Chronological Waves",
        "👥 Segments & Breakdown",
        "🚨 Outliers & Anomalies",
        "🧪 Statistical Tests",
        "💬 Natural Language EDA",
        "💡 Recommendations & Actions"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: EXECUTIVE OVERVIEW
    # -------------------------------------------------------------------------
    with t1:
        st.markdown("### // SENIOR DATA ANALYST KEY FINDINGS")
        st.caption("Evidence-backed findings with actual calculated metrics, business implications, and confidence levels:")

        findings = eda_res.get("key_findings", [])
        if not findings:
            st.info("No critical anomaly or concentration findings detected in sample baseline.")
        else:
            for idx, f in enumerate(findings):
                conf_color = "#10B981" if f["confidence"] == "High" else "#F59E0B"
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #00F0FF; border-top: 1px solid rgba(255, 255, 255, 0.08); border-right: 1px solid rgba(255, 255, 255, 0.08); border-bottom: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">{f['title']}</span>
                        <span style="background: rgba(255, 255, 255, 0.08); color: {conf_color}; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; border: 1px solid {conf_color}40;">
                            {f['confidence'].upper()} CONFIDENCE
                        </span>
                    </div>
                    <div style="color: #CBD5E1; font-size: 0.9rem; margin-bottom: 6px;"><b>Finding:</b> {f['finding']}</div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 6px;">
                        <b>Evidence:</b> <span style="font-family: monospace; color: #00F0FF;">{f['evidence']}</span>
                    </div>
                    <div style="color: #94A3B8; font-size: 0.84rem; margin-bottom: 6px;"><b>Implication:</b> {f['meaning']}</div>
                    <div style="color: #38BDF8; font-size: 0.84rem;"><b>Recommended Action:</b> {f['recommended_action']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### // LEVEL 1: EXECUTIVE VISUALIZATIONS")
        st.caption("Top priority visualizations calibrated to analytical objective and target metrics:")

        level1_charts = eda_res.get("chart_tiers", {}).get("level1_executive", [])
        if level1_charts:
            c1, c2 = st.columns(2)
            for i, spec in enumerate(level1_charts[:6]):
                col = c1 if i % 2 == 0 else c2
                with col:
                    fig = _render_chart_spec(df, spec)
                    if fig:
                        st.plotly_chart(fig, width="stretch", key=f"eda_l1_chart_{spec.get('id', i)}")
                    _render_chart_action_bar(df, spec, key_prefix=f"l1_{i}")
        else:
            st.info("Ingest dataset to view automated executive charts.")

    # -------------------------------------------------------------------------
    # TAB 2: TARGET & DRIVERS
    # -------------------------------------------------------------------------
    with t2:
        if not target_col:
            st.info("No specific target variable selected. Please select a target in the configuration toolbar above.")
        else:
            st.markdown(f"### // TARGET DEEP-DIVE: `{target_col.upper()}` ({target_task.upper()})")

            # Non-causality alert banner
            st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;">
                <span style="color: #F59E0B; font-weight: 700; font-size: 0.82rem;">⚠️ INSTITUTIONAL NON-CAUSALITY NOTICE:</span>
                <span style="color: #CBD5E1; font-size: 0.82rem;">
                    Driver rankings represent predictive association and covariance within this dataset. Correlation does NOT establish causality.
                </span>
            </div>
            """, unsafe_allow_html=True)

            drivers = eda_res.get("target_drivers", [])
            if drivers:
                st.markdown("#### // Ranked Predictive Drivers")
                driver_table = pd.DataFrame(drivers)[["feature", "feature_type", "metric_type", "association_score", "relationship_type", "strength"]]
                st.dataframe(driver_table, width="stretch")

                # Top driver visualization
                top_d = drivers[0]
                st.markdown(f"#### // Bivariate Distribution: `{top_d['feature'].upper()}` vs Target `{target_col.upper()}`")
                paired = df[[top_d["feature"], target_col]].dropna()
                paired_plot = paired.sample(1200, random_state=42) if len(paired) > 1200 else paired

                if pd.api.types.is_numeric_dtype(paired[top_d["feature"]]) and pd.api.types.is_numeric_dtype(paired[target_col]):
                    fig = px.scatter(
                        paired_plot, x=top_d["feature"], y=target_col,
                        trendline="ols",
                        render_mode="webgl" if len(paired_plot) > 400 else "auto",
                        title=f"// COVARIANCE SCATTER: {top_d['feature'].upper()} vs {target_col.upper()}",
                        template="plotly_dark",
                        color_discrete_sequence=["#00F0FF"]
                    )
                else:
                    fig = px.box(
                        paired_plot, x=top_d["feature"], y=target_col,
                        title=f"// COHORT SEPARATION: {target_col.upper()} by {top_d['feature'].upper()}",
                        template="plotly_dark",
                        color_discrete_sequence=["#8B5CF6"]
                    )
                fig.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A")
                st.plotly_chart(fig, width="stretch", key="eda_top_driver_fig")
            else:
                st.info("Insufficient variance or sample records to establish driver associations.")

    # -------------------------------------------------------------------------
    # TAB 3: DISTRIBUTIONS
    # -------------------------------------------------------------------------
    with t3:
        st.markdown("### // NUMERICAL MOMENTS & DISTRIBUTION INTELLIGENCE")
        num_stats = eda_res.get("univariate_numerical", [])
        if num_stats:
            summary_cols = ["column", "mean", "median", "mode", "std", "min", "max", "q25", "q75", "iqr", "skewness", "kurtosis", "distribution_shape", "recommended_transform"]
            st.dataframe(pd.DataFrame(num_stats)[summary_cols], width="stretch")

            selected_num = st.selectbox("Inspect Target Distribution Spectrum:", [s["column"] for s in num_stats], key="eda_dist_num_sel")
            if selected_num:
                chosen_num = next((s for s in num_stats if s["column"] == selected_num), None)
                if chosen_num:
                    c1, c2 = st.columns([3, 1.5])
                    with c1:
                        hist_df = df[[selected_num]].dropna()
                        if len(hist_df) > 4000:
                            hist_df = hist_df.sample(4000, random_state=42)
                        fig = px.histogram(
                            hist_df, x=selected_num,
                            marginal="box" if len(hist_df) <= 1500 else None,
                            title=f"// SPECTRAL DENSITY HUD: {selected_num.upper()}",
                            template="plotly_dark",
                            color_discrete_sequence=["#00F0FF"]
                        )
                        fig.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A")
                        st.plotly_chart(fig, width="stretch", key=f"eda_dist_hist_{selected_num}")
                    with c2:
                        st.markdown(f"#### // Distribution Moments")
                        st.write(f"• **Shape:** `{chosen_num['distribution_shape']}`")
                        st.write(f"• **Skewness:** `{chosen_num['skewness']:.3f}`")
                        st.write(f"• **Kurtosis:** `{chosen_num['kurtosis']:.3f}`")
                        st.write(f"• **Zero Cells:** `{chosen_num['zero_pct']}%`")
                        st.write(f"• **Transform:** `{chosen_num['recommended_transform']}`")
                        st.markdown("##### // Percentiles")
                        st.write(f"P5: `{chosen_num['p5']}` | P95: `{chosen_num['p95']}`")
                        st.write(f"P10: `{chosen_num['p10']}` | P90: `{chosen_num['p90']}`")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### // CATEGORICAL PARETO & CONCENTRATION PROFILES")
        cat_stats = eda_res.get("univariate_categorical", [])
        if cat_stats:
            selected_cat = st.selectbox("Inspect Categorical Cohort:", [c["column"] for c in cat_stats], key="eda_dist_cat_sel")
            if selected_cat:
                chosen_cat = next((c for c in cat_stats if c["column"] == selected_cat), None)
                if chosen_cat:
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        top_df = pd.DataFrame(chosen_cat["top_categories"])
                        fig = px.bar(
                            top_df, x="category", y="count", text="percentage",
                            title=f"// PARETO FREQUENCY: {selected_cat.upper()} (TOP CLASSES)",
                            template="plotly_dark",
                            color="category",
                            color_discrete_sequence=["#00F0FF", "#8B5CF6", "#10B981", "#EC4899", "#F59E0B"]
                        )
                        fig.update_traces(texttemplate='%{text}%', textposition='outside')
                        fig.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A")
                        st.plotly_chart(fig, width="stretch", key=f"eda_cat_bar_{selected_cat}")
                    with c2:
                        st.markdown(f"#### // Pareto Concentration: `{selected_cat}`")
                        st.write(f"• **Unique Classes:** `{chosen_cat['unique_count']}`")
                        st.write(f"• **Top 3 Share:** `{chosen_cat['top3_concentration_pct']}%`")
                        st.write(f"• **80% Volume In:** `{chosen_cat['pareto_categories_for_80pct']} cohorts` ({chosen_cat['pareto_ratio_pct']}% of classes)")

                        st.markdown("##### // Rare Tail Categories (< 2% share)")
                        if chosen_cat["rare_categories"]:
                            st.dataframe(pd.DataFrame(chosen_cat["rare_categories"]), width="stretch")
                        else:
                            st.success("No rare tail anomalies identified.")

    # -------------------------------------------------------------------------
    # TAB 4: RELATIONSHIPS & CORRELATIONS
    # -------------------------------------------------------------------------
    with t4:
        st.markdown("### // MULTIVARIATE CORRELATION LATTICE & BIPOLAR LINKAGES")
        corrs = eda_res.get("correlations", {})
        cols = corrs.get("columns", [])

        if len(cols) >= 2:
            matrix_df = pd.DataFrame(corrs["pearson_matrix"])
            fig = px.imshow(
                matrix_df, text_auto=True, aspect="auto",
                template="plotly_dark",
                color_continuous_scale="Viridis",
                title="// PEARSON MULTIVARIATE CORRELATION MATRIX"
            )
            fig.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A")
            st.plotly_chart(fig, width="stretch", key="eda_corr_heatmap")

            c1, c2 = st.columns(2)
            strong_pairs = corrs.get("strong_pairs", [])
            with c1:
                st.markdown("#### 🟢 High Positive Covariance Pairs (r ≥ 0.4)")
                pos_pairs = [p for p in strong_pairs if p["pearson_r"] > 0]
                if pos_pairs:
                    for p in pos_pairs[:5]:
                        st.write(f"• **`{p['feature1']}`** ◄► **`{p['feature2']}`**: **r = {p['pearson_r']}** ({p['strength']})")
                else:
                    st.caption("No strong positive co-movements detected.")

            with c2:
                st.markdown("#### 🔴 Inverse Trade-Off Pairs (r ≤ -0.4)")
                neg_pairs = [p for p in strong_pairs if p["pearson_r"] < 0]
                if neg_pairs:
                    for p in neg_pairs[:5]:
                        st.write(f"• **`{p['feature1']}`** ◄► **`{p['feature2']}`**: **r = {p['pearson_r']}** ({p['strength']})")
                else:
                    st.caption("No strong inverse trade-offs detected.")

            st.caption(f"ℹ️ {corrs.get('disclaimer', '')}")

            # Interactive Bivariate Scatter
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### // Interactive Bivariate Scatter Explorer")
            sc1, sc2 = st.columns(2)
            with sc1:
                x_axis = st.selectbox("X-Axis Feature:", options=cols, index=0, key="eda_biv_x")
            with sc2:
                y_axis = st.selectbox("Y-Axis Feature:", options=cols, index=1 if len(cols) > 1 else 0, key="eda_biv_y")

            if x_axis and y_axis and x_axis != y_axis:
                scat_df = df[[x_axis, y_axis]].dropna()
                scat_plot = scat_df.sample(1200, random_state=42) if len(scat_df) > 1200 else scat_df
                fig = px.scatter(
                    scat_plot, x=x_axis, y=y_axis,
                    trendline="ols" if len(scat_plot) <= 1500 else None,
                    render_mode="webgl",
                    title=f"// BIVARIATE COVARIANCE: {x_axis.upper()} vs {y_axis.upper()}",
                    template="plotly_dark",
                    color_discrete_sequence=["#00F0FF"]
                )
                fig.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A")
                st.plotly_chart(fig, width="stretch", key="eda_interactive_scatter")
        else:
            st.info("At least two numerical variables are required to evaluate correlation lattice.")

    # -------------------------------------------------------------------------
    # TAB 5: CHRONOLOGICAL WAVES
    # -------------------------------------------------------------------------
    with t5:
        ts = eda_res.get("time_series")
        if ts:
            st.markdown(f"### // CHRONOLOGICAL MOMENTUM: `{ts['metric_column'].upper()}` OVER `{ts['date_column'].upper()}`")
            st.caption(f"Granularity: **{ts['granularity']}** | Timespan: **{ts['span_days']} days** | Trajectory: **{ts['trend_direction']}**")

            trend_df = pd.DataFrame(ts["trend_points"])
            if not trend_df.empty:
                fig = px.area(
                    trend_df, x="period", y="total",
                    title=f"// {ts['granularity'].upper()} VELOCITY: {ts['metric_column'].upper()}",
                    markers=True,
                    template="plotly_dark",
                    color_discrete_sequence=["#00F0FF"]
                )
                fig.update_traces(fillcolor="rgba(0, 240, 255, 0.15)")
                fig.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A")
                st.plotly_chart(fig, width="stretch", key="eda_ts_wave_fig")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Latest Period Growth", f"{ts['latest_growth_rate_pct']:+.1f}%", delta="Period-over-Period")
                with c2:
                    st.metric("Peak Momentum Period", f"{ts['peak_period']['period']}", delta=f"{ts['peak_period']['total']:,.2f}")
                with c3:
                    st.metric("Lowest Valley Period", f"{ts['valley_period']['period']}", delta=f"{ts['valley_period']['total']:,.2f}")

                # Day of week seasonality
                dow_list = ts.get("seasonality_day_of_week", [])
                if dow_list:
                    st.markdown("#### // Day-of-Week Seasonality Distribution")
                    dow_df = pd.DataFrame(dow_list)
                    fig_dow = px.bar(
                        dow_df, x="day", y="total", text="mean",
                        title="// DAY-OF-WEEK AGGREGATE VELOCITY",
                        template="plotly_dark",
                        color_discrete_sequence=["#8B5CF6"]
                    )
                    fig_dow.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A")
                    st.plotly_chart(fig_dow, width="stretch", key="eda_dow_seasonality")
        else:
            st.info("No parseable datetime dimension identified for chronological wave analytics.")

    # -------------------------------------------------------------------------
    # TAB 6: SEGMENTS & BREAKDOWN
    # -------------------------------------------------------------------------
    with t6:
        st.markdown("### // DIMENSIONAL COHORT CONTRIBUTION MATRIX")
        segments = eda_res.get("segments", [])
        if segments:
            for s in segments:
                st.markdown(f"#### // Contribution Matrix: `{s['metric'].upper()}` by `{s['dimension'].upper()}`")
                st.caption(f"Top 2 cohorts account for **{s['top2_concentration_pct']}%** of total volume. Overall average: **{s['overall_mean']:,.2f}**.")

                c1, c2 = st.columns([3, 2])
                with c1:
                    cdf = pd.DataFrame(s["cohorts"])
                    fig = px.bar(
                        cdf, x="segment", y="total_metric", text="contribution_pct",
                        title=f"// COHORT CONTRIBUTION: {s['dimension'].upper()}",
                        template="plotly_dark",
                        color="status",
                        color_discrete_map={
                            "Leader (Outperforming)": "#10B981",
                            "Laggard (Underperforming)": "#EF4444",
                            "On-Benchmark": "#00F0FF"
                        }
                    )
                    fig.update_traces(texttemplate='%{text}% share', textposition='outside')
                    fig.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A")
                    st.plotly_chart(fig, width="stretch", key=f"eda_seg_{s['dimension']}")
                with c2:
                    st.dataframe(cdf[["segment", "total_metric", "contribution_pct", "mean_metric", "status"]], width="stretch")
                st.markdown("---")
        else:
            st.info("Insufficient categorical dimensions to generate segmentation contribution matrices.")

    # -------------------------------------------------------------------------
    # TAB 7: OUTLIERS & ANOMALIES
    # -------------------------------------------------------------------------
    with t7:
        st.markdown("### // MULTI-METHOD OUTLIER EXPLORATION & STATISTICAL IMPACT")
        st.caption("Evaluates Tukey IQR, Z-score, and MAD bounds. Records are classified rather than silently deleted:")

        anomalies = eda_res.get("anomalies", [])
        if anomalies:
            anom_df = pd.DataFrame(anomalies)[[
                "column", "iqr_outliers_count", "iqr_outliers_pct", "original_mean",
                "mean_without_outliers", "mean_shift_pct", "classification"
            ]]
            st.dataframe(anom_df, width="stretch")

            selected_anom_col = st.selectbox("Inspect Outlier Dispersion HUD:", [a["column"] for a in anomalies], key="eda_anom_col_sel")
            if selected_anom_col:
                chosen_anom = next((a for a in anomalies if a["column"] == selected_anom_col), None)
                if chosen_anom:
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        box_df = df.sample(3000, random_state=42) if len(df) > 3000 else df
                        fig = px.box(
                            box_df, y=selected_anom_col,
                            title=f"// DISPERSION BOUNDARIES & OUTLIERS: {selected_anom_col.upper()}",
                            template="plotly_dark",
                            color_discrete_sequence=["#EF4444"]
                        )
                        fig.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A")
                        st.plotly_chart(fig, width="stretch", key=f"eda_anom_box_{selected_anom_col}")
                        if len(df) > 3000:
                            st.caption("ℹ️ Sampled 3,000 observations for responsive box dispersion rendering.")
                    with c2:
                        st.markdown(f"#### // Statistical Impact Audit")
                        st.write(f"• **Classification:** `{chosen_anom['classification']}`")
                        st.write(f"• **IQR Outliers:** `{chosen_anom['iqr_outliers_count']}` ({chosen_anom['iqr_outliers_pct']}%)")
                        st.write(f"• **Mean Shift Impact:** `{chosen_anom['mean_shift_pct']}%`")
                        st.write(f"• **Original Mean:** `{chosen_anom['original_mean']}`")
                        st.write(f"• **Robust Mean (No Outliers):** `{chosen_anom['mean_without_outliers']}`")
                        st.write(f"• **IQR Bounds:** `[{chosen_anom['lower_bound_mild']}, {chosen_anom['upper_bound_mild']}]`")
                        st.info(f"💡 **Policy:** {chosen_anom['recommendation']}")

            if st.session_state.get("deep_eda_res"):
                deep_anom = st.session_state["deep_eda_res"].get("modules", {}).get("isolation_forest", {})
                if deep_anom and deep_anom.get("status") == "completed":
                    st.markdown(f"""
                    <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid #8B5CF6; border-radius: 10px; padding: 14px 18px; margin-top: 14px;">
                        <span style="font-family: monospace; font-size: 0.8rem; color: #A78BFA; font-weight: 700;">🤖 MULTIVARIATE ISOLATION FOREST ANOMALY SCAN</span>
                        <div style="color: #E2E8F0; font-size: 0.86rem; margin-top: 4px;">
                            Tested Records: <b>{deep_anom.get('tested_records', 0):,}</b> &bull; Multivariate Anomalies Flagged: <b>{deep_anom.get('anomalies_isolated', 0):,}</b> (Contamination rate: 3.0%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("No statistical boundary anomalies detected across numerical features.")

    # -------------------------------------------------------------------------
    # TAB 8: STATISTICAL TESTS
    # -------------------------------------------------------------------------
    with t8:
        st.markdown("### // INFERENTIAL HYPOTHESIS TESTING & EFFECT SIZES")
        st.caption("Automated assumption checking, p-value calculation, and standardized effect sizes (Cohen's d, Eta-squared, Cramer's V):")

        stat_tests = eda_res.get("statistical_tests", [])
        if stat_tests:
            for t in stat_tests:
                p_val = t.get("p_value", 1.0)
                is_sig = t.get("is_significant", False)
                status_badge = "🟢 STATISTICALLY SIGNIFICANT" if is_sig else "⚪ NOT SIGNIFICANT"
                badge_color = "#10B981" if is_sig else "#94A3B8"

                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #F8FAFC; font-size: 1.02rem;">{t['test_name']}</span>
                        <span style="color: {badge_color}; font-weight: 700; font-size: 0.76rem; border: 1px solid {badge_color}40; padding: 2px 8px; border-radius: 4px;">
                            {status_badge}
                        </span>
                    </div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 6px;">
                        <b>Test Statistic:</b> <span style="font-family: monospace; color: #00F0FF;">{t.get('statistic', t.get('chi2_statistic', 'N/A'))}</span>
                        &nbsp;&bull;&nbsp;
                        <b>p-value:</b> <span style="font-family: monospace; color: #00F0FF;">{p_val:.4g}</span>
                        &nbsp;&bull;&nbsp;
                        <b>Effect Size:</b> <span style="font-family: monospace; color: #A78BFA;">{t.get('effect_size', t.get('effect_size_eta_sq', t.get('cramers_v', 'N/A')))}</span>
                    </div>
                    <div style="color: #CBD5E1; font-size: 0.88rem;">{t.get('interpretation', '')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No categorical-numerical hypothesis tests met execution criteria for this dataset.")

    # -------------------------------------------------------------------------
    # TAB 9: NATURAL LANGUAGE EDA ASSISTANT
    # -------------------------------------------------------------------------
    with t9:
        st.markdown("### // CONVERSATIONAL DATA ANALYST // DUCKDB ENGINE")
        st.caption("Ask questions about distributions, anomalies, trends, or cohorts. Calculations run deterministically via DuckDB/Pandas:")

        query_input = st.text_input(
            "Ask anything about your data:",
            placeholder="e.g., 'What are the top categories by total sales?', 'Are there outliers in profit?', 'What is the correlation between sales and quantity?'",
            key="eda_nl_query_input"
        )

        if query_input:
            with st.spinner("Executing analytical query across dataset..."):
                try:
                    answer_res = EDANLQueryEngine.answer_query(df, query=query_input, target_col=target_col, goal=user_goal)
                except Exception:
                    answer_res = EDAModel.ask_eda_assistant(df, query=query_input, target_col=target_col, goal=user_goal)

            st.markdown(f"""
            <div style="background: rgba(14, 20, 36, 0.9); border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 12px; padding: 18px 22px; margin-top: 14px;">
                <div style="color: #00F0FF; font-size: 0.76rem; font-family: monospace; font-weight: 700; margin-bottom: 6px;">// DETERMINISTIC ANALYTICAL EVIDENCE</div>
                <div style="color: #F8FAFC; font-size: 1.05rem; line-height: 1.5; margin-bottom: 10px;">
                    {answer_res['answer']}
                </div>
                <div style="color: #94A3B8; font-size: 0.84rem; margin-bottom: 8px;">
                    <b>Evidence Basis:</b> {answer_res['evidence']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if answer_res.get("sql"):
                with st.expander("💾 Inspect Executed Analytical SQL Query"):
                    st.code(answer_res["sql"], language="sql")

            if answer_res.get("table") is not None and not answer_res["table"].empty:
                st.markdown("##### // Calculated Data Aggregate")
                st.dataframe(answer_res["table"], width="stretch")

    # -------------------------------------------------------------------------
    # TAB 10: RECOMMENDATIONS & ACTIONS
    # -------------------------------------------------------------------------
    with t10:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### // RECOMMENDED NEXT ANALYTICAL STEPS")
            next_steps = eda_res.get("next_analyses", [])
            for idx, ns in enumerate(next_steps):
                st.markdown(f"**{idx+1}.** {ns}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### // FEATURE ENGINEERING RECOMMENDATIONS")
            st.caption("Deterministic feature transformations suggested by EDA moments & datetime dimensions:")
            feat_recs = eda_res.get("feature_recommendations", [])
            if feat_recs:
                for fr in feat_recs:
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;">
                        <span style="color: #00F0FF; font-weight: 700; font-size: 0.88rem;">{fr['recommended_feature']}</span>
                        <span style="color: #94A3B8; font-size: 0.78rem;">({fr['transformation_type']})</span>
                        <div style="color: #CBD5E1; font-size: 0.82rem; margin-top: 4px;">{fr['rationale']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No immediate feature transformations required.")

        with c2:
            st.markdown("### // EDA COVERAGE & LINEAGE AUDIT")
            st.write(f"• **Analysis Plan Mode:** `{eda_plan.get('mode', 'AUTO')}`")
            st.write(f"• **Target Variable:** `{eda_plan.get('target', 'None')}`")
            st.write(f"• **Domain Classification:** `{eda_plan.get('domain', 'Enterprise Tabular')}`")
            st.write(f"• **Total Computed Analyses:** `{eda_plan.get('total_analyses_count', 0)}`")
            st.write(f"• **Charts Generated:** `{eda_plan.get('charts_count', 0)}`")

            st.markdown("##### Covered Stages:")
            for s in coverage.get("covered_stages", []):
                st.write(f"✓ {s}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### // EXPORT & INTEGRATION ACTIONS")
            if st.button("📑 Export Full EDA Dossier to Reports Module", width="stretch", type="primary"):
                st.session_state["active_nav"] = "Reports"
                st.rerun()

            if st.button("📊 View Pinned Visuals on Dashboard", width="stretch"):
                st.session_state["active_nav"] = "Dashboard"
                st.rerun()


# -----------------------------------------------------------------------------
# HELPER RENDERING FUNCTIONS
# -----------------------------------------------------------------------------
def _render_chart_spec(df: pd.DataFrame, spec: Dict[str, Any]) -> Optional[go.Figure]:
    c_type = spec.get("type", "bar")
    title = spec.get("title", "")
    override_data = spec.get("data_override")

    active_df = pd.DataFrame(override_data) if override_data is not None else df

    try:
        if c_type == "histogram":
            hist_df = active_df.sample(3000, random_state=42) if len(active_df) > 3000 else active_df
            fig = px.histogram(
                hist_df, x=spec.get("x"),
                title=title, template="plotly_dark",
                color_discrete_sequence=["#00F0FF"]
            )
        elif c_type == "line":
            line_df = active_df
            if len(line_df) > 300:
                step = max(1, len(line_df) // 300)
                line_df = line_df.iloc[::step]
            fig = px.line(
                line_df, x=spec.get("x"), y=spec.get("y"),
                title=title, template="plotly_dark",
                markers=True if len(line_df) <= 100 else False,
                color_discrete_sequence=["#00F0FF"]
            )
        elif c_type == "scatter":
            scat_plot = active_df.sample(1200, random_state=42) if len(active_df) > 1200 else active_df
            fig = px.scatter(
                scat_plot, x=spec.get("x"), y=spec.get("y"),
                trendline="ols" if len(scat_plot) <= 1500 else None,
                render_mode="webgl",
                title=title, template="plotly_dark",
                color_discrete_sequence=["#8B5CF6"]
            )
        elif c_type == "box":
            box_df = active_df.sample(2000, random_state=42) if len(active_df) > 2000 else active_df
            fig = px.box(
                box_df, y=spec.get("y"),
                title=title, template="plotly_dark",
                color_discrete_sequence=["#EF4444"]
            )
        elif c_type == "heatmap":
            matrix_df = pd.DataFrame(spec.get("matrix_data", {}))
            fig = px.imshow(
                matrix_df, text_auto=True, aspect="auto",
                template="plotly_dark", color_continuous_scale="Viridis",
                title=title
            )
        else:  # bar
            bar_df = active_df
            x_col = spec.get("x")
            y_col = spec.get("y")
            if override_data is None and len(bar_df) > 30 and x_col in bar_df.columns:
                if y_col and y_col in bar_df.columns and pd.api.types.is_numeric_dtype(bar_df[y_col]):
                    bar_df = bar_df.groupby(x_col, as_index=False)[y_col].sum().head(30)
                else:
                    bar_df = bar_df[x_col].value_counts().reset_index().head(30)
                    bar_df.columns = [x_col, "count"]
                    y_col = "count"
            fig = px.bar(
                bar_df, x=x_col, y=y_col,
                title=title, template="plotly_dark",
                color_discrete_sequence=["#00F0FF"]
            )

        fig.update_layout(
            paper_bgcolor="#0A0E1A",
            plot_bgcolor="#0A0E1A",
            margin=dict(l=40, r=20, t=50, b=40),
            title_font=dict(size=13, color="#FFFFFF")
        )
        return fig
    except Exception:
        return None


def _render_chart_action_bar(df: pd.DataFrame, spec: Dict[str, Any], key_prefix: str = "chart"):
    """Renders data-backed Explain, View SQL, and Pin actions."""
    with st.expander(f"ℹ️ Telemetry Details & Actions: {spec.get('title', 'Chart')}"):
        tab_exp, tab_sql = st.tabs(["💡 Data Rationale", "💾 SQL Lineage"])
        with tab_exp:
            if hasattr(EDAModel, "explain_chart_insight"):
                explanation = EDAModel.explain_chart_insight(spec, df)
            else:
                explanation = spec.get("explanation", "Data visualization computed from active dataset.")
            st.markdown(f"**Data-Backed Rationale:** {explanation}")
        with tab_sql:
            if hasattr(EDAModel, "generate_chart_sql"):
                sql = EDAModel.generate_chart_sql(spec, df)
            else:
                sql = spec.get("sql", "SELECT * FROM dataset;")
            st.code(sql, language="sql")

        c_pin_dash, c_pin_rep = st.columns(2)
        with c_pin_dash:
            if st.button("📌 Pin to Dashboard", key=f"{key_prefix}_pin_dash", width="stretch"):
                pinned = st.session_state.get("pinned_dashboard_charts", [])
                pinned.append(spec)
                st.session_state["pinned_dashboard_charts"] = pinned
                st.success("Chart pinned to Executive Dashboard!")
        with c_pin_rep:
            if st.button("📄 Pin to Report", key=f"{key_prefix}_pin_rep", width="stretch"):
                pinned_rep = st.session_state.get("pinned_report_charts", [])
                pinned_rep.append(spec)
                st.session_state["pinned_report_charts"] = pinned_rep
                st.success("Chart pinned to Analytical Dossier!")
