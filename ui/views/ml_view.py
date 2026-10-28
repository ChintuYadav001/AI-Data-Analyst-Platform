"""
ui.views.ml_view
================
Institutional-grade Machine Learning & Predictive Analytics Cockpit for Quantum Analyst.
Provides a 13-tab interactive suite:
1. Overview
2. ML Readiness
3. Target
4. Features & Selection
5. Dimensionality (PCA & Beyond)
6. Models & Config
7. Training & AutoML
8. Evaluation & Diagnostics
9. Explainable AI
10. Predictions & Inference
11. Future Forecasting
12. Scenario Analysis (What-If)
13. Experiments & Registry
"""

import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.ml.ml_readiness import audit_ml_readiness, classify_task_type, detect_target_candidates
from core.ml.model_registry import REGISTRY, get_candidate_models
from core.ml.model_selection import run_benchmark
from core.ml.trainer import train_single_model
from core.ml.tuning import tune_model_hyperparameters
from core.ml.forecasting import run_time_series_forecast, auto_detect_time_series_meta
from core.ml.prediction import predict_single_record, predict_batch
from core.ml.explainability import explain_local_prediction, STATISTICAL_DISCLAIMER
from core.ml.dimensionality import run_dimensionality_reduction, METHOD_METADATA
from core.ml.feature_selection import select_best_features, FEATURE_SELECTION_METHODS
from core.ml.anomaly_detection import run_anomaly_detection, ANOMALY_ALGORITHMS
from core.ml.clustering import run_clustering, CLUSTERING_ALGORITHMS
from core.ml.scenario_analysis import simulate_what_if_scenario
from core.ml.report_generator import generate_ml_pdf_report, generate_ml_html_report, generate_ml_csv_summary
from core.ml.experiment_tracker import (
    save_experiment,
    list_experiments,
    get_experiment,
    load_model,
    set_active_champion,
    get_active_champion,
)


def init_ml_session_state():
    """Ensure all required session state keys for ML view exist."""
    defaults = {
        "ml_target_col": None,
        "ml_selected_features": [],
        "ml_task_type": "auto",
        "ml_compute_profile": "Balanced",
        "ml_selected_models": [],
        "ml_test_size": 0.20,
        "ml_cv_folds": 3,
        "ml_scaler": "standard",
        "ml_cat_encoding": "onehot",
        "ml_benchmark_results": None,
        "ml_active_model_id": None,
        "ml_last_training_time": None,
        "ml_forecast_results": None,
        "ml_pca_results": None,
        "ml_clustering_results": None,
        "ml_fs_results": None,
        "ml_anomaly_results": None,
        "ml_scenario_results": None,
        "ml_last_dataset_signature": None,
        "ml_autonomous_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_ml_view():
    init_ml_session_state()

    # Verify dataset loaded
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No active dataset found in workspace. Please upload a dataset or select a sample dataset first.")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("📂 Go to Upload Dataset", type="primary", width="stretch"):
                st.session_state["active_nav"] = "Upload Dataset"
                st.rerun()
        return

    df: pd.DataFrame = st.session_state["df"]
    proj_name = st.session_state.get("current_project_name", "Active Dataset")

    # Dynamic Dataset Signature Check
    current_signature = f"{proj_name}_{len(df)}_{tuple(df.columns)}"
    if st.session_state.get("ml_last_dataset_signature") != current_signature:
        st.session_state["ml_last_dataset_signature"] = current_signature
        st.session_state["ml_target_col"] = None
        st.session_state["ml_selected_features"] = []
        st.session_state["ml_benchmark_results"] = None
        st.session_state["ml_active_model_id"] = None
        st.session_state["ml_forecast_results"] = None
        st.session_state["ml_pca_results"] = None
        st.session_state["ml_clustering_results"] = None
        st.session_state["ml_fs_results"] = None
        st.session_state["ml_anomaly_results"] = None
        st.session_state["ml_scenario_results"] = None

    # Auto-initialize target, features, and task type if not yet set so all tabs work immediately
    if not st.session_state.get("ml_target_col") and len(df.columns) > 1:
        candidates = detect_target_candidates(df)
        st.session_state["ml_target_col"] = candidates[0].column if candidates else df.columns[-1]

    if not st.session_state.get("ml_selected_features"):
        target_c = st.session_state.get("ml_target_col")
        feature_candidates = [c for c in df.columns if c != target_c]
        st.session_state["ml_selected_features"] = feature_candidates[:min(15, len(feature_candidates))]

    if not st.session_state.get("ml_task_type") or st.session_state["ml_task_type"] == "auto":
        if st.session_state.get("ml_target_col"):
            st.session_state["ml_task_type"] = classify_task_type(df, st.session_state["ml_target_col"])

    # Header section
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px;">
        <div>
            <div style="font-size: 0.76rem; font-weight: 800; color: #00F0FF; letter-spacing: 0.08em; text-transform: uppercase;">
                // INSTITUTIONAL AUTOMATED ML & PREDICTIVE ANALYTICS
            </div>
            <h2 style="margin: 2px 0 6px 0; color: #FFFFFF; font-size: 1.7rem; font-weight: 800;">
                🤖 Quantum Predictive Engine
            </h2>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                End-to-end model benchmarking, PCA & decomposition, explainable AI, future forecasting, and What-If simulations.
            </div>
        </div>
        <div style="text-align: right; background: rgba(0, 240, 255, 0.06); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 6px 14px;">
            <div style="font-size: 0.68rem; font-weight: 700; color: #38BDF8;">ACTIVE WORKSPACE</div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">{proj_name}</div>
            <div style="font-size: 0.74rem; color: #94A3B8;">{len(df):,} rows × {len(df.columns)} cols</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AI Model Copilot Quick Access Ribbon
    c_banner1, c_banner2 = st.columns([7.8, 2.2])
    with c_banner1:
        st.markdown("""
        <div style="background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 10px; padding: 8px 14px; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.2rem;">🎙️</span>
            <div>
                <span style="font-weight: 700; color: #FFFFFF; font-size: 0.88rem;">AI Model Copilot & Voice Assistant Active</span>
                <span style="color: #94A3B8; font-size: 0.8rem; margin-left: 8px;">Ask about any model in simple words, PCA analogies, or listen with voice.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_banner2:
        if st.button("🤖 Ask AI Copilot", key="ml_cockpit_open_copilot_btn", width="stretch", type="primary"):
            st.session_state["copilot_open"] = True
            st.session_state["active_nav"] = "AI Model Copilot"
            st.rerun()

    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    # 14 Institutional Tabs (with 1-Click Autonomous Lifecycle as Flagship Tab)
    tabs_list = st.tabs([
        "⚡ 1-Click Autonomous ML",
        "Overview",
        "ML Readiness",
        "Target",
        "Features & Selection",
        "Dimensionality & Clustering",
        "Models & Config",
        "Training & AutoML",
        "Evaluation & Diagnostics",
        "Explainable AI",
        "Predictions & Inference",
        "Future Forecasting",
        "Scenario Analysis (What-If)",
        "Experiments & Registry",
    ])

    # =========================================================================
    # FLAGSHIP TAB: 1-CLICK AUTONOMOUS PREDICTIVE PIPELINE (ALL 10 STEPS)
    # =========================================================================
    with tabs_list[0]:
        from ui.views.autonomous_ml_card import render_autonomous_1click_ml_tab
        render_autonomous_1click_ml_tab(df, proj_name)

    tabs = tabs_list[1:]

    # =========================================================================
    # TAB 1: OVERVIEW
    # =========================================================================
    with tabs[0]:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1E293B; border-radius: 10px; padding: 16px; margin-bottom: 16px;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 1.05rem; margin-bottom: 6px;">
                🚀 Complete Machine Learning & Predictive Workflow
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
                Quantum Analyst provides an automated, statistically rigorous modeling pipeline:
                <b>Dataset → Type → Target → Features → AutoML / Manual → Validation / Compute → Train</b>.
                Every metric is computed directly from holdout splits with zero data leakage.
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Records", f"{len(df):,}")
        with m2:
            st.metric("Total Columns", f"{len(df.columns):,}")
        with m3:
            num_c = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Features", f"{num_c}")
        with m4:
            cat_c = len(df.select_dtypes(include=["object", "category"]).columns)
            st.metric("Categorical Features", f"{cat_c}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Quick Actions Card
        st.subheader("⚡ Quick Navigation Stepper")
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        with col_w1:
            st.markdown("##### 1. Audit & Target")
            st.caption("Verify data health and configure prediction target.")
        with col_w2:
            st.markdown("##### 2. Feature Selection & PCA")
            st.caption("Prune noise, decompose dimensions, and inspect scree curves.")
        with col_w3:
            st.markdown("##### 3. Benchmark Models")
            st.caption("Train competing estimators and identify champion model.")
        with col_w4:
            st.markdown("##### 4. Forecast & Scenarios")
            st.caption("Generate future projections and counterfactual What-If tests.")

        st.markdown("<br>", unsafe_allow_html=True)
        q_c1, q_c2 = st.columns([3, 1])
        with q_c1:
            st.info("💡 **Have questions about how any ML model works or which model to pick?** Ask our AI Copilot & Voice Assistant on the bottom right!")
        with q_c2:
            if st.button("🤖 Open AI Model Copilot", type="primary", width="stretch"):
                st.session_state["copilot_open"] = True
                st.rerun()

    # =========================================================================
    # TAB 2: ML READINESS
    # =========================================================================
    with tabs[1]:
        st.subheader("📋 Dataset ML Health & Readiness Audit")
        report = audit_ml_readiness(df, st.session_state.get("ml_target_col"))

        col_score, col_details = st.columns([1, 3])
        with col_score:
            score = report.overall_score
            color = "#10B981" if score >= 80 else ("#F59E0B" if score >= 60 else "#EF4444")
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.8); border: 2px solid {color}; border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">Readiness Score</div>
                <div style="font-size: 3.2rem; font-weight: 800; color: {color}; margin: 6px 0;">{score}<span style="font-size: 1.2rem; color: #64748B;">/100</span></div>
                <div style="font-size: 0.85rem; font-weight: 600; color: {color};">
                    {"PRODUCTION READY" if score >= 80 else ("ACCEPTABLE" if score >= 60 else "NEEDS ATTENTION")}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_details:
            st.markdown("##### Data Composition & Quality Checks")
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                st.write(f"• **Missing Values:** {df.isna().sum().sum():,} cells")
                st.write(f"• **Constant Columns:** {len(report.constant_columns)}")
            with c_b:
                st.write(f"• **High Cardinality:** {len(report.high_cardinality_columns)}")
                st.write(f"• **Identifier Columns:** {len(report.id_columns)}")
            with c_c:
                st.write(f"• **Leakage Risks:** {len(report.leakage_warnings)}")
                dup_estimate = int(df.sample(min(30000, len(df)), random_state=42).duplicated().sum() * (len(df) / min(30000, len(df)))) if len(df) > 30000 else int(df.duplicated().sum())
                st.write(f"• **Duplicates:** ~{dup_estimate:,} rows")

        if report.warnings:
            st.markdown("<br>", unsafe_allow_html=True)
            st.warning("**⚠️ Statistical & Quality Warnings:**\n" + "\n".join([f"- {w}" for w in report.warnings]))

        if report.recommendations:
            st.info("**💡 Recommended Actions:**\n" + "\n".join([f"- {r}" for r in report.recommendations]))

    # =========================================================================
    # TAB 3: TARGET
    # =========================================================================
    with tabs[2]:
        st.subheader("🎯 Target Variable & Problem Formulation")
        st.caption("Select the prediction target. The platform automatically infers task type, but you can manually override.")

        # Candidate discovery
        candidates = detect_target_candidates(df)
        cand_names = [c.column for c in candidates]

        all_cols = list(df.columns)
        default_idx = 0
        if st.session_state.get("ml_target_col") in all_cols:
            default_idx = all_cols.index(st.session_state["ml_target_col"])
        elif cand_names and cand_names[0] in all_cols:
            default_idx = all_cols.index(cand_names[0])

        col_t1, col_t2 = st.columns([2, 2])
        with col_t1:
            selected_target = st.selectbox(
                "Prediction Target Column (y):",
                options=all_cols,
                index=default_idx,
                help="The column you want the model to predict.",
            )
            st.session_state["ml_target_col"] = selected_target

        with col_t2:
            auto_task = classify_task_type(df, selected_target)
            task_options = [
                "auto (Infer Automatically)",
                "regression",
                "binary_classification",
                "multiclass_classification",
                "clustering",
                "anomaly",
                "forecasting",
            ]
            task_selection = st.selectbox(
                "Task Formulation (Override if necessary):",
                options=task_options,
                index=0,
            )
            if task_selection.startswith("auto"):
                st.session_state["ml_task_type"] = auto_task
            else:
                st.session_state["ml_task_type"] = task_selection

        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.05); border-left: 3px solid #38BDF8; padding: 10px 14px; border-radius: 4px; margin: 12px 0;">
            <b>Active Task Formulation:</b> <code>{st.session_state['ml_task_type'].replace('_', ' ').title()}</code>
            &nbsp;|&nbsp; Target: <code>{selected_target}</code>
            ({df[selected_target].nunique()} unique values, dtype: {df[selected_target].dtype})
        </div>
        """, unsafe_allow_html=True)

        # Visualizing Target Distribution (sampled for large datasets)
        plot_df = df.sample(min(30000, len(df)), random_state=42) if len(df) > 30000 else df
        if pd.api.types.is_numeric_dtype(df[selected_target]) and df[selected_target].nunique() > 10:
            fig_t = px.histogram(
                plot_df,
                x=selected_target,
                nbins=30,
                title=f"Target Distribution: {selected_target} ({'sampled' if len(df) > 30000 else 'full'})",
                color_discrete_sequence=["#00F0FF"],
            )
            fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
            st.plotly_chart(fig_t, width="stretch")
        else:
            vc = df[selected_target].value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            fig_t = px.bar(
                vc,
                x="Class",
                y="Count",
                title=f"Target Class Balance: {selected_target}",
                color="Count",
                color_continuous_scale="Viridis",
            )
            fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
            st.plotly_chart(fig_t, width="stretch")

    # =========================================================================
    # TAB 4: FEATURES & SELECTION
    # =========================================================================
    with tabs[3]:
        st.subheader("⚡ Feature Selection & Engineering Engine")
        st.caption("Prune redundant attributes, evaluate predictive relevance, and prevent multicollinearity.")

        target_c = st.session_state.get("ml_target_col")
        feature_candidates = [c for c in df.columns if c != target_c]

        # Multi-select for features
        sel_features = st.multiselect(
            "Selected Feature Set for Modeling:",
            options=feature_candidates,
            default=[c for c in feature_candidates if c in st.session_state.get("ml_selected_features", [])] or feature_candidates[:min(12, len(feature_candidates))],
            help="Select columns to include in training pipeline.",
        )
        st.session_state["ml_selected_features"] = sel_features

        st.markdown("---")
        st.markdown("#### 🔬 Statistical Feature Selection")
        fs_col1, fs_col2, fs_col3 = st.columns([2, 1, 1])
        with fs_col1:
            fs_method = st.selectbox(
                "Selection Metric:",
                options=list(FEATURE_SELECTION_METHODS.keys()),
                format_func=lambda k: FEATURE_SELECTION_METHODS[k],
                index=0,
            )
        with fs_col2:
            top_k = st.slider("Top K Features:", min_value=2, max_value=max(2, len(feature_candidates)), value=min(8, len(feature_candidates)))
        with fs_col3:
            st.write("")
            st.write("")
            run_fs_btn = st.button("🚀 Evaluate Features", type="primary", width="stretch")

        if run_fs_btn and target_c:
            with st.spinner("Executing statistical feature evaluation..."):
                try:
                    fs_res = select_best_features(
                        df=df,
                        target_col=target_c,
                        method=fs_method,
                        k=top_k,
                        task_type=st.session_state.get("ml_task_type"),
                    )
                    st.session_state["ml_fs_results"] = fs_res
                    st.session_state["ml_selected_features"] = fs_res["selected_features"]
                    st.success(fs_res["summary"])
                except Exception as e:
                    st.error(f"Feature selection failed: {str(e)}")

        # Render Feature Selection Leaderboard
        fs_res = st.session_state.get("ml_fs_results")
        if fs_res:
            st.markdown(f"**Ranking Leaderboard ({fs_res['method_name']}):**")
            df_rank = pd.DataFrame(fs_res["ranked_table"])
            fig_fs = px.bar(
                df_rank,
                x="score",
                y="feature",
                orientation="h",
                color="selected",
                color_discrete_map={True: "#00F0FF", False: "#64748B"},
                title=f"Feature Predictive Scores ({fs_res['method_name']})",
            )
            fig_fs.update_layout(
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font_color="#E2E8F0",
            )
            st.plotly_chart(fig_fs, width="stretch")

    # =========================================================================
    # TAB 5: DIMENSIONALITY (PCA) & ANOMALY DETECTION
    # =========================================================================
    with tabs[4]:
        st.subheader("🌐 Dimensionality (PCA) & Unsupervised Learning Suite")
        st.caption("Linear and non-linear manifold decomposition (PCA, Truncated SVD, Kernel PCA, ICA, NMF, t-SNE), clustering (K-Means, DBSCAN, Agglomerative), & anomaly detection (Isolation Forest, One-Class SVM, LOF, Elliptic Envelope).")

        t5_pca, t5_cluster, t5_anomaly = st.tabs([
            "🌐 PCA & Manifold Decomposition",
            "🔮 Unsupervised Clustering (K-Means, DBSCAN, Agglomerative)",
            "🚨 Unsupervised Anomaly & Outlier Detection",
        ])

        num_features = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.session_state.get("ml_target_col")
        if target_col in num_features:
            num_features.remove(target_col)

        with t5_pca:
            if len(num_features) < 2:
                st.info("Dimensionality reduction requires at least 2 numeric feature columns.")
            else:
                p_col1, p_col2, p_col3 = st.columns([2, 1, 1])
                with p_col1:
                    decomp_method = st.selectbox(
                        "Decomposition Algorithm:",
                        options=list(METHOD_METADATA.keys()),
                        format_func=lambda k: METHOD_METADATA[k]["name"],
                        index=0,
                    )
                with p_col2:
                    n_comps = st.slider("Component Dimensions:", min_value=2, max_value=min(10, len(num_features)), value=2)
                with p_col3:
                    st.write("")
                    st.write("")
                    run_pca_btn = st.button("⚡ Run Decomposition", type="primary", width="stretch")

                if run_pca_btn:
                    with st.spinner("Executing dimensionality reduction & calculating loadings..."):
                        try:
                            pca_res = run_dimensionality_reduction(
                                df=df,
                                feature_cols=num_features,
                                method=decomp_method,
                                n_components=n_comps,
                                target_col=target_col,
                            )
                            st.session_state["ml_pca_results"] = pca_res
                            st.success(f"Successfully decomposed {len(num_features)} features into {n_comps} dimensions using {METHOD_METADATA[decomp_method]['name']}.")
                        except Exception as e:
                            st.error(f"Decomposition error: {str(e)}")

                pca_res = st.session_state.get("ml_pca_results")
                if pca_res:
                    summary = pca_res["comparison_summary"]
                    sm1, sm2, sm3, sm4 = st.columns(4)
                    with sm1:
                        st.metric("Original Features", summary["original_features_count"])
                    with sm2:
                        st.metric("Reduced Dimensions", summary["reduced_dimensions_count"])
                    with sm3:
                        st.metric("Compression Ratio", f"{summary['compression_ratio_pct']}%")
                    with sm4:
                        st.metric("Variance Retained", f"{summary['variance_retained_pct']}%")

                    # Scree Plot (if linear PCA/SVD)
                    if pca_res.get("scree_data"):
                        st.markdown("#### 📉 Scree Plot & Cumulative Variance")
                        scree_df = pd.DataFrame(pca_res["scree_data"])
                        fig_scree = go.Figure()
                        fig_scree.add_trace(go.Bar(
                            x=scree_df["component"],
                            y=scree_df["variance_pct"],
                            name="Explained Variance %",
                            marker_color="#00F0FF",
                        ))
                        fig_scree.add_trace(go.Scatter(
                            x=scree_df["component"],
                            y=scree_df["cumulative_pct"],
                            name="Cumulative Variance %",
                            mode="lines+markers",
                            line=dict(color="#F59E0B", width=2),
                        ))
                        fig_scree.update_layout(
                            title="Explained Variance by Component",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15,23,42,0.6)",
                            font_color="#E2E8F0",
                        )
                        st.plotly_chart(fig_scree, width="stretch")

                    # 2D & 3D Projections
                    st.markdown("#### 🗺️ Low-Dimensional Space Projection")
                    p_tab2d, p_tab3d = st.tabs(["2D Projection Scatter", "3D Interactive Orbit"])
                    with p_tab2d:
                        proj_2d_df = pd.DataFrame(pca_res["proj_2d"])
                        fig_2d = px.scatter(
                            proj_2d_df,
                            x="dim1",
                            y="dim2",
                            color="target",
                            title=f"{pca_res['metadata']['name']} (2D Space)",
                            hover_data=["index"],
                            color_discrete_sequence=px.colors.qualitative.Prism,
                        )
                        fig_2d.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
                        st.plotly_chart(fig_2d, width="stretch")

                    with p_tab3d:
                        if pca_res.get("proj_3d"):
                            proj_3d_df = pd.DataFrame(pca_res["proj_3d"])
                            fig_3d = px.scatter_3d(
                                proj_3d_df,
                                x="dim1",
                                y="dim2",
                                z="dim3",
                                color="target",
                                title=f"{pca_res['metadata']['name']} (3D Manifold)",
                                hover_data=["index"],
                                color_discrete_sequence=px.colors.qualitative.Prism,
                            )
                            fig_3d.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
                            st.plotly_chart(fig_3d, width="stretch")
                        else:
                            st.info("3D projection requires selecting at least 3 component dimensions.")

                    # Loadings Matrix Heatmap
                    if pca_res.get("loadings"):
                        st.markdown("#### 🧬 Component Loadings (Feature Contributions)")
                        loadings_df = pd.DataFrame(pca_res["loadings"])
                        fig_load = px.imshow(
                            loadings_df.T,
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Feature Loadings Matrix ($V^T$)",
                        )
                        fig_load.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
                        st.plotly_chart(fig_load, width="stretch")

        with t5_cluster:
            if len(num_features) < 2:
                st.info("Clustering analysis requires at least 2 numeric feature columns.")
            else:
                cl_col1, cl_col2, cl_col3, cl_col4 = st.columns([2, 1, 1, 1])
                with cl_col1:
                    cl_algo = st.selectbox(
                        "Clustering Algorithm:",
                        options=list(CLUSTERING_ALGORITHMS.keys()),
                        format_func=lambda k: f"{CLUSTERING_ALGORITHMS[k]['name']} ({CLUSTERING_ALGORITHMS[k]['family']})",
                        index=0,
                        key="ml_clustering_algo_select",
                    )
                with cl_col2:
                    if cl_algo in ("kmeans", "agglomerative"):
                        auto_k = st.checkbox("Auto-Discover k", value=True, key="ml_cluster_auto_k_check", help="Evaluate Silhouette across k=2..8 to find optimal partition.")
                        cl_k = st.slider("Target Clusters (k):", min_value=2, max_value=min(10, max(2, len(df) - 1)), value=3, key="ml_cluster_k_slider", disabled=auto_k)
                        cl_eps = 0.5
                        cl_min_s = 5
                    else:
                        auto_k = False
                        cl_k = 3
                        cl_eps = st.slider("DBSCAN Epsilon (eps):", min_value=0.1, max_value=3.0, value=0.5, step=0.1, key="ml_cluster_eps_slider")
                        cl_min_s = st.slider("Min Samples:", min_value=2, max_value=20, value=5, key="ml_cluster_mins_slider")

                with cl_col3:
                    st.caption(CLUSTERING_ALGORITHMS[cl_algo]["description"])

                with cl_col4:
                    st.write("")
                    st.write("")
                    run_cl_btn = st.button("🔮 Run Clustering", type="primary", width="stretch", key="ml_run_clustering_btn")

                if run_cl_btn:
                    with st.spinner("Partitioning feature space and calculating cluster metrics..."):
                        try:
                            cl_res = run_clustering(
                                df=df,
                                feature_cols=num_features,
                                algorithm=cl_algo,
                                n_clusters=cl_k,
                                eps=cl_eps,
                                min_samples=cl_min_s,
                                auto_k=auto_k,
                            )
                            st.session_state["ml_clustering_results"] = cl_res
                            st.success(cl_res["summary"])
                        except Exception as e:
                            st.error(f"Clustering execution failed: {str(e)}")

                cl_res = st.session_state.get("ml_clustering_results")
                if cl_res:
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    with cm1:
                        st.metric("Discovered Clusters", cl_res["n_clusters"])
                    with cm2:
                        st.metric("Silhouette Score", f"{cl_res['silhouette_score']:.4f}", help="Range -1 to 1. Higher indicates well-separated compact clusters.")
                    with cm3:
                        st.metric("Davies-Bouldin Index", f"{cl_res['davies_bouldin_score']:.4f}", help="Lower indicates better clustering separation.")
                    with cm4:
                        st.metric("Calinski-Harabasz", f"{cl_res['calinski_harabasz_score']:,.1f}", help="Higher indicates denser, better-separated clusters.")

                    if cl_res.get("k_evaluations"):
                        with st.expander("📊 Automated k Evaluation Scores (Elbow / Silhouette Search)", expanded=False):
                            st.dataframe(pd.DataFrame(cl_res["k_evaluations"]), width="stretch")

                    # 2D PCA Space Projection
                    if cl_res.get("pca_projection"):
                        st.markdown("#### 🗺️ Cluster Partition in 2D Latent Space")
                        cl_proj_df = pd.DataFrame(cl_res["pca_projection"])
                        fig_cl = px.scatter(
                            cl_proj_df,
                            x="pca_x",
                            y="pca_y",
                            color="cluster",
                            title=f"2D PCA Projection ({cl_res['algorithm_name']}) - {cl_res.get('variance_explained_pct', 0)}% Var Explained",
                            hover_data=["index"],
                            color_discrete_sequence=px.colors.qualitative.Bold,
                        )
                        fig_cl.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
                        st.plotly_chart(fig_cl, width="stretch")

                    # Cluster Profiles Summary
                    if cl_res.get("cluster_summary"):
                        st.markdown("#### 📋 Cluster Profiles (Feature Centroids & Record Counts)")
                        prof_df = pd.DataFrame(cl_res["cluster_summary"])
                        st.dataframe(prof_df, width="stretch")

                        # CSV Export with cluster labels appended
                        df_with_clusters = df.copy()
                        df_with_clusters["Cluster_Label"] = cl_res["labels"]
                        csv_cl = df_with_clusters.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Export Clustered Dataset (CSV)", data=csv_cl, file_name=f"clustered_data_{cl_algo}.csv", mime="text/csv")

        with t5_anomaly:
            if len(num_features) < 2:
                st.info("Anomaly detection requires at least 2 numeric feature columns.")
            else:
                an_col1, an_col2, an_col3 = st.columns([2, 1, 1])
                with an_col1:
                    an_algo = st.selectbox(
                        "Anomaly Algorithm:",
                        options=list(ANOMALY_ALGORITHMS.keys()),
                        format_func=lambda k: f"{ANOMALY_ALGORITHMS[k]['name']} ({ANOMALY_ALGORITHMS[k]['family']})",
                        index=0,
                        key="ml_anomaly_algo_select",
                    )
                with an_col2:
                    an_contam = st.slider(
                        "Expected Contamination:",
                        min_value=0.01,
                        max_value=0.20,
                        value=0.05,
                        step=0.01,
                        format="%.2f",
                        key="ml_anomaly_contam_slider",
                        help="Estimated proportion of outliers in the data.",
                    )
                with an_col3:
                    st.write("")
                    st.write("")
                    run_an_btn = st.button("🚨 Detect Anomalies", type="primary", width="stretch", key="ml_run_anomaly_btn")

                if run_an_btn:
                    with st.spinner("Computing robust multi-dimensional outlier distances..."):
                        try:
                            an_res = run_anomaly_detection(
                                df=df,
                                feature_cols=num_features,
                                algorithm=an_algo,
                                contamination=an_contam,
                            )
                            st.session_state["ml_anomaly_results"] = an_res
                            st.success(an_res["summary"])
                        except Exception as e:
                            st.error(f"Anomaly detection failed: {str(e)}")

                an_res = st.session_state.get("ml_anomaly_results")
                if an_res:
                    an_m1, an_m2, an_m3, an_m4 = st.columns(4)
                    with an_m1:
                        st.metric("Total Records", f"{an_res['total_records']:,}")
                    with an_m2:
                        st.metric("Detected Anomalies", f"{an_res['anomaly_count']:,}", delta=f"-{an_res['anomaly_rate_pct']}%", delta_color="inverse")
                    with an_m3:
                        st.metric("Nominal Inliers", f"{an_res['inlier_count']:,}")
                    with an_m4:
                        st.metric("Algorithm", an_res["algorithm_name"][:16])

                    # 2D PCA Outlier Projection Scatter
                    if an_res.get("pca_projection"):
                        st.markdown("#### 🗺️ 2D Outlier Spatial Distribution")
                        an_proj_df = pd.DataFrame(an_res["pca_projection"])
                        fig_an = px.scatter(
                            an_proj_df,
                            x="pca_dim1",
                            y="pca_dim2",
                            color="status",
                            title=f"Outlier Boundary Projection ({an_res['algorithm_name']})",
                            color_discrete_map={"Anomaly": "#EF4444", "Inlier": "#00F0FF"},
                            hover_data=["index", "score"],
                        )
                        fig_an.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
                        st.plotly_chart(fig_an, width="stretch")

                    # Top Anomalous Records Table
                    if an_res.get("top_anomalies"):
                        st.markdown("#### 🚨 Top Ranked Outlier Records")
                        top_an_df = pd.DataFrame(an_res["top_anomalies"])
                        st.dataframe(top_an_df, width="stretch")
                        csv_an = top_an_df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Export Anomalies (CSV)", data=csv_an, file_name="detected_anomalies.csv", mime="text/csv")

    # =========================================================================
    # TAB 6: MODELS & CONFIG
    # =========================================================================
    with tabs[5]:
        st.subheader("⚙️ Algorithm Selection & Training Configuration")
        task_curr = st.session_state.get("ml_task_type", "auto")

        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            comp_prof = st.select_slider(
                "Compute Profile Mode:",
                options=["Fast", "Balanced", "Deep"],
                value=st.session_state.get("ml_compute_profile", "Balanced"),
                help="Fast: Quick baselines. Balanced: Ensembles. Deep: Gradient boosting & parameter tuning.",
            )
            st.session_state["ml_compute_profile"] = comp_prof

            val_split = st.slider(
                "Holdout Validation Split:",
                min_value=0.10,
                max_value=0.40,
                value=float(st.session_state.get("ml_test_size", 0.20)),
                step=0.05,
            )
            st.session_state["ml_test_size"] = val_split

        with cfg_col2:
            scaler_choice = st.selectbox(
                "Numeric Feature Scaling:",
                options=["standard", "robust", "minmax", "none"],
                index=0,
            )
            st.session_state["ml_scaler"] = scaler_choice

            cat_choice = st.selectbox(
                "Categorical Encoding:",
                options=["onehot", "ordinal"],
                index=0,
            )
            st.session_state["ml_cat_encoding"] = cat_choice

        st.markdown("---")
        st.markdown("#### 📚 Algorithm Library Catalog")
        available_models = get_candidate_models(task_curr, comp_prof)
        sel_models = st.multiselect(
            "Candidate Estimators to Benchmark:",
            options=available_models,
            default=available_models,
            format_func=lambda mid: f"{REGISTRY[mid].name} ({REGISTRY[mid].family.title()}, {REGISTRY[mid].speed})" if mid in REGISTRY else mid,
        )
        st.session_state["ml_selected_models"] = sel_models

    # =========================================================================
    # TAB 7: TRAINING & AUTOML
    # =========================================================================
    with tabs[6]:
        st.subheader("🏎️ Automated Model Training & Benchmark Execution")
        target_col = st.session_state.get("ml_target_col")
        feature_cols = st.session_state.get("ml_selected_features", [])

        if not target_col:
            st.warning("Please configure the target column in the **Target** tab first.")
        elif not feature_cols:
            st.warning("Please select at least one feature column in the **Features & Selection** tab.")
        else:
            st.info(f"Ready to benchmark **{len(st.session_state.get('ml_selected_models', []))} models** predicting `{target_col}` using **{len(feature_cols)} features**.")
            launch_train = st.button("🚀 Launch Automated Benchmark", type="primary", width="stretch")

            if launch_train:
                prog_bar = st.progress(0, text="Initializing scikit-learn preprocessing pipelines...")
                try:
                    time.sleep(0.3)
                    prog_bar.progress(30, text="Training candidate estimators across holdout validation...")
                    bench_res = run_benchmark(
                        df=df,
                        target_col=target_col,
                        feature_cols=feature_cols,
                        task_type=st.session_state.get("ml_task_type", "auto"),
                        compute_profile=st.session_state.get("ml_compute_profile", "Balanced"),
                        test_size=float(st.session_state.get("ml_test_size", 0.20)),
                        cv_folds=int(st.session_state.get("ml_cv_folds", 3)),
                        scaler_type=st.session_state.get("ml_scaler", "standard"),
                        cat_encoding=st.session_state.get("ml_cat_encoding", "onehot"),
                        model_subset=st.session_state.get("ml_selected_models"),
                        dataset_name=proj_name,
                    )
                    prog_bar.progress(85, text="Registering model artifacts and computing diagnostics...")
                    st.session_state["ml_benchmark_results"] = bench_res
                    st.session_state["ml_active_model_id"] = bench_res["champion_id"]
                    prog_bar.progress(100, text="Benchmark completed successfully!")
                    st.success(f"🏆 Champion Model: **{bench_res['champion_name']}** ({bench_res['champion_reason']})")
                except Exception as e:
                    st.error(f"Benchmark failed: {str(e)}")

        bench_res = st.session_state.get("ml_benchmark_results")
        if bench_res:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 14px; margin: 16px 0;">
                <div style="color: #10B981; font-weight: 800; font-size: 0.8rem; text-transform: uppercase;">🏆 Production Champion Selected</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{bench_res.get('champion_name', 'Evaluated Champion')}</div>
                <div style="font-size: 0.85rem; color: #94A3B8;">{bench_res.get('champion_reason', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Benchmark Leaderboard & Comparative Ranking")
            lb_df = pd.DataFrame(bench_res.get("leaderboard", []))
            if not lb_df.empty:
                fig_lb = px.bar(
                    lb_df,
                    x="score",
                    y="model_name",
                    orientation="h",
                    color="is_champion",
                    color_discrete_map={True: "#00F0FF", False: "#64748B"},
                    title=f"Comparative Benchmark Performance ({bench_res.get('primary_metric', 'Score').upper()})",
                    labels={"score": bench_res.get("primary_metric", "Score").upper(), "model_name": "Model", "is_champion": "Champion"},
                )
                fig_lb.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    font_color="#E2E8F0",
                )
                st.plotly_chart(fig_lb, width="stretch")

                display_cols = ["rank", "model_name", "score", "training_time", "is_overfitting"]
                available_disp = [c for c in display_cols if c in lb_df.columns]
                st.dataframe(lb_df[available_disp].rename(columns={
                    "rank": "Rank",
                    "model_name": "Model",
                    "score": f"Validation {bench_res.get('primary_metric', 'Score').upper()}",
                    "training_time": "Training Latency (s)",
                    "is_overfitting": "Overfitting Risk",
                }), width="stretch")

    # =========================================================================
    # TAB 8: EVALUATION & DIAGNOSTICS
    # =========================================================================
    with tabs[7]:
        st.subheader("🔍 Deep Diagnostic & Holdout Validation Analysis")
        bench_res = st.session_state.get("ml_benchmark_results")

        if not bench_res:
            st.info("Execute model training in the **Training & AutoML** tab to view diagnostics.")
        else:
            models_dict = bench_res.get("models") or bench_res.get("all_results") or {}
            if not models_dict:
                st.warning("No model evaluation diagnostics available. Please re-run training.")
            else:
                default_idx = 0
                keys_list = list(models_dict.keys())
                champ_id = bench_res.get("champion_id")
                if champ_id in keys_list:
                    default_idx = keys_list.index(champ_id)
                active_m_id = st.selectbox(
                    "Inspect Specific Model Diagnostics:",
                    options=keys_list,
                    format_func=lambda mid: models_dict[mid].get("model_name", mid),
                    index=default_idx,
                )
                m_data = models_dict.get(active_m_id, {})
                metrics = m_data.get("metrics", {})

            # Metric cards
            met_cols = st.columns(4)
            for i, (k, v) in enumerate(list(metrics.items())[:4]):
                with met_cols[i % 4]:
                    st.metric(k.upper(), f"{v:.4f}" if isinstance(v, (int, float)) else str(v))

            if m_data.get("warnings"):
                st.warning("\n".join([f"⚠️ {w}" for w in m_data["warnings"]]))

            # Scatter sample for regression or confusion matrix for classification
            if "scatter_sample" in m_data and m_data["scatter_sample"]:
                st.markdown("#### 🎯 Actual vs Predicted Residual Diagnostic")
                sc_df = pd.DataFrame(m_data["scatter_sample"])
                fig_act = px.scatter(
                    sc_df,
                    x="actual",
                    y="predicted",
                    title=f"Actual vs Predicted: {m_data.get('model_name')}",
                    color="residual",
                    color_continuous_scale="Viridis",
                )
                min_v = min(sc_df["actual"].min(), sc_df["predicted"].min())
                max_v = max(sc_df["actual"].max(), sc_df["predicted"].max())
                fig_act.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(color="#EF4444", dash="dash"), name="Ideal Fit (y = x)"))
                fig_act.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
                st.plotly_chart(fig_act, width="stretch")

            elif "confusion_matrix" in m_data and m_data["confusion_matrix"]:
                st.markdown("#### 🧩 Confusion Matrix")
                cm = np.array(m_data["confusion_matrix"])
                classes = m_data.get("classes", [f"Class {i}" for i in range(len(cm))])
                fig_cm = px.imshow(
                    cm,
                    x=classes,
                    y=classes,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Class", y="Actual Class"),
                )
                fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
                st.plotly_chart(fig_cm, width="stretch")

    # =========================================================================
    # TAB 9: EXPLAINABLE AI
    # =========================================================================
    with tabs[8]:
        st.subheader("💡 Explainable AI (XAI) & Attribution")
        bench_res = st.session_state.get("ml_benchmark_results")

        if not bench_res:
            st.info("Execute training to inspect global feature importance and local SHAP explanations.")
        else:
            models_dict = bench_res.get("models") or bench_res.get("all_results") or {}
            active_m_id = bench_res.get("champion_id") or (list(models_dict.keys())[0] if models_dict else None)
            m_data = models_dict.get(active_m_id, {})
            feat_imp = m_data.get("feature_importance", [])

            st.markdown("#### 🌐 Global Feature Importance (Tree MDI & Permutation)")
            if feat_imp:
                imp_df = pd.DataFrame(feat_imp)
                fig_fi = px.bar(
                    imp_df.head(12),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"Top Predictive Drivers: {m_data.get('model_name')}",
                    color="importance",
                    color_continuous_scale="Viridis",
                )
                fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
                st.plotly_chart(fig_fi, width="stretch")

            st.markdown("#### 🔍 Single Prediction Local Attribution Explainer")
            row_idx = st.slider("Select Record Index to Explain:", min_value=0, max_value=min(200, len(df) - 1), value=0)
            sample_row = df.iloc[row_idx].to_dict()
            try:
                pipe = m_data["pipeline"]
                local_exp = explain_local_prediction(pipe, sample_row, bench_res.get("task_type", "regression"))
                attr_df = pd.DataFrame(local_exp.get("attributions", []))
                if not attr_df.empty:
                    fig_loc = px.bar(
                        attr_df.head(10),
                        x="attribution",
                        y="feature",
                        orientation="h",
                        title=f"Local Attribution for Record #{row_idx} (Predicted: {local_exp.get('prediction')})",
                        color="attribution",
                        color_continuous_scale="RdBu",
                    )
                    fig_loc.update_layout(yaxis={"categoryorder": "total ascending"}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
                    st.plotly_chart(fig_loc, width="stretch")
            except Exception as e:
                st.caption(f"Local attribution unavailable: {str(e)}")

            st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.08); border-left: 3px solid #F59E0B; padding: 12px; border-radius: 4px; font-size: 0.82rem; color: #FCD34D;">
                <b>STATISTICAL ASSOCIATION NOTICE:</b> {STATISTICAL_DISCLAIMER}
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 10: PREDICTIONS & INFERENCE
    # =========================================================================
    with tabs[9]:
        st.subheader("🔮 Predictive Inference Cockpit")
        bench_res = st.session_state.get("ml_benchmark_results")

        if not bench_res:
            st.info("Train a model first to run single-record simulations or batch CSV/XLSX predictions.")
        else:
            models_dict = bench_res.get("models") or bench_res.get("all_results") or {}
            active_m_id = bench_res.get("champion_id") or (list(models_dict.keys())[0] if models_dict else None)
            m_data = models_dict.get(active_m_id, {})
            pipe = m_data.get("pipeline")
            feat_cols = bench_res.get("feature_cols") or m_data.get("feature_cols") or [c for c in df.columns if c != st.session_state.get("ml_target_col")]
            task_t = bench_res.get("task_type") or m_data.get("task_type", "regression")

            p_sub1, p_sub2 = st.tabs(["Interactive Single Record Simulator", "Batch File Prediction Engine"])

            # Sub-Tab 1: Single Record Simulator
            with p_sub1:
                st.markdown("##### Adjust Feature Inputs to Simulate Output")
                input_dict = {}
                c_inputs = st.columns(3)
                for i, f_name in enumerate(feat_cols[:12]):
                    with c_inputs[i % 3]:
                        if f_name in df.columns:
                            col_s = df[f_name]
                            if pd.api.types.is_numeric_dtype(col_s):
                                mn = float(col_s.min()) if not pd.isna(col_s.min()) else 0.0
                                mx = float(col_s.max()) if not pd.isna(col_s.max()) else 100.0
                                md = float(col_s.median()) if not pd.isna(col_s.median()) else 50.0
                                if mn == mx:
                                    mx = mn + 1.0
                                input_dict[f_name] = st.slider(f"{f_name}:", min_value=mn, max_value=mx, value=md)
                            else:
                                unq = [str(u) for u in col_s.dropna().unique()[:20]]
                                input_dict[f_name] = st.selectbox(f"{f_name}:", options=unq or ["Default"])
                        else:
                            input_dict[f_name] = st.text_input(f"{f_name}:", value="0.0")

                if st.button("🔮 Compute Prediction", type="primary"):
                    p_res = predict_single_record(pipe, input_dict, task_type=task_t)
                    if "error" in p_res:
                        st.error(p_res["error"])
                    else:
                        pred_v = p_res.get("prediction")
                        st.markdown(f"""
                        <div style="background: rgba(0, 240, 255, 0.08); border: 1px solid #00F0FF; border-radius: 10px; padding: 18px; text-align: center; margin-top: 14px;">
                            <div style="font-size: 0.8rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">Simulated Model Output</div>
                            <div style="font-size: 2.4rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{pred_v}</div>
                            <div style="font-size: 0.8rem; color: #94A3B8;">Confidence / Latency: {p_res.get('latency_ms')} ms</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Sub-Tab 2: Batch Inference
            with p_sub2:
                st.markdown("##### Upload CSV/XLSX for Batch Predictions")
                batch_file = st.file_uploader("Upload Data File:", type=["csv", "xlsx"])
                if batch_file:
                    try:
                        batch_df = pd.read_csv(batch_file) if batch_file.name.endswith(".csv") else pd.read_excel(batch_file)
                        st.write(f"Loaded {len(batch_df):,} rows.")
                        if st.button("🚀 Run Batch Scoring"):
                            scored_df = predict_batch(pipe, batch_df, task_type=task_t)
                            st.dataframe(scored_df.head(10), width="stretch")
                            csv_data = scored_df.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download Scored CSV", data=csv_data, file_name="scored_predictions.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Batch inference failed: {str(e)}")

    # =========================================================================
    # TAB 11: FUTURE FORECASTING
    # =========================================================================
    with tabs[10]:
        st.subheader("📈 Institutional Time Series Forecasting Suite")
        st.caption("Multi-model AutoML forecasting with statistical diagnostics, trend, seasonality, and confidence intervals.")

        date_candidates = df.select_dtypes(include=["datetime", "object"]).columns.tolist()
        num_candidates = df.select_dtypes(include=[np.number]).columns.tolist()

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            date_col = st.selectbox("Date / Timestamp Column:", options=date_candidates or list(df.columns), index=0)
        with f_col2:
            target_metric = st.selectbox("Forecast Metric (y):", options=num_candidates or list(df.columns), index=0)
        with f_col3:
            horizon_steps = st.slider("Forecast Horizon (periods):", min_value=3, max_value=60, value=14)

        f_opt1, f_opt2 = st.columns(2)
        with f_opt1:
            method_fc = st.selectbox(
                "Forecasting Algorithm:",
                options=["auto", "exponential_smoothing", "holt_winters", "arima", "sarimax", "ml_lag_rf", "naive"],
                format_func=lambda x: f"AutoML Competition ({x})" if x == "auto" else x.replace("_", " ").title(),
                index=0,
            )
        with f_opt2:
            ci_level = st.selectbox("Confidence Interval:", options=[0.90, 0.95], format_func=lambda x: f"{int(x*100)}% Confidence Band")

        if st.button("🚀 Generate Future Forecast", type="primary", width="stretch"):
            with st.spinner("Analyzing time-series diagnostics and training candidate models..."):
                try:
                    fc_res = run_time_series_forecast(
                        df=df,
                        date_col=date_col,
                        target_col=target_metric,
                        horizon=horizon_steps,
                        method=method_fc,
                        confidence_level=ci_level,
                    )
                    st.session_state["ml_forecast_results"] = fc_res
                    st.success(fc_res["trend_summary"])
                except Exception as e:
                    st.error(f"Forecasting error: {str(e)}")

        fc_res = st.session_state.get("ml_forecast_results")
        if fc_res:
            diag = fc_res.get("diagnostics", {})
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("Detected Trend", diag.get("trend", "N/A"))
            with d2:
                st.metric("Seasonality", diag.get("seasonality", "N/A")[:18])
            with d3:
                st.metric("Projected Growth", f"{fc_res.get('metrics', {}).get('projected_growth_pct', 0.0):+.1f}%")
            with d4:
                st.metric("Champion Model", fc_res.get("best_model"))

            # Interactive Plotly actual vs forecast chart with confidence band
            hist_df = pd.DataFrame(fc_res["historical"])
            pred_df = pd.DataFrame(fc_res["forecast"])

            fig_fc = go.Figure()
            # Historical actuals
            fig_fc.add_trace(go.Scatter(
                x=hist_df["date"],
                y=hist_df["actual"],
                name="Historical Actual",
                line=dict(color="#38BDF8", width=2),
            ))
            # Upper CI
            fig_fc.add_trace(go.Scatter(
                x=pred_df["date"],
                y=pred_df["upper_ci"],
                name="Upper CI",
                line=dict(width=0),
                showlegend=False,
            ))
            # Lower CI with fill
            fig_fc.add_trace(go.Scatter(
                x=pred_df["date"],
                y=pred_df["lower_ci"],
                name="Confidence Interval",
                fill="tonexty",
                fillcolor="rgba(0, 240, 255, 0.15)",
                line=dict(width=0),
            ))
            # Future Forecast
            fig_fc.add_trace(go.Scatter(
                x=pred_df["date"],
                y=pred_df["predicted"],
                name="Projected Forecast",
                line=dict(color="#00F0FF", width=3, dash="dash"),
            ))

            fig_fc.update_layout(
                title=f"Future Forecast Trajectory: {target_metric} ({fc_res['best_model']})",
                xaxis_title="Timeline",
                yaxis_title=target_metric,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font_color="#E2E8F0",
            )
            st.plotly_chart(fig_fc, width="stretch")

            # Model Competition Leaderboard
            if fc_res.get("leaderboard"):
                st.markdown("#### 🏆 Forecasting Model Benchmark Comparison")
                st.dataframe(pd.DataFrame(fc_res["leaderboard"]), width="stretch")

    # =========================================================================
    # TAB 12: SCENARIO ANALYSIS (WHAT-IF)
    # =========================================================================
    with tabs[11]:
        st.subheader("🔀 What-If Counterfactual Scenario Simulator")
        st.caption("Perturb input variables to analyze marginal impacts, driver sensitivities, and counterfactual predictions.")

        bench_res = st.session_state.get("ml_benchmark_results")
        if not bench_res:
            st.info("Train a model in the **Training & AutoML** tab first to run What-If simulations.")
        else:
            models_dict = bench_res.get("models") or bench_res.get("all_results") or {}
            active_m_id = bench_res.get("champion_id") or (list(models_dict.keys())[0] if models_dict else None)
            m_data = models_dict.get(active_m_id, {})
            pipe = m_data.get("pipeline")
            feat_cols = bench_res.get("feature_cols") or m_data.get("feature_cols") or [c for c in df.columns if c != st.session_state.get("ml_target_col")]
            task_t = bench_res.get("task_type") or m_data.get("task_type", "regression")
            valid_feats = [c for c in feat_cols if c in df.columns]

            # Baseline record selection
            b_row_idx = st.slider("Select Baseline Record:", min_value=0, max_value=min(100, len(df) - 1), value=0)
            base_record = df[valid_feats].iloc[b_row_idx].to_dict()

            st.markdown("##### Counterfactual Feature Perturbations")
            modifications = {}
            p_cols = st.columns(3)
            for i, f_name in enumerate(feat_cols[:9]):
                with p_cols[i % 3]:
                    orig_val = base_record.get(f_name)
                    if isinstance(orig_val, (int, float, np.number)):
                        mn = float(orig_val * 0.5)
                        mx = float(orig_val * 1.5) if orig_val != 0 else 10.0
                        if mn > mx:
                            mn, mx = mx, mn
                        new_val = st.slider(f"{f_name}:", min_value=mn, max_value=mx, value=float(orig_val), key=f"whatif_{f_name}")
                        if abs(new_val - float(orig_val)) > 1e-4:
                            modifications[f_name] = new_val

            if st.button("⚡ Run Counterfactual Simulation", type="primary"):
                scen_res = simulate_what_if_scenario(
                    pipeline=pipe,
                    base_record=base_record,
                    modifications=modifications,
                    task_type=task_t,
                )
                st.session_state["ml_scenario_results"] = scen_res

            scen_res = st.session_state.get("ml_scenario_results")
            if scen_res and scen_res.get("success"):
                comp = scen_res["comparison"]
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Baseline Output", f"{comp.get('base_prediction')}")
                with c2:
                    st.metric("Simulated Scenario", f"{comp.get('scenario_prediction')}")
                with c3:
                    if comp.get("delta") is not None:
                        st.metric("Output Delta (Δy)", f"{comp.get('delta'):+.2f}", f"{comp.get('pct_change'):+.1f}%")
                    else:
                        st.metric("Class Shift", f"{'Changed' if comp.get('class_changed') else 'Unchanged'}")

                st.info(comp["summary"])

                if scen_res.get("feature_impacts"):
                    st.markdown("##### 🔍 Feature Elasticity & Sensitivity Ranking")
                    st.dataframe(pd.DataFrame(scen_res["feature_impacts"]), width="stretch")

    # =========================================================================
    # TAB 13: EXPERIMENTS & REGISTRY
    # =========================================================================
    with tabs[12]:
        st.subheader("📦 Model Registry, Lineage & Institutional Reports")
        champ = get_active_champion()

        if champ:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1E293B; border-radius: 10px; padding: 16px; margin-bottom: 16px;">
                <div style="color: #38BDF8; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">Active Production Champion</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{champ.get('model_name')}</div>
                <div style="font-size: 0.85rem; color: #94A3B8;">Target: <code>{champ.get('target_col')}</code> | Experiment ID: <code>{champ.get('experiment_id')}</code></div>
            </div>
            """, unsafe_allow_html=True)

        # Historical Experiments Table
        exps = list_experiments()
        if exps:
            st.markdown("#### 📜 Experiment Version History")
            st.dataframe(pd.DataFrame(exps), width="stretch")

        st.markdown("---")
        st.markdown("#### 📄 Institutional ML Report Export")
        st.caption("Compile a comprehensive audit report in PDF, HTML, or CSV formats.")

        rep_target = st.session_state.get("ml_target_col", "Target")
        r_col1, r_col2, r_col3 = st.columns(3)

        bench_data = st.session_state.get("ml_benchmark_results") or {}
        champ_models = bench_data.get("models") or {}
        champ_exp_id = champ.get("experiment_id") if champ else ""
        champ_feat_imp = (champ_models.get(champ_exp_id) or {}).get("feature_importance", [])

        report_payload = {
            "dataset_name": proj_name,
            "target_col": rep_target,
            "task_type": st.session_state.get("ml_task_type", "regression"),
            "readiness_score": 85,
            "total_rows": len(df),
            "feature_count": len(df.columns) - 1,
            "champion_name": champ.get("model_name") if champ else "Evaluated Model",
            "leaderboard": bench_data.get("leaderboard", []),
            "feature_importance": champ_feat_imp,
            "forecast_results": st.session_state.get("ml_forecast_results"),
        }

        with r_col1:
            try:
                pdf_bytes = generate_ml_pdf_report(report_payload)
                st.download_button(
                    "📥 Export Audit Report (PDF)",
                    data=pdf_bytes,
                    file_name="Quantum_Analyst_ML_Audit_Report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            except Exception as e:
                st.caption(f"PDF unavailable: {e}")

        with r_col2:
            try:
                html_code = generate_ml_html_report(report_payload)
                st.download_button(
                    "🌐 Export Standalone HTML",
                    data=html_code.encode("utf-8"),
                    file_name="Quantum_Analyst_ML_Report.html",
                    mime="text/html",
                    width="stretch",
                )
            except Exception as e:
                st.caption(f"HTML export unavailable: {e}")

        with r_col3:
            try:
                csv_sum = generate_ml_csv_summary(report_payload)
                st.download_button(
                    "📊 Export Metrics (CSV)",
                    data=csv_sum.encode("utf-8"),
                    file_name="ml_model_metrics.csv",
                    mime="text/csv",
                    width="stretch",
                )
            except Exception as e:
                st.caption(f"CSV export unavailable: {e}")
