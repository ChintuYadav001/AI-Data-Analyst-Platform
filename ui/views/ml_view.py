"""
ui.views.ml_view
================
Institutional-grade Machine Learning and Predictive Analytics Cockpit for Quantum Analyst.
Provides a 9-tab interactive suite:
1. Overview
2. ML Readiness & Target
3. Model Selection & Config
4. Training & Benchmarking
5. Evaluation & Diagnostics
6. Explainable AI
7. Predictions & Inference
8. Time-Series Forecasting
9. Experiments & Registry
"""

import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.ml.ml_readiness import audit_ml_readiness, classify_task_type
from core.ml.model_registry import REGISTRY, get_candidate_models
from core.ml.model_selection import run_benchmark
from core.ml.trainer import train_single_model
from core.ml.tuning import tune_model_hyperparameters
from core.ml.forecasting import run_time_series_forecast
from core.ml.prediction import predict_single_record, predict_batch
from core.ml.explainability import explain_local_prediction, STATISTICAL_DISCLAIMER
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
        "ml_last_dataset_signature": None,
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
            if st.button("📂 Go to Upload Dataset", type="primary", use_container_width=True):
                st.session_state["active_nav"] = "Upload Dataset"
                st.rerun()
        return

    df: pd.DataFrame = st.session_state["df"]
    proj_name = st.session_state.get("current_project_name", "Active Dataset")

    # Header section
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px;">
        <div>
            <div style="font-size: 0.76rem; font-weight: 800; color: #00F0FF; letter-spacing: 0.08em; text-transform: uppercase;">
                // AUTOMATED MACHINE LEARNING & PREDICTIVE ANALYTICS
            </div>
            <h2 style="margin: 2px 0 6px 0; color: #FFFFFF; font-size: 1.7rem; font-weight: 800;">
                🤖 Quantum Predictive Engine
            </h2>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                End-to-end model benchmarking, reproducible pipelines, explainable AI, time series, and deployment.
            </div>
        </div>
        <div style="text-align: right; background: rgba(0, 240, 255, 0.06); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 6px 14px;">
            <div style="font-size: 0.68rem; font-weight: 700; color: #38BDF8;">WORKSPACE DATASET</div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">{proj_name}</div>
            <div style="font-size: 0.74rem; color: #94A3B8;">{len(df):,} rows × {len(df.columns)} cols</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Detect dataset change and synchronize session state
    curr_signature = f"{proj_name}_{len(df)}_{tuple(df.columns)}"
    if st.session_state.get("ml_last_dataset_signature") != curr_signature:
        st.session_state["ml_last_dataset_signature"] = curr_signature
        st.session_state["ml_target_col"] = None
        st.session_state["ml_selected_features"] = []
        st.session_state["ml_benchmark_results"] = None
        st.session_state["ml_active_model_id"] = None
        st.session_state["ml_forecast_results"] = None

    # ML Readiness Audit
    readiness_report = audit_ml_readiness(df, target_col=st.session_state.get("ml_target_col"))

    # If target not set or no longer exists in current df, auto-select
    if st.session_state.get("ml_target_col") not in df.columns:
        if readiness_report.selected_target and readiness_report.selected_target in df.columns:
            st.session_state["ml_target_col"] = readiness_report.selected_target
        elif len(df.columns) > 0:
            st.session_state["ml_target_col"] = df.columns[-1]

    current_target = st.session_state.get("ml_target_col")

    # Filter and validate active feature cohort
    valid_features = [c for c in st.session_state.get("ml_selected_features", []) if c in df.columns and c != current_target]
    if not valid_features:
        exclude = set(readiness_report.id_columns + readiness_report.constant_columns)
        if current_target:
            exclude.add(current_target)
        valid_features = [c for c in df.columns if c not in exclude]
    st.session_state["ml_selected_features"] = valid_features

    # Update Task Type
    if current_target and current_target in df.columns:
        st.session_state["ml_task_type"] = classify_task_type(df, current_target)

    # 9 Tabs Layout
    tabs = st.tabs([
        "🔭 Overview",
        "🎯 ML Readiness & Target",
        "🤖 Model Selection & Config",
        "⚡ Training & Benchmarking",
        "📊 Evaluation & Diagnostics",
        "🔍 Explainable AI",
        "🔮 Predictions & Inference",
        "📈 Time-Series Forecasting",
        "📜 Experiments & Registry",
    ])

    # -------------------------------------------------------------
    # TAB 1: OVERVIEW
    # -------------------------------------------------------------
    with tabs[0]:
        st.markdown("### 🔭 Predictive Architecture Overview")
        st.markdown("Automated intelligence pipeline assessing data health, selecting optimal algorithmic candidates, training with cross-validation, and verifying generalization stability.")

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            score = readiness_report.overall_score
            score_color = "#10B981" if score >= 80 else ("#F59E0B" if score >= 60 else "#EF4444")
            st.markdown(f"""
            <div style="background: rgba(14, 20, 36, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 9px; padding: 14px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: #94A3B8;">ML READINESS SCORE</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {score_color}; margin: 4px 0;">{score}/100</div>
                <div style="font-size: 0.74rem; color: #64748B;">{'Optimal Health' if score >= 80 else 'Action Recommended'}</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            task_label = st.session_state["ml_task_type"].replace("_", " ").title()
            st.markdown(f"""
            <div style="background: rgba(14, 20, 36, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 9px; padding: 14px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: #94A3B8;">DETECTED TASK</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #00F0FF; margin: 8px 0;">{task_label}</div>
                <div style="font-size: 0.74rem; color: #64748B;">Target: <b>{current_target or 'None'}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            num_feats = len(st.session_state["ml_selected_features"])
            st.markdown(f"""
            <div style="background: rgba(14, 20, 36, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 9px; padding: 14px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: #94A3B8;">FEATURE COHORT</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #A855F7; margin: 4px 0;">{num_feats} Features</div>
                <div style="font-size: 0.74rem; color: #64748B;">Excluded: {len(df.columns) - num_feats - (1 if current_target else 0)} non-predictive</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            champ = get_active_champion()
            champ_name = champ.get("model_name", "None registered") if champ else "None registered"
            st.markdown(f"""
            <div style="background: rgba(14, 20, 36, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 9px; padding: 14px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: #94A3B8;">CHAMPION MODEL</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #F59E0B; margin: 9px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">🏆 {champ_name}</div>
                <div style="font-size: 0.74rem; color: #64748B;">Production Ready</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Hero workflow launch card
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 240, 255, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0 0 6px 0; color: #FFFFFF;">🚀 Quick Automated Pipeline Run</h3>
                    <div style="color: #94A3B8; font-size: 0.88rem; max-width: 650px;">
                        Benchmark candidate algorithms across Fast, Balanced, and Deep ensembles. Quantum Analyst trains cross-validated pipelines, evaluates metrics, and automatically promotes the winning Champion model.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c_go1, c_go2 = st.columns([1, 3])
        with c_go1:
            if st.button("⚡ Run Full Automated Benchmark", type="primary", use_container_width=True):
                target_to_use = st.session_state.get("ml_target_col")
                if not target_to_use or target_to_use not in df.columns:
                    st.error("Please select a valid target column in the ML Readiness tab first.")
                else:
                    feats_to_use = [c for c in st.session_state.get("ml_selected_features", []) if c in df.columns and c != target_to_use]
                    if not feats_to_use:
                        st.error("Please select at least one valid feature column.")
                    else:
                        with st.spinner("Executing automated cross-validation benchmarking..."):
                            try:
                                res = run_benchmark(
                                    df=df,
                                    target_col=target_to_use,
                                    feature_cols=feats_to_use,
                                    task_type=st.session_state["ml_task_type"],
                                    compute_profile=st.session_state["ml_compute_profile"],
                                )
                                st.session_state["ml_benchmark_results"] = res
                                st.session_state["ml_active_model_id"] = res["champion_id"]
                                
                                # Auto-save champion to experiment tracker
                                champ_id = res["champion_id"]
                                if champ_id and champ_id in res["all_results"]:
                                    champ_data = res["all_results"][champ_id]
                                    save_experiment(
                                        pipeline=champ_data["pipeline"],
                                        model_id=champ_id,
                                        model_name=champ_data["model_name"],
                                        task_type=st.session_state["ml_task_type"],
                                        target_col=target_to_use,
                                        feature_cols=feats_to_use,
                                        metrics=champ_data["metrics"],
                                        dataset_name=proj_name,
                                        is_champion=True,
                                    )
                                st.success(f"Benchmark finished! Champion Model: **{res['champion_id']}**")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Training failed: {str(e)}")

        with c_go2:
            st.info("💡 To fine-tune features, validation ratios, or algorithms, explore the **Target**, **Model Selection**, and **Diagnostics** tabs above.")

    # -------------------------------------------------------------
    # TAB 2: ML READINESS & TARGET
    # -------------------------------------------------------------
    with tabs[1]:
        st.markdown("### 🎯 Dataset Suitability & Target Auto-Detection")
        st.markdown("Examines semantic column names, cardinality, zero-variance attributes, and look-ahead target leakage.")

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("#### 1. Target Column Selection")
            
            if readiness_report.target_candidates:
                st.markdown("<div style='font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px;'>AI Target Candidates (Ranked):</div>", unsafe_allow_html=True)
                for cand in readiness_report.target_candidates[:3]:
                    badge_col = "#10B981" if cand.confidence >= 0.85 else "#38BDF8"
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 7px; padding: 7px 10px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 700; color: #FFFFFF;">`{cand.column}`</span>
                            <span style="color: #64748B; font-size: 0.74rem; margin-left: 8px;">{cand.reason}</span>
                        </div>
                        <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid {badge_col}; border-radius: 5px; color: {badge_col}; font-size: 0.72rem; font-weight: 700; padding: 2px 6px;">
                            {int(cand.confidence * 100)}% Confidence
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            target_options = list(df.columns)
            target_default_idx = target_options.index(current_target) if current_target in target_options else 0
            
            chosen_target = st.selectbox(
                "Select Target Variable ($y$):",
                options=target_options,
                index=target_default_idx,
                help="The column your machine learning model will learn to predict.",
            )

            if chosen_target != st.session_state.get("ml_target_col"):
                st.session_state["ml_target_col"] = chosen_target
                st.session_state["ml_task_type"] = classify_task_type(df, chosen_target)
                st.rerun()

            st.markdown(f"**Formally Classified Task**: `{st.session_state['ml_task_type'].upper()}`")

        with col_t2:
            st.markdown("#### 2. Feature Selection Cohort ($X$)")
            available_features = [c for c in df.columns if c != chosen_target]
            
            b_sel1, b_sel2 = st.columns(2)
            with b_sel1:
                if st.button("Select All Features", use_container_width=True):
                    st.session_state["ml_selected_features"] = available_features
                    st.rerun()
            with b_sel2:
                if st.button("Deselect IDs & Constants", use_container_width=True):
                    bad_cols = set(readiness_report.id_columns + readiness_report.constant_columns)
                    st.session_state["ml_selected_features"] = [c for c in available_features if c not in bad_cols]
                    st.rerun()

            selected_features = st.multiselect(
                "Active Features for Model Training:",
                options=available_features,
                default=[c for c in st.session_state["ml_selected_features"] if c in available_features],
                help="Variables used to train the estimator.",
            )
            st.session_state["ml_selected_features"] = selected_features

        # Leakage Warnings Card
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 3. Target Leakage & Health Audit")
        
        leak_reports = readiness_report.leakage_warnings
        if leak_reports:
            for leak in leak_reports:
                st.warning(f"⚠️ **Target Leakage Alert ({leak['risk_level']})**: {leak['message']}")
        else:
            st.success("✅ No look-ahead target leakage detected (|r| < 0.98 across all feature columns).")

        if readiness_report.warnings:
            with st.expander("⚠️ Data Health Advisory Notes", expanded=False):
                for w in readiness_report.warnings:
                    st.write(f"- {w}")

    # -------------------------------------------------------------
    # TAB 3: MODEL SELECTION & CONFIG
    # -------------------------------------------------------------
    with tabs[2]:
        st.markdown("### 🤖 Algorithmic Architecture & Validation Tuning")
        st.markdown("Select compute profiles, model candidates, cross-validation parameters, and preprocessing steps.")

        cfg_col1, cfg_col2 = st.columns([1, 1])

        with cfg_col1:
            st.markdown("#### 1. Compute Profile & Candidate Models")
            comp_profile = st.radio(
                "Compute Profile:",
                options=["Fast", "Balanced", "Deep"],
                index=["Fast", "Balanced", "Deep"].index(st.session_state.get("ml_compute_profile", "Balanced")),
                horizontal=True,
                help="Fast: Linear/Trees. Balanced: Random Forest & HistGBDT. Deep: XGBoost, LightGBM, Gradient Boosting.",
            )
            st.session_state["ml_compute_profile"] = comp_profile

            # Available models for task
            candidates = get_candidate_models(st.session_state["ml_task_type"], comp_profile)
            st.markdown(f"**Included Algorithms for this run ({len(candidates)}):**")
            
            selected_models = []
            for m_id in candidates:
                info = REGISTRY[m_id]
                checked = st.checkbox(f"{info.name} ({info.family.title()}, {info.speed})", value=True, key=f"cb_model_{m_id}")
                if checked:
                    selected_models.append(m_id)
            st.session_state["ml_selected_models"] = selected_models

        with cfg_col2:
            st.markdown("#### 2. Validation & Preprocessing Pipeline")
            
            test_split = st.slider(
                "Validation Split Ratio (Test Size):",
                min_value=0.10,
                max_value=0.40,
                value=float(st.session_state.get("ml_test_size", 0.20)),
                step=0.05,
                help="Proportion of dataset held out strictly for validation.",
            )
            st.session_state["ml_test_size"] = test_split

            cv_folds = st.slider(
                "Cross-Validation Folds ($K$):",
                min_value=2,
                max_value=8,
                value=int(st.session_state.get("ml_cv_folds", 3)),
                step=1,
                help="Folds used for cross-validation.",
            )
            st.session_state["ml_cv_folds"] = cv_folds

            scaler = st.selectbox(
                "Numeric Feature Scaler:",
                options=["standard", "robust", "minmax", "none"],
                index=["standard", "robust", "minmax", "none"].index(st.session_state.get("ml_scaler", "standard")),
                help="StandardScaler: Zero mean/unit variance. RobustScaler: Resilient to outliers. MinMaxScaler: [0, 1] bounds.",
            )
            st.session_state["ml_scaler"] = scaler

            cat_enc = st.selectbox(
                "Categorical Feature Encoding:",
                options=["onehot", "ordinal"],
                index=["onehot", "ordinal"].index(st.session_state.get("ml_cat_encoding", "onehot")),
                help="One-Hot: Unordered distinct categories. Ordinal: Integer encoding for high cardinality.",
            )
            st.session_state["ml_cat_encoding"] = cat_enc

    # -------------------------------------------------------------
    # TAB 4: TRAINING & BENCHMARKING
    # -------------------------------------------------------------
    with tabs[3]:
        st.markdown("### ⚡ Live Model Benchmarking & Leaderboard")
        st.markdown("Trains all active candidates with zero data leakage, compares primary performance metrics, and promotes the Champion Model.")

        if st.button("⚡ Launch Benchmark Now", type="primary", use_container_width=True):
            target_to_use = st.session_state.get("ml_target_col")
            if not target_to_use or target_to_use not in df.columns:
                st.error("Please select a valid target column in Tab 2 first.")
            else:
                feats_to_use = [c for c in st.session_state.get("ml_selected_features", []) if c in df.columns and c != target_to_use]
                if not feats_to_use:
                    st.error("Please select at least one feature column in Tab 2.")
                else:
                    with st.spinner("Benchmarking algorithmic candidates across folds..."):
                        try:
                            res = run_benchmark(
                                df=df,
                                target_col=target_to_use,
                                feature_cols=feats_to_use,
                                task_type=st.session_state["ml_task_type"],
                                compute_profile=st.session_state["ml_compute_profile"],
                                model_ids=st.session_state.get("ml_selected_models"),
                                test_size=st.session_state["ml_test_size"],
                                scaler_type=st.session_state["ml_scaler"],
                                cat_encoding=st.session_state["ml_cat_encoding"],
                            )
                            st.session_state["ml_benchmark_results"] = res
                            st.session_state["ml_active_model_id"] = res["champion_id"]

                            # Auto-persist Champion
                            champ_id = res["champion_id"]
                            if champ_id and champ_id in res["all_results"]:
                                champ_data = res["all_results"][champ_id]
                                save_experiment(
                                    pipeline=champ_data["pipeline"],
                                    model_id=champ_id,
                                    model_name=champ_data["model_name"],
                                    task_type=st.session_state["ml_task_type"],
                                    target_col=target_to_use,
                                    feature_cols=feats_to_use,
                                    metrics=champ_data["metrics"],
                                    dataset_name=proj_name,
                                    is_champion=True,
                                )

                            st.success("✅ Benchmarking completed successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Benchmarking error: {str(e)}")

        bench_data = st.session_state.get("ml_benchmark_results")
        if bench_data:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Champion Banner
            champ_id = bench_data.get("champion_id")
            champ_reason = bench_data.get("champion_reason", "")
            champ_info = REGISTRY.get(champ_id)
            champ_name = champ_info.name if champ_info else champ_id

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(16, 185, 129, 0.12) 100%); border: 1px solid #F59E0B; border-radius: 10px; padding: 16px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.6rem;">🏆</span>
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 800; color: #F59E0B; letter-spacing: 0.05em;">CHAMPION MODEL PROMOTED</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF;">{champ_name}</div>
                        <div style="font-size: 0.84rem; color: #E2E8F0; margin-top: 4px;">{champ_reason}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Leaderboard Table
            st.markdown("#### Algorithmic Leaderboard")
            lb_rows = []
            for entry in bench_data.get("leaderboard", []):
                primary_val = entry["score"]
                lb_rows.append({
                    "Rank": f"#{entry['rank']}",
                    "Model": entry["model_name"],
                    "Primary Score": f"{primary_val:.4f}" if primary_val > -900 else "Failed",
                    "Train Time": f"{entry['training_time']}s",
                    "Overfitting Risk": "⚠️ High" if entry["is_overfitting"] else "✅ Low",
                    "Status": "🏆 Champion" if entry.get("is_champion") else "Evaluated",
                })
            st.table(pd.DataFrame(lb_rows))

            # Select model to inspect in Evaluation / Explainability
            model_options = [entry["model_id"] for entry in bench_data.get("leaderboard", []) if entry["score"] > -900]
            if model_options:
                active_idx = model_options.index(st.session_state.get("ml_active_model_id", model_options[0])) if st.session_state.get("ml_active_model_id") in model_options else 0
                st.session_state["ml_active_model_id"] = st.selectbox(
                    "Select Model for In-Depth Evaluation & Diagnostics:",
                    options=model_options,
                    index=active_idx,
                    format_func=lambda m: f"{REGISTRY.get(m, {}).name if m in REGISTRY else m}",
                )
        else:
            st.info("No benchmark results yet. Click **'Launch Benchmark Now'** above to train and compare candidate models.")

    # -------------------------------------------------------------
    # TAB 5: EVALUATION & DIAGNOSTICS
    # -------------------------------------------------------------
    with tabs[4]:
        st.markdown("### 📊 Comprehensive Evaluation & Diagnostic Alerts")
        
        bench_data = st.session_state.get("ml_benchmark_results")
        active_model_id = st.session_state.get("ml_active_model_id")

        if bench_data and active_model_id and active_model_id in bench_data["all_results"]:
            eval_data = bench_data["all_results"][active_model_id]
            metrics = eval_data["metrics"]
            is_classification = "classification" in eval_data["task_type"]

            st.markdown(f"#### Model: **{eval_data['model_name']}**")

            # Metrics KPI Row
            if is_classification:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                m2.metric("F1 (Weighted)", f"{metrics.get('f1_weighted', 0):.3f}")
                m3.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                m4.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 'N/A')}")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
                m2.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                m3.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                m4.metric("MAPE", f"{metrics.get('mape', 0):.1%}")

            # Overfitting & Diagnostic Alerts
            if eval_data.get("is_overfitting"):
                st.error("⚠️ **Generalization Alert (Overfitting)**: Significant disparity detected between training performance and validation performance (> 15% gap). Regularization or feature pruning recommended.")
            else:
                st.success("✅ **Generalization Verified**: Validation performance aligns closely with training bounds. Model demonstrates high stability.")

            # Diagnostic Plots
            st.markdown("<br>", unsafe_allow_html=True)
            if is_classification:
                cm = eval_data.get("confusion_matrix")
                classes = eval_data.get("classes", ["0", "1"])
                if cm:
                    fig_cm = px.imshow(
                        cm,
                        x=classes,
                        y=classes,
                        labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
                        color_continuous_scale="Blues",
                        text_auto=True,
                        title="Confusion Matrix Heatmap",
                    )
                    fig_cm.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_cm, use_container_width=True)
            else:
                scatter_data = eval_data.get("scatter_sample", [])
                if scatter_data:
                    sc_df = pd.DataFrame(scatter_data)
                    c_plot1, c_plot2 = st.columns(2)
                    with c_plot1:
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=sc_df["actual"],
                            y=sc_df["predicted"],
                            mode="markers",
                            name="Holdout Points",
                            marker=dict(color="#00F0FF", size=7, opacity=0.8),
                        ))
                        # Ideal fit line
                        min_v = float(min(sc_df["actual"].min(), sc_df["predicted"].min()))
                        max_v = float(max(sc_df["actual"].max(), sc_df["predicted"].max()))
                        fig_scatter.add_trace(go.Scatter(
                            x=[min_v, max_v],
                            y=[min_v, max_v],
                            mode="lines",
                            name="Ideal Fit (y = x)",
                            line=dict(color="#F59E0B", dash="dash", width=2),
                        ))
                        fig_scatter.update_layout(
                            template="plotly_dark",
                            title="Actual vs Predicted Scatter",
                            xaxis_title="Actual Value",
                            yaxis_title="Predicted Value",
                            height=380,
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    with c_plot2:
                        fig_res = px.histogram(
                            sc_df,
                            x="residual",
                            nbins=25,
                            title="Residuals Error Distribution ($y_{true} - y_{pred}$)",
                            color_discrete_sequence=["#A855F7"],
                        )
                        fig_res.update_layout(template="plotly_dark", height=380)
                        st.plotly_chart(fig_res, use_container_width=True)

            # Hyperparameter Tuning Action
            st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
            st.markdown("#### ⚡ Hyperparameter Fine-Tuning")
            if st.button("🚀 Run Hyperparameter Optimization on this Model", type="secondary"):
                feats_to_use = [c for c in eval_data["feature_cols"] if c in df.columns]
                with st.spinner(f"Running hyperparameter optimization on {eval_data['model_name']}..."):
                    try:
                        tune_res = tune_model_hyperparameters(
                            df=df,
                            target_col=current_target,
                            feature_cols=feats_to_use,
                            task_type=st.session_state["ml_task_type"],
                            model_id=active_model_id,
                        )
                        if tune_res.get("tuned"):
                            st.success(f"Tuning complete in {tune_res['elapsed_seconds']}s! Best Score: {tune_res['tuned_score']:.4f} (Δ {tune_res['improvement']:+.4f})")
                            st.json(tune_res["best_params"])
                        else:
                            st.info(tune_res.get("message", "Model tuning not required."))
                    except Exception as e:
                        st.error(f"Tuning error: {str(e)}")
        else:
            st.info("Train candidate models in **Tab 4 (Training & Benchmarking)** to inspect diagnostic metrics.")

    # -------------------------------------------------------------
    # TAB 6: EXPLAINABLE AI
    # -------------------------------------------------------------
    with tabs[5]:
        st.markdown("### 🔍 Explainable AI (XAI) & Attribution")
        st.markdown("Global feature importances, local prediction explanations, and statistical causation disclaimers.")

        bench_data = st.session_state.get("ml_benchmark_results")
        active_model_id = st.session_state.get("ml_active_model_id")

        if bench_data and active_model_id and active_model_id in bench_data["all_results"]:
            eval_data = bench_data["all_results"][active_model_id]
            feat_imp = eval_data.get("feature_importance", [])

            if feat_imp:
                imp_df = pd.DataFrame(feat_imp)
                
                # Chart
                fig_bar = px.bar(
                    imp_df.head(15),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"Top 15 Feature Importances — {eval_data['model_name']}",
                    labels={"importance": "Relative Importance (%)", "feature": "Feature"},
                    color="importance",
                    color_continuous_scale="Viridis",
                )
                fig_bar.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), height=420)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Feature Table
                with st.expander("📋 View Complete Feature Importance Matrix", expanded=False):
                    st.dataframe(imp_df[["rank", "feature", "importance", "cumulative_importance"]], use_container_width=True)

            # Prominent Disclaimer Box
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #F59E0B; border-radius: 6px; padding: 12px 16px; margin: 16px 0;">
                <div style="font-size: 0.8rem; color: #CBD5E1; line-height: 1.5;">
                    {STATISTICAL_DISCLAIMER}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Train a model in **Tab 4** to inspect feature importance and attribution.")

    # -------------------------------------------------------------
    # TAB 7: PREDICTIONS & INFERENCE
    # -------------------------------------------------------------
    with tabs[6]:
        st.markdown("### 🔮 Model Inference & Prediction Simulator")
        st.markdown("Interactive single-record simulator and batch file prediction with zero mutation of the source dataset.")

        bench_data = st.session_state.get("ml_benchmark_results")
        active_model_id = st.session_state.get("ml_active_model_id")

        if not bench_data or not active_model_id or active_model_id not in bench_data["all_results"]:
            st.info("Please train a model first in **Tab 4 (Training & Benchmarking)** to enable inference.")
        else:
            eval_data = bench_data["all_results"][active_model_id]
            pipeline = eval_data["pipeline"]
            feat_cols = eval_data["feature_cols"]

            sub_pred_tab1, sub_pred_tab2 = st.tabs(["🧪 Interactive Record Simulator", "📦 Batch File Scoring"])

            # Single Record Simulator
            with sub_pred_tab1:
                st.markdown("#### Input Feature Values for Real-Time Prediction:")
                input_vals = {}
                
                # Render inputs in 3 columns
                cols_per_row = 3
                num_features = len(feat_cols)
                for i in range(0, num_features, cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < num_features:
                            f_name = feat_cols[i + j]
                            with row_cols[j]:
                                if f_name in df.columns:
                                    series = df[f_name]
                                    if pd.api.types.is_numeric_dtype(series):
                                        min_val = float(series.min()) if series.notna().any() else 0.0
                                        max_val = float(series.max()) if series.notna().any() else 100.0
                                        mean_val = float(series.mean()) if series.notna().any() else 50.0
                                        input_vals[f_name] = st.number_input(
                                            f"{f_name}:",
                                            value=round(mean_val, 2),
                                            key=f"sim_in_{f_name}_{i+j}",
                                        )
                                    else:
                                        unique_vals = [str(x) for x in series.dropna().unique()[:25]]
                                        default_val = unique_vals[0] if unique_vals else ""
                                        input_vals[f_name] = st.selectbox(
                                            f"{f_name}:",
                                            options=unique_vals if unique_vals else [""],
                                            key=f"sim_in_{f_name}_{i+j}",
                                        )
                                else:
                                    # Fallback if column was renamed or removed in active df
                                    input_vals[f_name] = st.text_input(
                                        f"{f_name}:",
                                        value="0",
                                        key=f"sim_in_{f_name}_{i+j}",
                                    )

                if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
                    try:
                        pred_res = predict_single_record(
                            pipeline=pipeline,
                            input_data=input_vals,
                            task_type=eval_data["task_type"],
                        )
                        pred_val = pred_res["prediction"]

                        # Local attribution explanation
                        valid_base_cols = [c for c in feat_cols if c in df.columns]
                        baseline_df = df[valid_base_cols] if valid_base_cols else pd.DataFrame([input_vals])

                        loc_exp = explain_local_prediction(
                            model=eval_data["estimator"],
                            preprocessor=eval_data["preprocessor"],
                            input_row=input_vals,
                            baseline_df=baseline_df,
                            predicted_value=pred_val,
                            feature_names=feat_cols,
                        )

                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(0, 240, 255, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%); border: 1px solid #00F0FF; border-radius: 10px; padding: 18px; margin: 16px 0; text-align: center;">
                            <div style="font-size: 0.76rem; font-weight: 800; color: #00F0FF; letter-spacing: 0.08em;">PREDICTED VALUE ({current_target})</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin: 8px 0;">{pred_val}</div>
                            <div style="font-size: 0.85rem; color: #CBD5E1;">{loc_exp['summary']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Show probabilities if classification
                        if pred_res.get("probabilities"):
                            st.markdown("##### Class Probability Distribution:")
                            prob_df = pd.DataFrame(list(pred_res["probabilities"].items()), columns=["Class", "Probability"])
                            st.bar_chart(prob_df.set_index("Class"))

                    except Exception as e:
                        st.error(f"Inference error: {str(e)}")

            # Batch Prediction
            with sub_pred_tab2:
                st.markdown("#### Batch Prediction Engine")
                batch_mode = st.radio("Batch Source:", ["Score Current Active Dataset", "Upload New CSV / XLSX File"], horizontal=True)

                batch_source_df = None
                if batch_mode == "Score Current Active Dataset":
                    batch_source_df = df
                    st.write(f"Scoring all **{len(df):,}** rows in current workspace.")
                else:
                    uploaded_batch = st.file_uploader("Upload Batch File:", type=["csv", "xlsx"])
                    if uploaded_batch:
                        if uploaded_batch.name.endswith(".csv"):
                            batch_source_df = pd.read_csv(uploaded_batch)
                        else:
                            batch_source_df = pd.read_excel(uploaded_batch)

                if batch_source_df is not None:
                    if st.button("🚀 Run Batch Prediction", type="primary"):
                        with st.spinner("Scoring dataset..."):
                            try:
                                scored_df = predict_batch(
                                    pipeline=pipeline,
                                    batch_df=batch_source_df,
                                    feature_cols=feat_cols,
                                    task_type=eval_data["task_type"],
                                    output_pred_col=f"predicted_{current_target}",
                                )
                                st.success(f"Batch inference complete! Generated {len(scored_df):,} predictions.")
                                st.dataframe(scored_df.head(10), use_container_width=True)

                                # Download CSV
                                csv_buffer = io.StringIO()
                                scored_df.to_csv(csv_buffer, index=False)
                                st.download_button(
                                    label="⬇ Download Scored CSV",
                                    data=csv_buffer.getvalue(),
                                    file_name=f"{proj_name}_scored_predictions.csv",
                                    mime="text/csv",
                                    type="primary",
                                )
                            except Exception as e:
                                st.error(f"Batch prediction error: {str(e)}")

    # -------------------------------------------------------------
    # TAB 8: TIME-SERIES FORECASTING
    # -------------------------------------------------------------
    with tabs[7]:
        st.markdown("### 📈 Time-Series Forecasting Engine")
        st.markdown("Automated temporal trend detection, exponential smoothing, ARIMA, and ML lag regression forecasting.")

        datetime_cols = readiness_report.datetime_columns
        potential_dates = [c for c in df.columns if any(k in str(c).lower() for k in ["date", "time", "day", "month", "year", "timestamp"])]
        date_candidates = list(dict.fromkeys(datetime_cols + potential_dates))
        numeric_cols = readiness_report.numeric_columns

        if not numeric_cols:
            st.warning("⚠️ Time-series forecasting requires at least one numeric metric column.")
        else:
            fc_col1, fc_col2, fc_col3 = st.columns(3)
            with fc_col1:
                if date_candidates:
                    date_col = st.selectbox("Date / Timestamp Column:", options=date_candidates)
                else:
                    date_col = "__row_index__"
                    st.info("No Date column detected. Using sequential row sequence.")
            with fc_col2:
                num_target = st.selectbox("Metric to Forecast:", options=[c for c in numeric_cols if c != date_col])
            with fc_col3:
                horizon = st.slider("Forecast Horizon (periods):", min_value=7, max_value=60, value=14)

            method_opt = st.selectbox(
                "Forecasting Algorithm:",
                options=["auto", "exponential_smoothing", "arima", "ml_lag", "naive"],
                format_func=lambda x: {
                    "auto": "Auto (Select Best: Holt-Winters / ARIMA / Lag Regression)",
                    "exponential_smoothing": "Exponential Smoothing (Holt-Winters)",
                    "arima": "ARIMA(1,1,1) Autoregressive Moving Average",
                    "ml_lag": "Machine Learning Lag Regression (Ridge)",
                    "naive": "Naive Persistence Baseline",
                }.get(x, x),
            )

            if st.button("🚀 Generate Time-Series Forecast", type="primary", use_container_width=True):
                with st.spinner("Fitting forecasting model and generating future confidence intervals..."):
                    try:
                        work_df = df.copy()
                        if date_col == "__row_index__":
                            # Create synthetic date sequence
                            work_df["__synthetic_date__"] = pd.date_range("2024-01-01", periods=len(work_df), freq="D")
                            actual_date_col = "__synthetic_date__"
                        else:
                            actual_date_col = date_col

                        fc_res = run_time_series_forecast(
                            df=work_df,
                            date_col=actual_date_col,
                            target_col=num_target,
                            horizon=horizon,
                            method=method_opt,
                        )
                        st.session_state["ml_forecast_results"] = fc_res
                        st.success("Forecast generated successfully!")
                    except Exception as e:
                        st.error(f"Forecasting error: {str(e)}")

            fc_data = st.session_state.get("ml_forecast_results")
            if fc_data:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"#### Projected Trend: {fc_data.get('method_used')}")
                st.info(fc_data.get("trend_summary"))

                # Chart Historical vs Forecast with Shaded Confidence Interval Band
                hist_pts = fc_data.get("historical", [])
                fore_pts = fc_data.get("forecast", [])

                if hist_pts and fore_pts:
                    h_df = pd.DataFrame(hist_pts)
                    f_df = pd.DataFrame(fore_pts)

                    fig_fc = go.Figure()

                    # Historical line
                    fig_fc.add_trace(go.Scatter(
                        x=h_df["date"],
                        y=h_df["actual"],
                        mode="lines+markers",
                        name="Historical Actual",
                        line=dict(color="#38BDF8", width=2),
                    ))

                    # Upper CI
                    fig_fc.add_trace(go.Scatter(
                        x=f_df["date"],
                        y=f_df["upper_ci"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    ))

                    # Lower CI + fill
                    fig_fc.add_trace(go.Scatter(
                        x=f_df["date"],
                        y=f_df["lower_ci"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(0, 240, 255, 0.15)",
                        name="90% Confidence Interval",
                    ))

                    # Forecast Line
                    fig_fc.add_trace(go.Scatter(
                        x=f_df["date"],
                        y=f_df["predicted"],
                        mode="lines+markers",
                        name="Point Forecast",
                        line=dict(color="#00F0FF", width=3, dash="dash"),
                    ))

                    fig_fc.update_layout(
                        template="plotly_dark",
                        title=f"{num_target} Forecast (Next {horizon} Periods)",
                        xaxis_title="Timeline",
                        yaxis_title=num_target,
                        height=440,
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)

                    # Metrics
                    fm = fc_data.get("metrics", {})
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric("Backtest MAE", f"{fm.get('mae', 0):.2f}")
                    c_m2.metric("Backtest RMSE", f"{fm.get('rmse', 0):.2f}")
                    c_m3.metric("Backtest MAPE", f"{fm.get('mape', 0):.1f}%")

    # -------------------------------------------------------------
    # TAB 9: EXPERIMENTS & REGISTRY
    # -------------------------------------------------------------
    with tabs[8]:
        st.markdown("### 📜 Experiment Registry & MLOps Versioning")
        st.markdown("Inspect historical training runs, serialized artifacts, rollback to previous champions, or export model pipelines.")

        all_experiments = list_experiments()
        if not all_experiments:
            st.info("No saved experiments in registry yet. Train models in **Tab 4** to create versions.")
        else:
            exp_table_data = []
            for exp in all_experiments:
                exp_table_data.append({
                    "Experiment ID": exp["experiment_id"],
                    "Model": exp.get("model_name", exp.get("model_id")),
                    "Task": exp.get("task_type"),
                    "Target": exp.get("target_col"),
                    "Created At": exp.get("created_at"),
                    "Champion": "🏆 Active Champion" if exp.get("is_champion") else "Archived",
                })
            st.table(pd.DataFrame(exp_table_data))

            st.markdown("#### Experiment Actions:")
            selected_exp_id = st.selectbox("Select Experiment ID:", options=[e["experiment_id"] for e in all_experiments])
            
            exp_meta = get_experiment(selected_exp_id)
            if exp_meta:
                c_act1, c_act2 = st.columns(2)
                with c_act1:
                    if st.button("🏆 Set as Active Production Champion", use_container_width=True):
                        set_active_champion(selected_exp_id)
                        st.success(f"Experiment `{selected_exp_id}` is now the Active Champion!")
                        st.rerun()

                with c_act2:
                    try:
                        import os
                        model_file_path = os.path.join(os.getcwd(), "data", "experiments", selected_exp_id, "model.joblib")
                        if os.path.exists(model_file_path):
                            with open(model_file_path, "rb") as f:
                                st.download_button(
                                    label="⬇ Download Model Pipeline (.joblib)",
                                    data=f.read(),
                                    file_name=f"{selected_exp_id}.joblib",
                                    mime="application/octet-stream",
                                    use_container_width=True,
                                )
                    except Exception as e:
                        st.error(f"Download error: {str(e)}")
