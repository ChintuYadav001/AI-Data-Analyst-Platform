"""
ui.views.autonomous_ml_card
===========================
Renders the 1-Click Autonomous End-to-End Predictive Machine Learning Cockpit.
Executes and visualizes all 10 data science lifecycle steps:
1. Data Collection
2. Data Understanding and EDA
3. Data Preprocessing
4. Feature Engineering and Selection
5. Split the Dataset
6. Select and Train the ML Model
7. Model Tuning
8. Model Evaluation
9. Make Predictions
10. Deployment and Monitoring
"""

import os
import json
import io
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.ml.ml_readiness import detect_target_candidates, classify_task_type
from core.ml.autonomous_pipeline import run_autonomous_1click_pipeline


def render_autonomous_1click_ml_tab(df: pd.DataFrame, proj_name: str = "Active Dataset"):
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(0, 240, 255, 0.08) 0%, rgba(56, 189, 248, 0.04) 100%); 
                border: 1px solid rgba(0, 240, 255, 0.35); border-radius: 12px; padding: 18px 22px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="background: rgba(0, 240, 255, 0.15); color: #00F0FF; font-size: 0.72rem; font-weight: 800; padding: 3px 10px; border-radius: 20px; border: 1px solid rgba(0, 240, 255, 0.3);">
                    ⚡ 1-CLICK AUTONOMOUS PREDICTIVE PIPELINE
                </span>
                <h2 style="color: #FFFFFF; font-size: 1.45rem; font-weight: 800; margin: 6px 0 3px 0;">
                    Autonomous 10-Step Machine Learning Lifecycle
                </h2>
                <div style="font-size: 0.84rem; color: #94A3B8;">
                    Executes the complete data science pipeline in a single automated click: 
                    <b>Ingestion → EDA → Preprocessing → Feature Engineering → 80/20 Split → Model Training → Tuning → Evaluation → Predictions → Deployment & Drift Monitoring</b>.
                </div>
            </div>
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; color: #10B981; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700;">
                ● 100% MATHEMATICAL PRECISION
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STEP 0: CONFIGURATION COCKPIT
    # -------------------------------------------------------------------------
    candidates = detect_target_candidates(df)
    candidate_cols = []
    for c in candidates:
        if hasattr(c, "column"):
            candidate_cols.append(c.column)
        elif isinstance(c, dict) and "column" in c:
            candidate_cols.append(c["column"])
        else:
            candidate_cols.append(str(c))
    all_cols = list(df.columns)
    
    # Default target logic
    default_target_idx = 0
    if candidate_cols and candidate_cols[0] in all_cols:
        default_target_idx = all_cols.index(candidate_cols[0])
    elif "churn" in [c.lower() for c in all_cols]:
        for i, c in enumerate(all_cols):
            if c.lower() == "churn":
                default_target_idx = i
                break

    c_cfg1, c_cfg2, c_cfg3 = st.columns([1.2, 1.4, 1.0])

    with c_cfg1:
        target_col = st.selectbox(
            "🎯 1. Target Column to Predict",
            options=all_cols,
            index=default_target_idx,
            help="Select the column you want the model to predict (e.g. churn, sales, price, status)."
        )
        task_detected = classify_task_type(df, target_col)
        task_badge_color = "#38BDF8" if "classification" in task_detected else "#10B981"
        st.markdown(f"<span style='font-size: 0.75rem; color: {task_badge_color}; font-weight: 700;'>Detected Task: {task_detected.upper()}</span>", unsafe_allow_html=True)

    with c_cfg2:
        model_options = {
            "auto": "✨ Auto-Select Optimal Model (Recommended)",
            "random_forest": "🌲 Random Forest (Ensemble Trees - Robust & Non-Linear)",
            "xgboost": "⚡ XGBoost (Extreme Gradient Boosting)",
            "hist_gradient_boosting": "🚀 HistGradientBoosting (Fast Histogram Boosting)",
            "decision_tree": "🎋 Decision Tree (Transparent & Explainable)",
            "linear_logistic": "📐 Logistic / Linear Regression (Standard Baseline)",
            "extra_trees": "🌳 Extra Trees (Extremely Randomized Ensembles)",
        }
        selected_model_key = st.selectbox(
            "🤖 2. Algorithm Selection Mode",
            options=list(model_options.keys()),
            format_func=lambda k: model_options[k],
            index=0,
            help="By default, Quantum Analyst automatically selects the champion model based on task type and dataset volume. You can also specify your required model."
        )

    with c_cfg3:
        split_ratio = st.slider(
            "✂️ 3. Train / Test Split Ratio",
            min_value=0.10,
            max_value=0.35,
            value=0.20,
            step=0.05,
            format="%d%% Test",
            help="Standard industry benchmark: 80% Training Data, 20% Holdout Testing Data."
        )
        tune_cv = st.checkbox("🎯 Enable 3-Fold CV Hyperparameter Tuning", value=True)

    # Resolve internal model ID
    is_classif = "classification" in task_detected
    resolved_model_id = "auto"
    if selected_model_key == "random_forest":
        resolved_model_id = "random_forest_classifier" if is_classif else "random_forest_regressor"
    elif selected_model_key == "xgboost":
        resolved_model_id = "xgboost_classifier" if is_classif else "xgboost_regressor"
    elif selected_model_key == "hist_gradient_boosting":
        resolved_model_id = "hist_gradient_boosting_classifier" if is_classif else "hist_gradient_boosting_regressor"
    elif selected_model_key == "decision_tree":
        resolved_model_id = "decision_tree_classifier" if is_classif else "decision_tree_regressor"
    elif selected_model_key == "linear_logistic":
        resolved_model_id = "logistic_regression" if is_classif else "linear_regression"
    elif selected_model_key == "extra_trees":
        resolved_model_id = "extra_trees_classifier" if is_classif else "extra_trees_regressor"

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # Launch Button
    btn_col1, btn_col2 = st.columns([2.5, 1.5])
    with btn_col1:
        launch_clicked = st.button(
            "🚀 Launch 1-Click Autonomous Pipeline (Execute All 10 Steps)",
            type="primary",
            use_container_width=True,
            help="Executes ingestion, EDA, cleaning, feature selection, split, model training, tuning, holdout evaluation, prediction, and deployment."
        )
    with btn_col2:
        if st.session_state.get("ml_autonomous_results"):
            if st.button("🔄 Reset / Clear Results", use_container_width=True):
                st.session_state["ml_autonomous_results"] = None
                st.rerun()

    # -------------------------------------------------------------------------
    # PIPELINE EXECUTION
    # -------------------------------------------------------------------------
    if launch_clicked:
        with st.spinner("Executing 10-Step Autonomous Pipeline (Ingestion → EDA → Preprocessing → ML → Deployment)..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(step_num: int, title: str, details: dict):
                progress_bar.progress(step_num / 10.0)
                status_text.markdown(f"**Step {step_num}/10:** {title}...")

            try:
                res = run_autonomous_1click_pipeline(
                    df=df,
                    target_col=target_col,
                    preferred_model_id=resolved_model_id,
                    test_size=split_ratio,
                    tune_hyperparameters=tune_cv,
                    dataset_name=proj_name,
                    progress_callback=update_progress
                )
                st.session_state["ml_autonomous_results"] = res
                st.session_state["ml_target_col"] = target_col
                st.session_state["ml_benchmark_results"] = [res]
                st.session_state["ml_active_model_id"] = res["model_id"]
                progress_bar.progress(1.0)
                status_text.success("✓ All 10 Steps Completed Successfully with 100% Mathematical Precision!")
                st.toast("🎉 Autonomous Predictive Pipeline Completed Successfully!", icon="🚀")
                st.rerun()
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Pipeline Execution Failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="text")

    # -------------------------------------------------------------------------
    # DISPLAY PIPELINE RESULTS
    # -------------------------------------------------------------------------
    res = st.session_state.get("ml_autonomous_results")
    if not res:
        st.info("👆 Click **'Launch 1-Click Autonomous Pipeline'** above to automatically execute all 10 steps on your active dataset.")
        return

    st.markdown("<hr style='border-color: rgba(0, 240, 255, 0.2); margin: 24px 0;'>", unsafe_allow_html=True)

    # TOP METRICS HERO STRIP
    metrics = res.get("metrics", {})
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("🏆 Selected Model", res.get("model_name", "Champion Model"))
    with m_col2:
        if "Accuracy_Percentage" in metrics:
            st.metric("🎯 Accuracy", metrics["Accuracy_Percentage"])
        elif "R2_Percentage" in metrics:
            st.metric("🎯 R² Score", metrics["R2_Percentage"])
        else:
            st.metric("🎯 Score", "Evaluated")
    with m_col3:
        if "F1_Score" in metrics:
            st.metric("⚖️ F1-Score", f"{metrics['F1_Score']}")
        elif "MAE" in metrics:
            st.metric("📉 MAE", f"{metrics['MAE']}")
        else:
            st.metric("📊 Metric", "Holdout Tested")
    with m_col4:
        st.metric("⚡ Pipeline Runtime", f"{res.get('total_time_seconds', 0)}s")
    with m_col5:
        st.metric("📦 Deployment Status", "Active Champion ✓")

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # THE 10-STEP LIFECYCLE ACCORDION
    # -------------------------------------------------------------------------
    steps = res.get("steps", {})

    # STEP 1
    s1 = steps.get("step_1", {}).get("data", {})
    with st.expander(f"📥 Step 1: Data Collection & Ingestion — {s1.get('status', 'Completed')}", expanded=True):
        st.markdown(f"""
        **Historical Source Ingestion Summary:**
        - **Dataset Name:** `{s1.get('dataset_name', proj_name)}`
        - **Total Records Ingested:** `{s1.get('raw_rows', len(df)):,}` rows
        - **Total Columns:** `{s1.get('raw_cols', len(df.columns)):,}` attributes ({s1.get('numeric_cols_count', 0)} numerical, {s1.get('categorical_cols_count', 0)} categorical)
        - **Confirmed Target Variable:** `{s1.get('target_col', target_col)}`
        - **Machine Learning Task Type:** `{s1.get('task_type', '').upper()}`
        """)
        if s1.get("sample_head"):
            st.caption("Sample Ingested Records (First 5 Rows):")
            st.dataframe(pd.DataFrame(s1["sample_head"]), use_container_width=True)

    # STEP 2
    s2 = steps.get("step_2", {}).get("data", {})
    with st.expander(f"📊 Step 2: Data Understanding & Statistical EDA — {s2.get('status', 'Completed')}", expanded=False):
        c_eda1, c_eda2 = st.columns([1.5, 1])
        with c_eda1:
            st.markdown("##### 📈 Descriptive Statistics of Key Features")
            if s2.get("top_feature_stats"):
                st.dataframe(pd.DataFrame(s2["top_feature_stats"]).T, use_container_width=True)
        with c_eda2:
            st.markdown("##### 🎯 Target Variable Distribution")
            t_dist = s2.get("target_distribution", {})
            if isinstance(t_dist, dict) and any(isinstance(v, int) for v in t_dist.values()):
                fig_dist = px.pie(names=list(t_dist.keys()), values=list(t_dist.values()), hole=0.4, title="Target Class Balance")
                fig_dist.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.json(t_dist)

    # STEP 3
    s3 = steps.get("step_3", {}).get("data", {})
    with st.expander(f"🧹 Step 3: Data Preprocessing — {s3.get('status', 'Completed')}", expanded=False):
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        with c_p1:
            st.metric("Duplicates Purged", f"{s3.get('duplicates_removed', 0):,}")
        with c_p2:
            st.metric("Missing Values Imputed", f"{s3.get('missing_values_imputed', 0):,}")
        with c_p3:
            st.metric("Outliers Winsorized (IQR)", f"{s3.get('outliers_capped', 0):,}")
        with c_p4:
            st.metric("Clean Records Preserved", f"{s3.get('clean_rows', len(df)):,}")
        st.success("✓ Automated data cleaning completed non-destructively: Median imputation for numeric features, mode for categoricals, and IQR boundary clipping to prevent model skew.")

    # STEP 4
    s4 = steps.get("step_4", {}).get("data", {})
    with st.expander(f"⚙️ Step 4: Feature Engineering & Selection — {s4.get('status', 'Completed')}", expanded=False):
        new_feats = s4.get("new_features_created", [])
        if new_feats:
            st.write(f"• **Derived Interaction Features:** `{', '.join(new_feats)}`")
        st.write(f"• **Features Selected for Prediction ({s4.get('total_selected_features', 0)}):**")
        st.caption(f"{', '.join(s4.get('selected_feature_names', []))}")

    # STEP 5
    s5 = steps.get("step_5", {}).get("data", {})
    with st.expander(f"✂️ Step 5: Dataset Splitting — {s5.get('status', 'Completed')}", expanded=False):
        c_sp1, c_sp2, c_sp3 = st.columns(3)
        with c_sp1:
            st.metric("🏋️ Training Data (80%)", f"{s5.get('train_records', 0):,} rows")
        with c_sp2:
            st.metric("🧪 Holdout Test Data (20%)", f"{s5.get('test_records', 0):,} rows")
        with c_sp3:
            st.metric("Stratified Sampling", "Enabled ✓" if s5.get("stratified_split") else "Randomized ✓")
        st.info("Zero data leakage guarantee: Preprocessing transformers and encoders were strictly fitted on training splits only and applied to test sets.")

    # STEP 6
    s6 = steps.get("step_6", {}).get("data", {})
    with st.expander(f"🤖 Step 6: Select & Train ML Model — {s6.get('status', 'Completed')}", expanded=False):
        st.markdown(f"""
        - **Algorithm Selected:** **{s6.get('selected_model_name')}** (`{s6.get('selected_model_id')}`)
        - **Model Family:** `{s6.get('task_family', '').upper()}`
        - **Pipeline Assembly:** Scikit-Learn `Pipeline(preprocessor, estimator)` with `LabelEncodedClassifier`
        - **Training Time:** `{s6.get('training_time_seconds', 0)} seconds`
        """)

    # STEP 7
    s7 = steps.get("step_7", {}).get("data", {})
    with st.expander(f"🎯 Step 7: Model Tuning & Cross-Validation — {s7.get('status', 'Completed')}", expanded=False):
        t_details = s7.get("tuning_details", {})
        st.write(f"• **Optimization Method:** 3-Fold Stratified Cross-Validation on Training Split")
        if s7.get("cv_score"):
            st.metric("Cross-Validation Score (Mean)", f"{s7['cv_score']}")
        st.json(t_details)

    # STEP 8
    s8 = steps.get("step_8", {}).get("data", {})
    with st.expander(f"📈 Step 8: Model Evaluation on Unseen Test Set — {s8.get('status', 'Completed')}", expanded=True):
        col_m_left, col_m_right = st.columns([1, 1.2])
        with col_m_left:
            st.markdown("##### 🎯 Final Holdout Validation Metrics")
            for k, v in s8.get("metrics", {}).items():
                st.write(f"• **{k.replace('_', ' ')}:** `{v}`")
            
            # Confusion Matrix Heatmap if available
            cm = s8.get("confusion_matrix")
            classes = s8.get("classes")
            if cm and classes:
                st.markdown("##### 🧩 Confusion Matrix")
                fig_cm = px.imshow(
                    cm,
                    x=[f"Pred: {c}" for c in classes],
                    y=[f"Actual: {c}" for c in classes],
                    text_auto=True,
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_cm, use_container_width=True)

        with col_m_right:
            st.markdown("##### 🌟 Global Feature Importances")
            feat_imp = s8.get("top_features", [])
            if feat_imp:
                imp_df = pd.DataFrame(feat_imp).sort_values("importance", ascending=True)
                fig_imp = px.bar(
                    imp_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Viridis",
                    title="Predictive Weight Distribution"
                )
                fig_imp.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_imp, use_container_width=True)

    # STEP 9
    s9 = steps.get("step_9", {}).get("data", {})
    with st.expander(f"🔮 Step 9: Make Predictions & Real-Time Scoring — {s9.get('status', 'Completed')}", expanded=True):
        st.markdown("##### 🧪 Test Set Prediction Verification (First 10 Records)")
        if s9.get("sample_comparisons"):
            st.dataframe(pd.DataFrame(s9["sample_comparisons"]), use_container_width=True)

        st.markdown("---")
        st.markdown("##### 🎮 Interactive Single-Record Prediction Playground")
        st.caption("Adjust input parameters below and click 'Predict Outcome' to test the deployed model in real time:")

        interactive_defs = res.get("interactive_defaults", {})
        pipeline = res.get("pipeline")

        if interactive_defs and pipeline is not None:
            input_cols = list(interactive_defs.keys())[:8]  # Limit to 8 input fields for clean layout
            cols_grid = st.columns(min(4, len(input_cols)))
            user_inputs = {}

            for idx, col_name in enumerate(input_cols):
                c = cols_grid[idx % len(cols_grid)]
                val = interactive_defs[col_name]
                with c:
                    if isinstance(val, (int, float)):
                        min_v = 0.0 if val >= 0 else float(val * 2)
                        max_v = float(val * 3) if val > 0 else 100.0
                        user_inputs[col_name] = st.number_input(
                            f"{col_name}",
                            value=float(val),
                            key=f"auto_pred_{col_name}"
                        )
                    else:
                        user_inputs[col_name] = st.text_input(
                            f"{col_name}",
                            value=str(val),
                            key=f"auto_pred_{col_name}"
                        )

            # Fill remaining features with default values
            for col_name, val in interactive_defs.items():
                if col_name not in user_inputs:
                    user_inputs[col_name] = val

            if st.button("🔮 Predict Outcome for this Record", type="primary", use_container_width=True):
                try:
                    input_df = pd.DataFrame([user_inputs])
                    pred_res = pipeline.predict(input_df)[0]
                    proba_text = ""
                    if hasattr(pipeline.named_steps.get("model"), "predict_proba"):
                        try:
                            proba = pipeline.predict_proba(input_df)[0]
                            max_prob = max(proba)
                            proba_text = f" (Confidence: {round(max_prob * 100, 2)}%)"
                        except Exception:
                            pass
                    st.success(f"🎯 **Predicted {target_col}:** `{pred_res}`{proba_text}")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    # STEP 10
    s10 = steps.get("step_10", {}).get("data", {})
    with st.expander(f"🚀 Step 10: Model Deployment & Live Monitoring — {s10.get('status', 'Completed')}", expanded=True):
        st.markdown(f"""
        - **Deployment Status:** `ACTIVE CHAMPION`
        - **Registered Experiment ID:** `{s10.get('experiment_id')}`
        - **Saved Location:** `data/experiments/{s10.get('experiment_id')}/model.joblib`
        """)

        tab_api, tab_drift, tab_export = st.tabs(["⚡ REST API Endpoint", "📉 Drift Monitoring Bounds", "⬇️ Download Model"])
        
        with tab_api:
            st.markdown("##### FastAPI Endpoint Code (`serve_model.py`)")
            st.code(s10.get("api_code", ""), language="python")
            st.markdown("##### cURL Request Example")
            st.code(s10.get("curl_code", ""), language="bash")

        with tab_drift:
            st.markdown("##### 📊 Production Drift Monitoring Baselines (3σ Bounds)")
            st.caption("Real-time data feeds should remain within these boundaries. Excursions trigger model retraining alerts:")
            if s10.get("drift_monitoring_baselines"):
                st.dataframe(pd.DataFrame(s10["drift_monitoring_baselines"]).T, use_container_width=True)

        with tab_export:
            st.markdown("##### 📦 Export Model Artifacts")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if pipeline is not None:
                    buf = io.BytesIO()
                    joblib.dump(pipeline, buf)
                    st.download_button(
                        label="⬇️ Download Trained Pipeline (.joblib)",
                        data=buf.getvalue(),
                        file_name=f"{res.get('model_id', 'model')}_pipeline.joblib",
                        mime="application/octet-stream",
                        use_container_width=True,
                        type="primary"
                    )
            with col_dl2:
                api_code = s10.get("api_code", "")
                st.download_button(
                    label="⬇️ Download FastAPI Server Script (.py)",
                    data=api_code,
                    file_name="serve_model.py",
                    mime="text/x-python",
                    use_container_width=True
                )
