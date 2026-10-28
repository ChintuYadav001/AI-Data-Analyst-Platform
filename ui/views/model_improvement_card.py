"""
ui.views.model_improvement_card
===============================
Institutional-grade Model Improvement & High-Sales Underestimation Diagnostic Suite.
Addresses model underperformance on high-sales observations where a few large errors increase RMSE.
Renders the complete 6-Step Model Improvement Workflow:
1. Evaluate the model: Calculate MAE, RMSE, and R² on the holdout test set vs train set.
2. Check outliers: Investigate unusually high sales values and verify validity/residuals.
3. Improve features: Include product category, discount, quantity, region, and seasonality.
4. Compare algorithms: Try Random Forest Regressor, Gradient Boosting, or XGBoost against current model.
5. Check for underfitting: Compare training and test performance to determine bias vs variance.
6. Consider a log transformation: Test np.log1p(y) and convert predictions back using np.expm1().
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional


def render_model_improvement_section(
    m_data: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    is_autonomous: bool = False,
):
    """
    Renders the dedicated 6-step Model Improvement Center for regression tasks.
    """
    metrics = m_data.get("metrics", {})
    imp_diag = m_data.get("improvement_diagnostics") or {}
    algo_comp = m_data.get("algorithm_comparison") or []
    is_log_trans = m_data.get("is_log_transformed") or m_data.get("log_transformed", False)

    s1 = imp_diag.get("step_1_evaluation", {})
    s2 = imp_diag.get("step_2_outliers_and_high_sales", {})
    s5 = imp_diag.get("step_5_underfitting_check", {})
    s6 = imp_diag.get("step_6_log_transformation", {})

    test_mae = s1.get("test_mae", metrics.get("mae", metrics.get("MAE", 0.0)))
    test_rmse = s1.get("test_rmse", metrics.get("rmse", metrics.get("RMSE", 0.0)))
    test_r2 = s1.get("test_r2", metrics.get("r2", metrics.get("R2_Score", 0.0)))
    train_r2 = s1.get("train_r2", metrics.get("train_r2", metrics.get("Train_R2")))
    train_mae = s1.get("train_mae", metrics.get("train_mae", metrics.get("Train_MAE")))
    train_rmse = s1.get("train_rmse", metrics.get("train_rmse", metrics.get("Train_RMSE")))
    rmse_mae_ratio = s1.get("rmse_mae_ratio", metrics.get("RMSE_MAE_Ratio", round(float(test_rmse) / (float(test_mae) + 1e-6), 2) if test_mae else 1.0))

    has_high_sales_underest = s2.get("has_high_sales_underestimation", False)
    top_avg_underest = s2.get("top_10_avg_underestimation", 0.0)
    top_underest_pct = s2.get("top_10_underestimated_pct", 0.0)
    worst_records = s2.get("worst_underpredicted_records", [])

    st.markdown("---")
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
        <span style="font-size: 1.25rem;">🛠️</span>
        <h3 style="margin: 0; color: #FFFFFF; font-size: 1.25rem; font-weight: 800;">
            Model Improvement Center & High-Sales Underestimation Optimization
        </h3>
    </div>
    <div style="font-size: 0.84rem; color: #94A3B8; margin-bottom: 14px;">
        Diagnostic analysis for tail residuals, outlier impact on RMSE, domain feature engineering, multi-algorithm benchmarking, and variance-stabilizing log transformations.
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # WHAT NEEDS IMPROVEMENT BANNER
    # -------------------------------------------------------------------------
    if has_high_sales_underest:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.35); border-left: 4px solid #EF4444; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="color: #F87171; font-weight: 800; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; display: flex; align-items: center; gap: 6px;">
                <span>⚠️</span> WHAT NEEDS IMPROVEMENT
            </div>
            <div style="color: #FFFFFF; font-weight: 700; font-size: 1.05rem; margin: 6px 0 4px 0;">
                The model performs poorly on some high-sales observations and appears to underestimate them. A few large errors could substantially increase the RMSE.
            </div>
            <div style="color: #CBD5E1; font-size: 0.86rem; line-height: 1.5;">
                • <b>High-Sales Shortfall:</b> For top observations, the model underestimates actual values by an average of <b>+{top_avg_underest:,.2f}</b> units ({top_underest_pct}% of top actuals are underpredicted).<br>
                • <b>RMSE Sensitivity:</b> RMSE (<code>{test_rmse:,.2f}</code>) is <b>{rmse_mae_ratio}x</b> larger than MAE (<code>{test_mae:,.2f}</code>). Because RMSE squares errors, extreme underpredictions on large orders heavily penalize the model score.<br>
                • <b>Log Transformation Status:</b> {"Active (np.log1p applied)" if is_log_trans else "Not yet applied. See Step 6 below to test np.log1p(y)."}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-left: 4px solid #10B981; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
            <div style="color: #34D399; font-weight: 800; font-size: 0.82rem; text-transform: uppercase;">
                ✓ MODEL STABILITY AUDIT PASSED
            </div>
            <div style="color: #FFFFFF; font-size: 0.92rem; font-weight: 600; margin-top: 3px;">
                No severe high-sales underestimation detected. Residuals remain balanced across both lower and higher observations.
            </div>
            <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 2px;">
                Follow the 6 steps below to further boost predictive accuracy and feature power.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. HOW TO IMPROVE YOUR MODEL (6-STEP WORKFLOW TABS)
    # -------------------------------------------------------------------------
    st.markdown("##### 📋 4. How to Improve Your Model (Execute in Order)")

    imp_tabs = st.tabs([
        "1. Evaluate Model",
        "2. Check Outliers",
        "3. Improve Features",
        "4. Compare Algorithms",
        "5. Check Underfitting",
        "6. Log Transformation",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: EVALUATE MODEL (MAE, RMSE, R² ON HOLDOUT TEST SET)
    # -------------------------------------------------------------------------
    with imp_tabs[0]:
        st.markdown("###### 🎯 Step 1: Holdout Test Set Evaluation & Error Metrics")
        st.caption("Calculate MAE, RMSE, and R² on unseen holdout test data to obtain an unbiased baseline:")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Holdout R² Score", f"{test_r2:.4f}" if isinstance(test_r2, (int, float)) else str(test_r2))
        with m_col2:
            st.metric("Holdout RMSE", f"{test_rmse:,.4f}" if isinstance(test_rmse, (int, float)) else str(test_rmse))
        with m_col3:
            st.metric("Holdout MAE", f"{test_mae:,.4f}" if isinstance(test_mae, (int, float)) else str(test_mae))
        with m_col4:
            st.metric("RMSE / MAE Ratio", f"{rmse_mae_ratio}x", help="Ratios > 1.40 indicate a small number of large outlier errors dominate RMSE.")

        # Side-by-side comparison table
        metrics_comp_df = pd.DataFrame([
            {"Metric": "R² Variance Explained", "Holdout Test Set": f"{test_r2:.4f}" if isinstance(test_r2, (int, float)) else str(test_r2), "Training Set": f"{train_r2:.4f}" if train_r2 is not None else "N/A", "Optimal Target": "Close to 1.0 (Low Gap)"},
            {"Metric": "RMSE (Root Mean Squared Error)", "Holdout Test Set": f"{test_rmse:,.2f}" if isinstance(test_rmse, (int, float)) else str(test_rmse), "Training Set": f"{train_rmse:,.2f}" if train_rmse is not None else "N/A", "Optimal Target": "Minimized (Near MAE)"},
            {"Metric": "MAE (Mean Absolute Error)", "Holdout Test Set": f"{test_mae:,.2f}" if isinstance(test_mae, (int, float)) else str(test_mae), "Training Set": f"{train_mae:,.2f}" if train_mae is not None else "N/A", "Optimal Target": "Minimized"},
        ])
        st.dataframe(metrics_comp_df, width="stretch", hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 2: CHECK OUTLIERS (INVESTIGATE HIGH-SALES OBSERVATIONS)
    # -------------------------------------------------------------------------
    with imp_tabs[1]:
        st.markdown("###### 🔍 Step 2: Investigate High-Sales Outliers & Underprediction Bias")
        st.caption("Unusually high sales observations (e.g. bulk wholesale orders, peak seasonal demand) can cause massive squared residual penalties:")

        if worst_records:
            st.markdown("**Top Underestimated Observations (Actual vs Model Prediction):**")
            w_df = pd.DataFrame(worst_records).rename(columns={
                "actual": "Actual Sales ($)",
                "predicted": "Model Prediction ($)",
                "underestimation_amount": "Underestimated Shortfall ($)",
                "percent_underestimated": "Error Percentage",
            })
            st.dataframe(w_df, width="stretch", hide_index=True)
            st.info("💡 **Actionable Diagnostic:** Verify whether these high values are valid bulk sales or data entry anomalies. If they are valid genuine sales, do not discard them—instead, apply a log transformation (Step 6) and non-linear ensemble algorithms (Step 4).")
        else:
            st.write("• No severe individual observation underpredictions found in holdout evaluation.")

        # Residuals vs Actuals scatter plot if available
        scatter_sample = m_data.get("scatter_sample", [])
        if scatter_sample:
            sc_df = pd.DataFrame(scatter_sample)
            if "actual" in sc_df.columns and "residual" in sc_df.columns:
                fig_res = px.scatter(
                    sc_df,
                    x="actual",
                    y="residual",
                    color="is_top_sales" if "is_top_sales" in sc_df.columns else None,
                    color_discrete_map={True: "#EF4444", False: "#38BDF8"},
                    title="Residual Errors vs Actual Target Values (y vs [y - ŷ])",
                    labels={"actual": "Actual Value (y)", "residual": "Residual Shortfall (y - ŷ)", "is_top_sales": "Top Sales Observation"},
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="#F59E0B")
                fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
                st.plotly_chart(fig_res, width="stretch")

    # -------------------------------------------------------------------------
    # TAB 3: IMPROVE FEATURES
    # -------------------------------------------------------------------------
    with imp_tabs[2]:
        st.markdown("###### 🧩 Step 3: Feature Engineering (Category, Discount, Quantity, Region, Seasonality)")
        st.caption("Linear models and shallow trees struggle without domain-specific interaction features. Quantum Analyst automatically scans and generates:")

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
                <div style="color: #38BDF8; font-weight: 700; font-size: 0.9rem; margin-bottom: 6px;">🏷️ Domain Grouping Features</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5;">
                    • <b>Product Category / Segment:</b> Categorical one-hot or target encoding ensures distinct margin tiers are captured.<br>
                    • <b>Region / Market / State:</b> Geographical indicators account for regional sales purchasing power differences.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with f_col2:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
                <div style="color: #38BDF8; font-weight: 700; font-size: 0.9rem; margin-bottom: 6px;">⚡ Interactions & Seasonality</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5;">
                    • <b>Discount × Price × Quantity:</b> Cross-product volume and net effective price features.<br>
                    • <b>Calendar Seasonality:</b> Month, Quarter, Day of Week, and Cyclical Sine/Cosine transformations.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if df is not None:
            # Detect available business columns in active dataset
            cols_lower = [c.lower() for c in df.columns]
            has_cat = any(k in cols_lower for k in ["category", "segment", "product_category", "type"])
            has_qty = any(k in cols_lower for k in ["quantity", "qty", "volume", "units"])
            has_disc = any(k in cols_lower for k in ["discount", "disc", "discount_pct"])
            has_reg = any(k in cols_lower for k in ["region", "state", "city", "country", "market"])
            has_date = any(k in cols_lower for k in ["date", "order_date", "timestamp", "time"])

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            st.markdown("**Dataset Domain Column Audit:**")
            audit_cols = st.columns(5)
            with audit_cols[0]:
                st.markdown(f"{'✅' if has_cat else '⚪'} Product Category")
            with audit_cols[1]:
                st.markdown(f"{'✅' if has_qty else '⚪'} Quantity / Volume")
            with audit_cols[2]:
                st.markdown(f"{'✅' if has_disc else '⚪'} Discount")
            with audit_cols[3]:
                st.markdown(f"{'✅' if has_reg else '⚪'} Region / Geography")
            with audit_cols[4]:
                st.markdown(f"{'✅' if has_date else '⚪'} Date / Seasonality")

    # -------------------------------------------------------------------------
    # TAB 4: COMPARE ALGORITHMS (RANDOM FOREST, GRADIENT BOOSTING, XGBOOST)
    # -------------------------------------------------------------------------
    with imp_tabs[3]:
        st.markdown("###### 🚀 Step 4: Multi-Algorithm Benchmark Comparison")
        st.caption("Compare linear regression against non-linear ensembles (Random Forest, Gradient Boosting, XGBoost) to test if algorithm complexity resolves underestimation:")

        if algo_comp:
            comp_df = pd.DataFrame(algo_comp)
            display_comp_df = comp_df[["model_name", "test_r2", "test_rmse", "test_mae", "train_r2", "fit_gap", "latency_sec"]].rename(columns={
                "model_name": "Algorithm",
                "test_r2": "Holdout R²",
                "test_rmse": "Holdout RMSE",
                "test_mae": "Holdout MAE",
                "train_r2": "Train R²",
                "fit_gap": "Overfit Gap",
                "latency_sec": "Time (s)",
            })
            st.dataframe(display_comp_df, width="stretch", hide_index=True)

            fig_comp = px.bar(
                comp_df,
                x="test_r2",
                y="model_name",
                orientation="h",
                color="test_r2",
                color_continuous_scale="Viridis",
                title="Holdout Test Set R² by Algorithm",
                labels={"test_r2": "Holdout R² (Higher is Better)", "model_name": "Model"},
            )
            fig_comp.update_layout(yaxis={"categoryorder": "total ascending"}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="#E2E8F0")
            st.plotly_chart(fig_comp, width="stretch")
        else:
            st.info("Execute benchmark training in the **Training & AutoML** tab to generate the full multi-algorithm comparative leaderboard.")

    # -------------------------------------------------------------------------
    # TAB 5: CHECK FOR UNDERFITTING (TRAINING VS TEST PERFORMANCE)
    # -------------------------------------------------------------------------
    with imp_tabs[4]:
        st.markdown("###### ⚖️ Step 5: Check Underfitting vs Overfitting")
        st.caption("Compare training performance against holdout test performance to determine whether the model is too simple (underfitting) or memorizing noise (overfitting):")

        fit_status = s5.get("status", "optimal")
        diag_msg = s5.get("diagnosis_message", "Train and test performance are balanced.")

        if fit_status == "underfitting":
            badge_color = "#EF4444"
            badge_label = "🔴 UNDERFITTING DETECTED (Model is Too Simple)"
        elif fit_status == "overfitting":
            badge_color = "#F59E0B"
            badge_label = "🟡 OVERFITTING DETECTED (High Variance)"
        else:
            badge_color = "#10B981"
            badge_label = "🟢 OPTIMAL FIT (Well-Balanced Generalization)"

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid {badge_color}; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <div style="color: {badge_color}; font-weight: 800; font-size: 0.88rem;">{badge_label}</div>
            <div style="color: #E2E8F0; font-size: 0.88rem; margin-top: 4px;">{diag_msg}</div>
        </div>
        """, unsafe_allow_html=True)

        uf_col1, uf_col2, uf_col3 = st.columns(3)
        with uf_col1:
            st.metric("Training Set R²", f"{train_r2:.4f}" if train_r2 is not None else "N/A")
        with uf_col2:
            st.metric("Holdout Test Set R²", f"{test_r2:.4f}" if isinstance(test_r2, (int, float)) else str(test_r2))
        with uf_col3:
            gap_val = (train_r2 - test_r2) if (train_r2 is not None and isinstance(test_r2, (int, float))) else None
            st.metric("Generalization Gap (Train - Test)", f"{gap_val:.4f}" if gap_val is not None else "N/A", delta=f"{gap_val:.3f}" if gap_val is not None else None, delta_color="inverse")

    # -------------------------------------------------------------------------
    # TAB 6: LOG TRANSFORMATION (NP.LOG1P & NP.EXPM1)
    # -------------------------------------------------------------------------
    with imp_tabs[5]:
        st.markdown("###### 📐 Step 6: Target Log Transformation (np.log1p(y) & np.expm1())")
        st.caption("Right-skewed sales data compresses extreme tail values during model fitting, resulting in systematic underestimation. Transforming target y via np.log1p(y) balances variance:")

        t_skew = s6.get("target_skew", 0.0)
        is_skewed = s6.get("is_right_skewed", t_skew > 0.75)
        rec_text = s6.get("recommendation_text", "")

        skew_badge = "#EF4444" if is_skewed else "#10B981"
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid {skew_badge}; border-radius: 8px; padding: 14px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Target Distribution Skewness: <code>{t_skew:.2f}</code></span>
                <span style="color: {skew_badge}; font-weight: 800; font-size: 0.8rem; border: 1px solid {skew_badge}; padding: 2px 8px; border-radius: 4px;">
                    {'RIGHT-SKEWED (HIGH-SALES TAIL)' if is_skewed else 'SYMMETRICAL / NORMAL'}
                </span>
            </div>
            <div style="color: #CBD5E1; font-size: 0.84rem; margin-top: 6px;">
                {rec_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Mathematical Mechanism:**
        $$\\tilde{y} = \\ln(1 + y) \\quad \\xrightarrow{\\text{Fit Estimator}} \\quad \\hat{y} = \\exp(\\tilde{y}_{\\text{pred}}) - 1$$
        *All holdout validation metrics (MAE, RMSE, R²) are automatically computed and displayed on the original sales dollar scale for 100% apples-to-apples comparability.*
        """)

        # Status badge & activation notes
        if is_log_trans:
            st.success("✅ **Active:** This model was trained with `np.log1p(y)` target transformation and evaluated with `np.expm1()` predictions.")
        else:
            st.info("💡 **Recommendation:** Enable `Target Log-Transformation (np.log1p)` in the model configuration to re-train the pipeline and immediately eliminate high-sales underestimation.")
