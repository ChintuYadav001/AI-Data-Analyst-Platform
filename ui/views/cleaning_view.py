"""
Quantum Analyst - Autonomous AI Data Cleaning Engine View
Provides an intelligent, dataset-aware, explainable data cleaning dashboard
with 12 cleaning categories, confidence scoring, 1-Click Auto-Clean,
granular Review & Clean, before/after diff preview, post-cleaning validation,
and a non-destructive audit ledger with one-click rollback.
"""
from __future__ import annotations

import io
import re
import pandas as pd
import numpy as np
import streamlit as st

from core.models.cleaner_model import DataCleanerModel
from core.models.project_model import ProjectModel
from core.models.quality_model import DataQualityModel


def render_cleaning_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No active dataset detected. Please upload or load a sample dataset first.")
        if st.button("⬅️ Return to Dashboard / Ingestion", type="primary"):
            st.session_state["active_nav"] = "Dashboard"
            st.rerun()
        return

    df = st.session_state["df"]
    
    # Ensure raw_df is preserved for non-destructive operations and rollback
    if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
        st.session_state["raw_df"] = df.copy()
    raw_df = st.session_state["raw_df"]

    cleaning_log = st.session_state.get("cleaning_log", [])
    is_auth = bool(st.session_state.get("authenticated", False))
    project_id = st.session_state.get("current_project_id")
    project_name = st.session_state.get("current_project_name", "Active Dataset")

    # -------------------------------------------------------------------------
    # 1. RUN AI CLEANING SCAN
    # -------------------------------------------------------------------------
    with st.spinner("Analyzing dataset across 12 cleaning dimensions..."):
        scan_result = DataCleanerModel.scan_dataset(df)

    plan_items = scan_result.get("plan_items", [])
    summary = scan_result.get("summary", {})
    category_counts = scan_result.get("category_counts", {})

    total_issues = summary.get("total_issues", len(plan_items))
    safe_fixes = summary.get("safe_fixes", 0)
    review_required = summary.get("review_required", 0)
    info_items = summary.get("info_items", 0)

    # Calculate Data Quality metrics (Before vs Current)
    val_report = DataCleanerModel.validate_cleaning(raw_df, df)
    score_before = val_report.get("score_before", 75)
    score_current = val_report.get("score_after", 75)
    score_delta = score_current - score_before

    # -------------------------------------------------------------------------
    # 2. FUTURISTIC HEADER & TOP KPI SCORECARD
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(14, 20, 36, 0.95), rgba(7, 10, 20, 0.98)); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 16px; padding: 22px 28px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0, 240, 255, 0.08);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(0, 240, 255, 0.12); border: 1px solid rgba(0, 240, 255, 0.4); display: flex; align-items: center; justify-content: center; font-size: 1.6rem;">
                    🧹
                </div>
                <div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <h2 style="margin: 0; font-size: 1.55rem; color: #FFFFFF; font-weight: 800; letter-spacing: -0.02em;">
                            Autonomous AI Data Cleaning Engine
                        </h2>
                        <span style="background: rgba(0, 240, 255, 0.15); color: #00F0FF; border: 1px solid rgba(0, 240, 255, 0.35); font-size: 0.72rem; padding: 3px 9px; border-radius: 9999px; font-weight: 700; text-transform: uppercase;">
                            12-Category Scanner
                        </span>
                    </div>
                    <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.88rem;">
                        Intelligent, dataset-aware data hygiene scanner with explainable rule evaluation, non-destructive execution, and before/after verification.
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

    # Top KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        delta_str = f"+{score_delta} pts" if score_delta > 0 else ("0 pts" if score_delta == 0 else f"{score_delta} pts")
        st.metric("Quality Score", f"{score_current}/100", delta=delta_str)
    with m2:
        st.metric("Detected Issues", f"{total_issues}", delta=f"{safe_fixes} Auto-fixable", delta_color="normal")
    with m3:
        row_delta = f"{len(df) - len(raw_df):,} vs raw" if len(df) != len(raw_df) else None
        st.metric("Current Rows", f"{len(df):,}", delta=row_delta)
    with m4:
        raw_missing = int(raw_df.isna().sum().sum())
        cur_missing = int(df.isna().sum().sum())
        miss_delta = f"-{raw_missing - cur_missing:,}" if raw_missing > cur_missing else None
        st.metric("Missing Cells", f"{cur_missing:,}", delta=miss_delta, delta_color="inverse")
    with m5:
        st.metric("Executed Actions", f"{len(cleaning_log)} Operations", delta="Non-destructive")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. ACTION TOOLBAR
    # -------------------------------------------------------------------------
    tb_c1, tb_c2, tb_c3 = st.columns([2.5, 1.2, 1.2])
    with tb_c1:
        if safe_fixes > 0:
            if st.button(
                f"⚡ RUN 1-CLICK AI AUTO-CLEAN ({safe_fixes} Safe Fixes)",
                type="primary",
                width="stretch",
                help="Automatically applies all high-confidence (>95%), non-destructive fixes."
            ):
                with st.spinner("Executing high-confidence AI cleaning operations..."):
                    cleaned_df, new_logs = DataCleanerModel.auto_clean(df)
                    st.session_state["df"] = cleaned_df
                    st.session_state["cleaning_log"] = cleaning_log + new_logs
                    if project_id and is_auth:
                        p = ProjectModel(project_id, project_name)
                        p.save_cleaned_dataset(cleaned_df, st.session_state["cleaning_log"])
                    st.toast(f"AI Auto-Clean applied {len(new_logs)} safe transformations!", icon="✨")
                    st.rerun()
        else:
            st.button("✨ All Auto-Fix Items Resolved", disabled=True, width="stretch")

    with tb_c2:
        revert_disabled = (len(df) == len(raw_df) and int(df.isna().sum().sum()) == int(raw_df.isna().sum().sum()) and len(cleaning_log) == 0)
        if st.button("🔄 Rollback to Raw", type="secondary", width="stretch", disabled=revert_disabled, help="Rolls back all modifications and restores original uploaded dataset."):
            st.session_state["df"] = raw_df.copy()
            st.session_state["cleaning_log"] = []
            if project_id and is_auth:
                p = ProjectModel(project_id, project_name)
                p.save_cleaned_dataset(raw_df.copy(), [])
            st.toast("Restored dataset to original uploaded version.", icon="🔄")
            st.rerun()

    with tb_c3:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "💾 Export CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{project_name.lower().replace(' ', '_')}_cleaned.csv",
            mime="text/csv",
            width="stretch"
        )

    st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.08); margin: 18px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. TABBED WORKSPACE
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 AI Cleaning Plan & Review",
        "👁️ Before / After Diff Preview",
        "✅ Validation Report",
        "📜 Audit Ledger & Revert"
    ])

    # =========================================================================
    # TAB 1: AI CLEANING PLAN & REVIEW
    # =========================================================================
    with tab1:
        if not plan_items:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 32px; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">✨</div>
                <h3 style="color: #10B981; margin: 0 0 6px 0; font-weight: 800;">Flawless Data Hygiene</h3>
                <p style="color: #94A3B8; margin: 0; font-size: 0.95rem;">
                    No pending defects detected across all 12 cleaning categories. Your dataset is clean, consistent, and model-ready!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            f_col1, f_col2 = st.columns([2, 1])
            with f_col1:
                available_categories = ["All Categories"] + sorted(list({i.get("category", "General") for i in plan_items}))
                selected_category = st.selectbox("Filter by Category:", available_categories, key="cleaning_cat_filter")
            with f_col2:
                filter_action = st.selectbox("Action Type:", ["All Action Levels", "⚡ AUTO_FIX Only", "🔎 REVIEW Only", "ℹ️ INFO Only"], key="cleaning_action_filter")

            filtered_items = []
            for item in plan_items:
                cat_match = (selected_category == "All Categories" or item.get("category") == selected_category)
                act = item.get("action", "REVIEW")
                if filter_action == "⚡ AUTO_FIX Only" and act != "AUTO_FIX":
                    act_match = False
                elif filter_action == "🔎 REVIEW Only" and act != "REVIEW":
                    act_match = False
                elif filter_action == "ℹ️ INFO Only" and act not in ("KEEP", "DO_NOT_CHANGE"):
                    act_match = False
                else:
                    act_match = True

                if cat_match and act_match:
                    filtered_items.append(item)

            st.caption(f"Showing {len(filtered_items)} of {len(plan_items)} cleaning items:")

            items_to_apply = []
            for idx, item in enumerate(filtered_items):
                item_id = item.get("id", f"item_{idx}")
                col_name = item.get("column", "All")
                category = item.get("category", "Data Quality")
                count = item.get("count", 0)
                action_level = item.get("action", "REVIEW")
                confidence = int(item.get("confidence", 0.9) * 100)
                severity = item.get("severity", "MEDIUM")
                detection = item.get("detection_method", "Statistical Analysis")
                reason = item.get("reason", "")
                tech_details = item.get("technical_details", "")
                options = item.get("options", ["Apply Fix", "Keep"])
                default_opt = item.get("default_option", options[0])

                sev_color = "#EF4444" if severity == "HIGH" else ("#F59E0B" if severity == "MEDIUM" else ("#3B82F6" if severity == "LOW" else "#8B5CF6"))
                act_badge = "⚡ AUTO_FIX" if action_level == "AUTO_FIX" else ("🔎 REVIEW" if action_level == "REVIEW" else "ℹ️ KEEP")
                act_color = "#10B981" if action_level == "AUTO_FIX" else ("#F59E0B" if action_level == "REVIEW" else "#64748B")

                with st.container():
                    st.markdown(f"""
                    <div style="background: rgba(14, 20, 36, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid {sev_color}; border-radius: 10px; padding: 16px 18px; margin-bottom: 12px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                <span style="background: rgba(255, 255, 255, 0.06); color: #CBD5E1; font-size: 0.76rem; padding: 3px 8px; border-radius: 6px; font-weight: 700;">
                                    📁 {category}
                                </span>
                                <span style="background: rgba(0, 240, 255, 0.1); color: #00F0FF; font-size: 0.76rem; padding: 3px 8px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-weight: 600;">
                                    Column: {col_name}
                                </span>
                                <span style="background: {sev_color}22; color: {sev_color}; border: 1px solid {sev_color}44; font-size: 0.72rem; padding: 2px 7px; border-radius: 4px; font-weight: 700;">
                                    {severity}
                                </span>
                                <span style="background: {act_color}22; color: {act_color}; border: 1px solid {act_color}44; font-size: 0.72rem; padding: 2px 7px; border-radius: 4px; font-weight: 700;">
                                    {act_badge} ({confidence}%)
                                </span>
                            </div>
                            <div style="font-size: 0.78rem; color: #94A3B8;">
                                <span style="color: #F1F5F9; font-weight: 700;">{count:,}</span> affected
                            </div>
                        </div>
                        <div style="font-size: 0.9rem; color: #E2E8F0; margin-bottom: 6px; font-weight: 500;">
                            {reason}
                        </div>
                        <div style="font-size: 0.76rem; color: #64748B;">
                            Detection Method: <span style="color: #94A3B8;">{detection}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    card_c1, card_c2 = st.columns([1, 2])
                    with card_c1:
                        is_checked = (action_level == "AUTO_FIX")
                        include_item = st.checkbox(
                            "Include in execution plan",
                            value=is_checked,
                            key=f"chk_{item_id}"
                        )
                    with card_c2:
                        default_index = options.index(default_opt) if default_opt in options else 0
                        chosen_opt = st.selectbox(
                            f"Selected Action for {col_name}:",
                            options=options,
                            index=default_index,
                            key=f"opt_{item_id}"
                        )

                    if tech_details:
                        with st.expander("Diagnostics & Details", expanded=False):
                            st.markdown(f"<code style='color: #00F0FF; font-size: 0.8rem;'>{tech_details}</code>", unsafe_allow_html=True)

                    if include_item:
                        item_copy = dict(item)
                        item_copy["chosen_option"] = chosen_opt
                        items_to_apply.append(item_copy)

                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            btn_c1, btn_c2 = st.columns([2, 1])
            with btn_c1:
                if st.button(
                    f"🚀 Apply {len(items_to_apply)} Selected AI Actions",
                    type="primary",
                    width="stretch",
                    disabled=(len(items_to_apply) == 0)
                ):
                    with st.spinner("Applying selected AI cleaning operations..."):
                        cleaned_df, new_logs = DataCleanerModel.execute_plan(df, items_to_apply)
                        st.session_state["df"] = cleaned_df
                        st.session_state["cleaning_log"] = cleaning_log + new_logs
                        if project_id and is_auth:
                            p = ProjectModel(project_id, project_name)
                            p.save_cleaned_dataset(cleaned_df, st.session_state["cleaning_log"])
                        st.toast(f"Successfully executed {len(new_logs)} transformations!", icon="🚀")
                        st.rerun()
            with btn_c2:
                st.caption("All executions are non-destructive and tracked in the audit ledger.")

    # =========================================================================
    # TAB 2: BEFORE / AFTER DIFF PREVIEW
    # =========================================================================
    with tab2:
        st.markdown("### 👁️ Before / After Cell-Level Diff Comparison")
        st.caption("Inspect altered rows side-by-side to verify exact value transformations.")

        diff_res = DataCleanerModel.generate_sample_diff(raw_df, df, max_rows=12)
        total_changed = diff_res.get("total_changed_rows", 0)
        sample_diffs = diff_res.get("sample_diffs", [])

        diff_m1, diff_m2, diff_m3 = st.columns(3)
        with diff_m1:
            st.metric("Modified Records", f"{total_changed:,} rows")
        with diff_m2:
            st.metric("Missing Reduction", f"{max(0, int(raw_df.isna().sum().sum()) - int(df.isna().sum().sum())):,} cells")
        with diff_m3:
            st.metric("Quality Uplift", f"+{score_delta} points" if score_delta >= 0 else f"{score_delta} points")

        if not sample_diffs:
            if len(cleaning_log) == 0:
                st.info("ℹ️ No transformations applied yet. Run 1-Click Auto-Clean or apply selected actions to see before/after diffs.")
            else:
                st.success("✅ Transformations completed with no remaining cell variations against the current dataset.")
        else:
            st.markdown(f"#### 🔍 Sample of Changed Rows ({len(sample_diffs)} displayed)")
            
            diff_records = []
            for item in sample_diffs:
                row_idx = item.get("row_index")
                changes = item.get("changes", {})
                for col_name, chg in changes.items():
                    diff_records.append({
                        "Row Index": row_idx,
                        "Column": col_name,
                        "Original (Raw) Value": chg.get("before", "NaN"),
                        "Cleaned Value": chg.get("after", "NaN"),
                        "Status": "MODIFIED"
                    })

            diff_df = pd.DataFrame(diff_records)
            st.dataframe(diff_df, width="stretch")

            st.markdown("#### 📊 Side-by-Side Dataset Comparison")
            side_c1, side_c2 = st.columns(2)
            with side_c1:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #EF4444; margin-bottom: 6px;'>RAW ORIGINAL DATASET (First 5 Rows)</div>", unsafe_allow_html=True)
                st.dataframe(raw_df.head(5), width="stretch")
            with side_c2:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #10B981; margin-bottom: 6px;'>CURRENT CLEANED DATASET (First 5 Rows)</div>", unsafe_allow_html=True)
                st.dataframe(df.head(5), width="stretch")

    # =========================================================================
    # TAB 3: VALIDATION REPORT
    # =========================================================================
    with tab3:
        st.markdown("### ✅ Post-Cleaning Hygiene Verification Report")
        st.caption("Rigorous statistical recalculation verifying that data cleaning truly resolved anomalies.")

        status_headline = val_report.get("status_headline", "CLEANING VERIFIED")
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.3rem;">🛡️</span>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 700; color: #10B981; letter-spacing: 0.05em; text-transform: uppercase;">
                        HYGIENE STATUS
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #FFFFFF;">
                        {status_headline}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        v_cols = st.columns(2)
        with v_cols[0]:
            st.markdown(f"""
            <div style="background: rgba(14, 20, 36, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">
                    Raw Upload Quality
                </div>
                <div style="font-size: 2.5rem; font-weight: 900; color: #EF4444;">
                    {score_before}
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem;">
                    Grade: <b style="color: #FFFFFF;">{val_report.get('grade_before', 'Fair')}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with v_cols[1]:
            delta_label = f"+{score_delta} pts" if score_delta >= 0 else f"{score_delta} pts"
            st.markdown(f"""
            <div style="background: rgba(14, 20, 36, 0.7); border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 0.78rem; color: #00F0FF; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">
                    Current Quality ({delta_label})
                </div>
                <div style="font-size: 2.5rem; font-weight: 900; color: #10B981;">
                    {score_current}
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem;">
                    Grade: <b style="color: #FFFFFF;">{val_report.get('grade_after', 'Excellent')}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Core Data Hygiene Verification Ledger")
        
        validations = val_report.get("validations", [])
        if validations:
            v_df = pd.DataFrame(validations)
            st.dataframe(v_df, width="stretch")

        remaining = val_report.get("remaining_issues", [])
        if remaining:
            st.markdown("#### ⚠️ Remaining Items / Manual Review Advice")
            for rem in remaining:
                st.markdown(f"- **{rem.get('dimension', 'Quality')}**: {rem.get('description', '')} *(Score: {rem.get('score', 0)}%)*")

    # =========================================================================
    # TAB 4: AUDIT LEDGER & REVERT
    # =========================================================================
    with tab4:
        st.markdown("### 📜 Non-Destructive Audit Ledger")
        st.caption("Immutable, reproducible execution trail recording every transformation applied to your data.")

        if cleaning_log:
            log_display = []
            for entry in cleaning_log:
                log_display.append({
                    "Timestamp": entry.get("timestamp", "-"),
                    "Category": entry.get("category", entry.get("issue", "General")),
                    "Operation": entry.get("operation", entry.get("action", "Processed")),
                    "Target Column": entry.get("column", "All"),
                    "Rows Affected": entry.get("rows_affected", entry.get("count", 0)),
                    "Status": entry.get("status", "COMPLETED"),
                    "Reason": entry.get("reason", "-")
                })
            st.dataframe(pd.DataFrame(log_display), width="stretch")
        else:
            st.info("No cleaning operations executed yet in this session.")

        st.markdown("---")
        st.markdown("#### 🔄 Non-Destructive Data Protection & Rollback")
        st.write("Your original raw dataset is permanently preserved in session storage. You can rollback all changes or export clean datasets at any point.")

        roll_c1, roll_c2 = st.columns(2)
        with roll_c1:
            if st.button("⚠️ Rollback to Raw Dataset", type="secondary", width="stretch"):
                st.session_state["df"] = raw_df.copy()
                st.session_state["cleaning_log"] = []
                if project_id and is_auth:
                    p = ProjectModel(project_id, project_name)
                    p.save_cleaned_dataset(raw_df.copy(), [])
                st.toast("Restored dataset to original uploaded version.", icon="🔄")
                st.rerun()
        with roll_c2:
            st.download_button(
                "📥 Export Full Clean Dataset (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"{project_name.lower().replace(' ', '_')}_cleaned.csv",
                mime="text/csv",
                width="stretch"
            )
