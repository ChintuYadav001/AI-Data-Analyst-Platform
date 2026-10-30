import re
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.models import DataProfilerModel
from core.eda.profiler import EDAProfiler
from ui.components import render_kpi_card

# Neon SaaS Theme Palette
NEON_CYAN = "#00F0FF"
NEON_GREEN = "#10B981"
NEON_VIOLET = "#8B5CF6"
NEON_AMBER = "#F59E0B"
NEON_RED = "#F43F5E"
DARK_BG = "#0B0F19"
CARD_BG = "#131B2E"
BORDER_CLR = "#1E293B"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    margin=dict(l=25, r=25, t=40, b=35),
    font=dict(family="JetBrains Mono, sans-serif", size=11, color="#94A3B8"),
    xaxis=dict(showgrid=True, gridcolor="#172033", linecolor="#1E293B"),
    yaxis=dict(showgrid=True, gridcolor="#172033", linecolor="#1E293B"),
)


def _get_health_badge(score: float) -> str:
    if score >= 90:
        return f'<span style="background: rgba(16, 185, 129, 0.15); color: {NEON_GREEN}; font-weight: 700; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem;">🟢 EXCELLENT ({score}%)</span>'
    elif score >= 75:
        return f'<span style="background: rgba(0, 240, 255, 0.15); color: {NEON_CYAN}; font-weight: 700; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem;">🔵 GOOD ({score}%)</span>'
    elif score >= 50:
        return f'<span style="background: rgba(245, 158, 11, 0.15); color: {NEON_AMBER}; font-weight: 700; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem;">🟠 FAIR ({score}%)</span>'
    else:
        return f'<span style="background: rgba(244, 63, 94, 0.15); color: {NEON_RED}; font-weight: 700; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem;">🔴 ATTENTION ({score}%)</span>'


def render_profile_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset or load a sample dataset first.")
        return

    df: pd.DataFrame = st.session_state["df"]
    if df.empty:
        st.warning("⚠️ The loaded dataset is empty (0 rows).")
        return

    # ═══════════════════════════════════════════════════════════════════
    # Header & Executive Title
    # ═══════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <div>
            <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                📋 Automatic Data Profiling & Dataset Inspector
            </h1>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #94A3B8;">
                Comprehensive schema intelligence, interactive raw data exploration, statistical moments, and data health audit.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Compute Profile
    try:
        profile = DataProfilerModel.profile_dataset(df)
    except Exception as e:
        st.error(f"Error computing data profile: {e}")
        return

    ds = profile.get("dataset_info", {})
    health_score = ds.get("health_score", 95.0)

    # ═══════════════════════════════════════════════════════════════════
    # Top KPI Metrics Strip (6 Columns)
    # ═══════════════════════════════════════════════════════════════════
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_kpi_card("Total Rows", f"{ds.get('rows', len(df)):,}", "Records", "🔢", status="neutral", badge="RECORDS")
    with c2:
        render_kpi_card("Total Columns", f"{ds.get('columns', len(df.columns))}", "Features", "🗂️", status="neutral", badge="FEATURES")
    with c3:
        render_kpi_card("Memory Size", f"{ds.get('memory_mb', 0.0)} MB", "Deep footprint", "💾", status="neutral", badge="MEMORY")
    with c4:
        dup_cnt = ds.get('duplicate_rows', 0)
        dup_pct = ds.get('duplicate_pct', 0.0)
        render_kpi_card("Duplicate Rows", f"{dup_cnt:,}", f"{dup_pct}% of dataset", "🔁", status="positive" if dup_cnt == 0 else "negative", badge="DUPLICATES")
    with c5:
        miss_cnt = ds.get('total_missing', 0)
        miss_pct = ds.get('missing_pct', 0.0)
        render_kpi_card("Missing Cells", f"{miss_cnt:,}", f"{miss_pct}% overall", "❓", status="positive" if miss_cnt == 0 else "negative", badge="SPARSITY")
    with c6:
        h_status = "positive" if health_score >= 80 else ("neutral" if health_score >= 60 else "negative")
        render_kpi_card("Data Health", f"{health_score}%", "Quality score", "🛡️", status=h_status, badge="HEALTH")

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # Multi-Dimensional Navigation Tabs
    # ═══════════════════════════════════════════════════════════════════
    tabs = st.tabs([
        "📂 Uploaded Data Explorer",
        "📊 Column Classification & Statistics",
        "📈 Interactive Distributions",
        "🔗 Correlation Matrix",
        "🚨 Data Quality & Anomalies"
    ])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: Uploaded Data Explorer ("SHOW DATA THAT WE UPLOAD")
    # ═══════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0,240,255,0.06), rgba(139,92,246,0.06));
                    border: 1px solid rgba(0,240,255,0.2); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: {NEON_CYAN}; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em;">
                        // INTERACTIVE DATASET VIEWER & RAW DATA EXPLORER
                    </span>
                    <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">
                        Inspect, filter, slice, and verify the records from your uploaded dataset.
                    </div>
                </div>
                <div>
                    {_get_health_badge(health_score)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Explorer Controls Bar
        ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.5, 1.2, 2.5, 1.2])

        with ctrl_c1:
            slice_mode = st.selectbox(
                "Slice Mode:",
                ["First Rows (Head)", "Last Rows (Tail)", "Random Sample", "All Records"],
                index=0,
                key="data_slice_mode"
            )

        with ctrl_c2:
            row_limit_options = [10, 25, 50, 100, 250, 500, 1000]
            row_limit = st.selectbox(
                "Rows to Display:",
                row_limit_options,
                index=1,
                key="data_row_limit",
                disabled=(slice_mode == "All Records")
            )

        with ctrl_c3:
            all_cols = list(df.columns)
            selected_cols = st.multiselect(
                "Filter Columns:",
                options=all_cols,
                default=all_cols,
                key="data_selected_cols",
                help="Select specific columns to inspect or leave all selected."
            )

        with ctrl_c4:
            search_query = st.text_input("🔍 Quick Search:", key="data_search_query", placeholder="Search text...")

        # Slice preparation
        display_cols = selected_cols if selected_cols else all_cols
        slice_df = df[display_cols]

        # Apply search if provided
        if search_query.strip():
            query_str = search_query.strip().lower()
            mask = slice_df.astype(str).apply(lambda row: row.str.lower().str.contains(query_str, regex=False).any(), axis=1)
            filtered_df = slice_df[mask]
        else:
            filtered_df = slice_df

        # Apply row slice
        total_matching = len(filtered_df)
        if slice_mode == "First Rows (Head)":
            view_df = filtered_df.head(row_limit)
        elif slice_mode == "Last Rows (Tail)":
            view_df = filtered_df.tail(row_limit)
        elif slice_mode == "Random Sample":
            sample_n = min(row_limit, total_matching)
            view_df = filtered_df.sample(sample_n, random_state=42) if total_matching > 0 else filtered_df
        else:
            view_df = filtered_df.head(5000)  # capped for browser rendering safety

        # Render Dataframe
        st.dataframe(view_df, width="stretch", height=420)

        # Slice summary & CSV export
        exp_c1, exp_c2 = st.columns([3, 1])
        with exp_c1:
            st.caption(f"Showing **{len(view_df):,}** of **{total_matching:,}** matching records (Dataset total: **{len(df):,}** rows × **{len(display_cols)}** columns).")
        with exp_c2:
            st.download_button(
                "📥 Download Current View (CSV)",
                data=view_df.to_csv(index=False),
                file_name="dataset_export.csv",
                mime="text/csv",
                width="stretch",
                key="download_view_csv"
            )

        # Quick Schema Dictionary
        with st.expander("🗂️ Dataset Schema & Field Types Quick Reference", expanded=False):
            schema_data = []
            for c in display_cols:
                s = df[c]
                non_null = int(s.count())
                n_uniq = int(s.nunique(dropna=True))
                first_valid = s.dropna().iloc[0] if non_null > 0 else "—"
                schema_data.append({
                    "Column Name": str(c),
                    "Data Type": str(s.dtype),
                    "Non-Null Count": f"{non_null:,} ({round(non_null/len(df)*100, 1)}%)",
                    "Unique Values": f"{n_uniq:,}",
                    "Sample Value": str(first_valid)[:50]
                })
            st.dataframe(pd.DataFrame(schema_data), width="stretch", hide_index=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: Column Classification & Statistical Metrics
    # ═══════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown(f"""
        <div style="font-size: 0.75rem; font-weight: 700; color: {NEON_VIOLET}; letter-spacing: 0.05em; margin-bottom: 8px;">
            // COLUMN CLASSIFICATION & DESCRIPTIVE STATISTICS
        </div>
        """, unsafe_allow_html=True)

        cols_data = profile.get("columns", [])
        col_df = pd.DataFrame(cols_data)

        if not col_df.empty:
            filter_c1, filter_c2 = st.columns([2, 2])
            with filter_c1:
                semantic_types = ["All"] + sorted(list(col_df["semantic_type"].dropna().unique()))
                chosen_type = st.selectbox("Filter by Semantic Type:", semantic_types, key="stat_type_filter")
            with filter_c2:
                col_search = st.text_input("Filter by Column Name:", key="stat_col_search", placeholder="Column name...")

            filtered_stat_df = col_df.copy()
            if chosen_type != "All":
                filtered_stat_df = filtered_stat_df[filtered_stat_df["semantic_type"] == chosen_type]
            if col_search.strip():
                filtered_stat_df = filtered_stat_df[filtered_stat_df["name"].str.lower().str.contains(col_search.strip().lower())]

            # Primary display columns
            base_cols = ["name", "semantic_type", "dtype", "null_count", "null_pct", "unique_count", "unique_pct", "mode"]
            numeric_metrics = ["mean", "std", "min", "q25", "median", "q75", "max", "iqr", "skewness", "outliers", "outlier_pct"]

            active_stat_cols = [c for c in base_cols if c in filtered_stat_df.columns]
            for nm in numeric_metrics:
                if nm in filtered_stat_df.columns:
                    active_stat_cols.append(nm)

            # Prettify column headers
            column_rename_map = {
                "name": "Column Name",
                "semantic_type": "Semantic Type",
                "dtype": "Dtype",
                "null_count": "Missing",
                "null_pct": "Missing %",
                "unique_count": "Distinct",
                "unique_pct": "Distinct %",
                "mode": "Mode / Top",
                "mean": "Mean",
                "std": "Std Dev",
                "min": "Min",
                "q25": "25% (Q1)",
                "median": "Median",
                "q75": "75% (Q3)",
                "max": "Max",
                "iqr": "IQR",
                "skewness": "Skewness",
                "outliers": "Outliers (IQR)",
                "outlier_pct": "Outlier %"
            }

            display_table = filtered_stat_df[active_stat_cols].rename(columns=column_rename_map)

            st.dataframe(
                display_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Missing %": st.column_config.ProgressColumn(
                        "Missing %",
                        help="Percentage of null values",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "Distinct %": st.column_config.ProgressColumn(
                        "Distinct %",
                        help="Cardinality ratio",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100
                    ),
                }
            )

            # Export Dictionary
            st.download_button(
                "📥 Export Full Data Dictionary (CSV)",
                data=display_table.to_csv(index=False),
                file_name="data_dictionary.csv",
                mime="text/csv",
                key="download_dict_csv"
            )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: Interactive Distributions & Deep Inspect
    # ═══════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown(f"""
        <div style="font-size: 0.75rem; font-weight: 700; color: {NEON_CYAN}; letter-spacing: 0.05em; margin-bottom: 8px;">
            // INTERACTIVE DISTRIBUTION VISUALIZER
        </div>
        """, unsafe_allow_html=True)

        chosen_col = st.selectbox(
            "Select column to deep inspect:",
            options=list(df.columns),
            key="profile_chosen_col",
            help="Choose any column to generate interactive Plotly distribution charts."
        )

        if chosen_col:
            col_series = df[chosen_col]
            c_meta = next((c for c in cols_data if c["name"] == chosen_col), {})
            sem_type = c_meta.get("semantic_type", "Unknown")

            # Column Metadata Pill Strip
            st.markdown(f"""
            <div style="background: rgba(19, 27, 46, 0.7); border: 1px solid {BORDER_CLR}; border-radius: 10px; padding: 10px 16px; margin: 8px 0 16px 0; display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
                <div><span style="color: #94A3B8; font-size: 0.78rem;">FIELD:</span> <strong style="color: #FFFFFF; font-size: 0.95rem;">{chosen_col}</strong></div>
                <div><span style="color: #94A3B8; font-size: 0.78rem;">TYPE:</span> <strong style="color: {NEON_CYAN}; font-size: 0.85rem;">{sem_type}</strong></div>
                <div><span style="color: #94A3B8; font-size: 0.78rem;">STORAGE:</span> <code style="color: #A78BFA; font-size: 0.82rem;">{col_series.dtype}</code></div>
                <div><span style="color: #94A3B8; font-size: 0.78rem;">NON-NULL:</span> <strong style="color: #FFFFFF; font-size: 0.85rem;">{col_series.notna().sum():,} / {len(df):,} ({round(col_series.notna().mean()*100, 1)}%)</strong></div>
                <div><span style="color: #94A3B8; font-size: 0.78rem;">DISTINCT:</span> <strong style="color: #FFFFFF; font-size: 0.85rem;">{col_series.nunique():,}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            # Check if Numeric
            if pd.api.types.is_numeric_dtype(col_series):
                valid_num = col_series.dropna()
                if not valid_num.empty:
                    # 4 Metric Cards
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Mean", f"{valid_num.mean():,.2f}")
                    with m2:
                        st.metric("Median (50%)", f"{valid_num.median():,.2f}")
                    with m3:
                        st.metric("Min / Max", f"{valid_num.min():,.1f} / {valid_num.max():,.1f}")
                    with m4:
                        outlier_c = c_meta.get("outliers", 0)
                        outlier_p = c_meta.get("outlier_pct", 0.0)
                        st.metric("Outliers (IQR)", f"{outlier_c:,}", f"{outlier_p}%")

                    # Plotly Charts Side-by-Side
                    ch_c1, ch_c2 = st.columns(2)
                    with ch_c1:
                        fig_hist = px.histogram(
                            valid_num,
                            x=chosen_col,
                            marginal="box",
                            color_discrete_sequence=[NEON_CYAN],
                            title=f"<span style='color:{NEON_CYAN}'>Distribution & Box Plot: {chosen_col}</span>",
                            nbins=40
                        )
                        fig_hist.update_layout(**PLOTLY_LAYOUT)
                        st.plotly_chart(fig_hist, width="stretch", key=f"hist_{chosen_col}")

                    with ch_c2:
                        fig_box = px.box(
                            valid_num,
                            y=chosen_col,
                            points="outliers",
                            color_discrete_sequence=[NEON_VIOLET],
                            title=f"<span style='color:{NEON_VIOLET}'>Outlier Detection & Quartiles</span>"
                        )
                        fig_box.update_layout(**PLOTLY_LAYOUT)
                        st.plotly_chart(fig_box, width="stretch", key=f"box_{chosen_col}")

                    # Percentiles Table
                    with st.expander("📊 Detailed Percentile Spectrum"):
                        percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
                        p_vals = valid_num.quantile(percentiles).to_dict()
                        p_df = pd.DataFrame([{"Percentile": f"{int(k*100)}th", "Value": f"{v:,.4f}"} for k, v in p_vals.items()])
                        st.dataframe(p_df, width="stretch", hide_index=True)
                else:
                    st.info("Column contains only NaN or empty values.")

            elif pd.api.types.is_datetime64_any_dtype(col_series) or sem_type == "Date/time":
                # Datetime Distribution
                try:
                    dt_series = pd.to_datetime(col_series.dropna(), errors="coerce").dropna()
                    if not dt_series.empty:
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Earliest Date", str(dt_series.min().date()))
                        with m2:
                            st.metric("Latest Date", str(dt_series.max().date()))
                        with m3:
                            date_range_days = (dt_series.max() - dt_series.min()).days
                            st.metric("Total Span", f"{date_range_days:,} days")

                        # Timeline distribution (resampled by month)
                        timeline_df = dt_series.dt.to_period("M").dt.to_timestamp().value_counts().reset_index()
                        timeline_df.columns = ["Period", "Record Count"]
                        timeline_df = timeline_df.sort_values("Period")

                        fig_dt = px.line(
                            timeline_df,
                            x="Period",
                            y="Record Count",
                            markers=True,
                            color_discrete_sequence=[NEON_GREEN],
                            title=f"<span style='color:{NEON_GREEN}'>Temporal Trend: {chosen_col}</span>"
                        )
                        fig_dt.update_layout(**PLOTLY_LAYOUT)
                        st.plotly_chart(fig_dt, width="stretch", key=f"dt_{chosen_col}")
                except Exception as e:
                    st.info(f"Could not parse datetime distribution: {e}")

            else:
                # Categorical / Text Distribution
                valid_cat = col_series.dropna().astype(str)
                vc = valid_cat.value_counts().reset_index()
                vc.columns = ["Category", "Count"]
                vc["Share %"] = (vc["Count"] / len(valid_cat) * 100).round(2)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Distinct Categories", f"{len(vc):,}")
                with m2:
                    top_mode = vc.iloc[0]["Category"] if not vc.empty else "—"
                    st.metric("Dominant Category", str(top_mode)[:25])
                with m3:
                    top_share = vc.iloc[0]["Share %"] if not vc.empty else 0.0
                    st.metric("Dominant Share", f"{top_share}%")

                ch_c1, ch_c2 = st.columns([1.5, 1])
                with ch_c1:
                    top_15 = vc.head(15)
                    fig_bar = px.bar(
                        top_15,
                        x="Count",
                        y="Category",
                        orientation="h",
                        color="Count",
                        color_continuous_scale=[[0, NEON_VIOLET], [1, NEON_CYAN]],
                        title=f"<span style='color:{NEON_CYAN}'>Top 15 Categories by Frequency</span>",
                        text_auto=True
                    )
                    fig_bar.update_layout(yaxis=dict(autorange="reversed"), **PLOTLY_LAYOUT)
                    st.plotly_chart(fig_bar, width="stretch", key=f"cat_{chosen_col}")

                with ch_c2:
                    top_8 = vc.head(8)
                    other_sum = vc.iloc[8:]["Count"].sum() if len(vc) > 8 else 0
                    if other_sum > 0:
                        donut_df = pd.concat([top_8[["Category", "Count"]], pd.DataFrame([{"Category": "Other Categories", "Count": other_sum}])], ignore_index=True)
                    else:
                        donut_df = top_8

                    fig_donut = px.pie(
                        donut_df,
                        names="Category",
                        values="Count",
                        hole=0.45,
                        title=f"<span style='color:{NEON_CYAN}'>Category Share</span>",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig_donut.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig_donut, width="stretch", key=f"donut_{chosen_col}")

                with st.expander("📋 Full Value Counts Table", expanded=False):
                    st.dataframe(vc, width="stretch", hide_index=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 4: Correlation Matrix & Feature Relationships
    # ═══════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown(f"""
        <div style="font-size: 0.75rem; font-weight: 700; color: {NEON_CYAN}; letter-spacing: 0.05em; margin-bottom: 8px;">
            // MULTI-FEATURE CORRELATION MATRIX
        </div>
        """, unsafe_allow_html=True)

        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr_ctrl1, corr_ctrl2 = st.columns([1, 2])
            with corr_ctrl1:
                corr_method = st.selectbox("Correlation Method:", ["Pearson (Linear)", "Spearman (Rank-order)"], key="corr_method")
                method_key = "pearson" if "Pearson" in corr_method else "spearman"

            with corr_ctrl2:
                selected_num_cols = st.multiselect(
                    "Select Numeric Columns:",
                    options=list(numeric_df.columns),
                    default=list(numeric_df.columns)[:10],
                    key="corr_num_cols"
                )

            if len(selected_num_cols) >= 2:
                corr_sub_df = numeric_df[selected_num_cols].corr(method=method_key).round(3)

                fig_corr = px.imshow(
                    corr_sub_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1.0,
                    zmax=1.0,
                    title=f"<span style='color:{NEON_CYAN}'>{corr_method} Correlation Matrix</span>"
                )
                fig_corr.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_corr, width="stretch", key="corr_matrix_heatmap")

                # Top Correlated Feature Pairs
                st.markdown("#### 🔗 Notable Feature Correlations")
                pairs = []
                corr_cols = corr_sub_df.columns
                for i in range(len(corr_cols)):
                    for j in range(i + 1, len(corr_cols)):
                        c1_name = corr_cols[i]
                        c2_name = corr_cols[j]
                        val = corr_sub_df.loc[c1_name, c2_name]
                        if not np.isnan(val):
                            pairs.append({"Feature 1": c1_name, "Feature 2": c2_name, "Correlation": val, "Strength": abs(val)})

                if pairs:
                    pairs_df = pd.DataFrame(pairs).sort_values("Strength", ascending=False)
                    top_pairs = pairs_df.head(6)

                    cp1, cp2 = st.columns(2)
                    with cp1:
                        st.markdown("**Top Positive Correlations:**")
                        pos_df = top_pairs[top_pairs["Correlation"] > 0]
                        if not pos_df.empty:
                            st.dataframe(pos_df[["Feature 1", "Feature 2", "Correlation"]], width="stretch", hide_index=True)
                        else:
                            st.caption("No strong positive correlations found.")
                    with cp2:
                        st.markdown("**Top Negative Correlations:**")
                        neg_df = top_pairs[top_pairs["Correlation"] < 0]
                        if not neg_df.empty:
                            st.dataframe(neg_df[["Feature 1", "Feature 2", "Correlation"]], width="stretch", hide_index=True)
                        else:
                            st.caption("No strong negative correlations found.")
            else:
                st.info("Please select at least 2 numeric columns to compute the correlation matrix.")
        else:
            st.info("The dataset contains fewer than 2 numeric columns, so a correlation matrix cannot be computed.")

    # ═══════════════════════════════════════════════════════════════════
    # TAB 5: Data Quality, Anomalies & PII Warnings
    # ═══════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown(f"""
        <div style="font-size: 0.75rem; font-weight: 700; color: {NEON_AMBER}; letter-spacing: 0.05em; margin-bottom: 8px;">
            // DATA QUALITY, INTEGRITY & SECURITY AUDIT
        </div>
        """, unsafe_allow_html=True)

        # 1. Missing Values Overview
        null_counts = df.isna().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        if not cols_with_nulls.empty:
            st.markdown("#### ❓ Missing Values Distribution")
            null_pcts = (cols_with_nulls / len(df) * 100).round(2).reset_index()
            null_pcts.columns = ["Column", "Missing %"]
            null_pcts["Missing Count"] = cols_with_nulls.values
            null_pcts = null_pcts.sort_values("Missing %", ascending=False)

            fig_null = px.bar(
                null_pcts,
                x="Missing %",
                y="Column",
                orientation="h",
                color="Missing %",
                color_continuous_scale=[[0, NEON_AMBER], [1, NEON_RED]],
                title=f"<span style='color:{NEON_AMBER}'>Columns with Missing Values</span>",
                text_auto="%.1f%%"
            )
            fig_null.update_layout(yaxis=dict(autorange="reversed"), **PLOTLY_LAYOUT)
            st.plotly_chart(fig_null, width="stretch", key="null_summary_bar")
        else:
            st.success("✅ **Zero Missing Values**: The dataset is 100% complete across all columns.")

        # 2. Suspicious Columns Flags
        suspicious = profile.get("suspicious_columns", [])
        if suspicious:
            st.markdown("#### ⚠️ Detected Column Anomalies")
            for s in suspicious:
                st.warning(f"**Column `{s['column']}`**: {s['reason']}")
        else:
            st.info("✅ No constant or extreme-missingness column anomalies detected.")

        # 3. PII (Personally Identifiable Information) Audit
        st.markdown("#### 🔒 PII (Personally Identifiable Information) Screening")
        pii_keywords = ["email", "phone", "ssn", "social_security", "credit_card", "card_number", "passport", "license", "address", "zipcode", "postal", "dob", "birth"]
        detected_pii = []
        for c in df.columns:
            c_low = str(c).lower()
            if any(k in c_low for k in pii_keywords):
                detected_pii.append(c)

        if detected_pii:
            st.error(f"🚨 **Potential PII Columns Detected**: `{', '.join(detected_pii)}`. Ensure data compliance (GDPR/HIPAA/CCPA) prior to public reporting.")
        else:
            st.success("✅ No obvious PII column headers (email, SSN, phone, address) detected.")
