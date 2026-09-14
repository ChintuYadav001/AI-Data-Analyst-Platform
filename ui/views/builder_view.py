import streamlit as st
import pandas as pd
from core.models import DashboardModel, ChartModel
from ui.components import render_kpi_card

def render_builder_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Please upload a dataset or load a sample dataset first.")
        return

    df = st.session_state["df"]
    st.title("🛠️ Custom Dashboard Builder")
    st.markdown("Customize KPIs, configure interactive filter dimensions, and analyze cohort subsets dynamically.")

    # 1. Automatic Dynamic Filters
    dimensions = DashboardModel.detect_filter_dimensions(df)
    active_filters = {}

    if dimensions:
        st.markdown("### 🎛️ Dynamic Cross-Filter Dimensions")
        cols = st.columns(len(dimensions))
        for idx, dim in enumerate(dimensions):
            with cols[idx]:
                selected = st.multiselect(
                    f"{dim['column'].replace('_', ' ').title()}:",
                    options=dim["options"],
                    default=[],
                    key=f"builder_filter_{dim['column']}",
                    help=dim.get("rationale", "")
                )
                if selected:
                    active_filters[dim["column"]] = selected

    # Apply filters dynamically
    filtered_df = DashboardModel.apply_filters(df, active_filters)
    if len(filtered_df) < len(df):
        st.caption(f"Showing **{len(filtered_df):,}** of {len(df):,} filtered records ({round((len(filtered_df)/len(df))*100, 1)}% of population).")

    st.markdown("---")

    # 2. Dynamic KPI metrics
    kpis = DashboardModel.compute_kpis(filtered_df)
    cols = st.columns(len(kpis))
    for idx, kpi in enumerate(kpis):
        with cols[idx]:
            render_kpi_card(
                title=kpi["label"],
                value=kpi["value"],
                subtext=kpi.get("subtext", ""),
                icon=kpi.get("icon", "📊"),
                status=kpi.get("status", "neutral"),
                badge=kpi.get("badge", "KPI")
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Chart Customization & Selector
    all_recs = ChartModel.recommend_charts(filtered_df)
    
    with st.expander("⚙️ Dashboard Visualization Configuration", expanded=False):
        chart_options = [r["title"] for r in all_recs]
        selected_titles = st.multiselect(
            "Select charts to display on the dashboard:",
            options=chart_options,
            default=chart_options[:4]
        )
        if st.button("💾 Save Dashboard Layout"):
            st.success("Dashboard configuration saved successfully!")

    # Render selected charts
    displayed_recs = [r for r in all_recs if r["title"] in selected_titles] if 'selected_titles' in locals() and selected_titles else all_recs[:4]

    c1, c2 = st.columns(2)
    for i, rec in enumerate(displayed_recs):
        col = c1 if i % 2 == 0 else c2
        with col:
            fig = ChartModel.create_plotly_figure(filtered_df, rec)
            st.plotly_chart(fig, use_container_width=True, key=f"builder_chart_{i}_{rec.get('id', i)}")
            with st.expander(f"ℹ️ Metric Lineage: {rec['title']}"):
                st.write(f"**Business Goal:** {rec.get('business_question', 'Analyze category performance')}")
                st.write(f"**Analytical Rule:** {rec.get('requirement_rationale', 'Type-mapped chart')}")
