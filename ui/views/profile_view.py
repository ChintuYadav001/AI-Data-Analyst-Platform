import streamlit as st
import pandas as pd
from core.models import DataProfilerModel
from ui.components import render_kpi_card

def render_profile_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Please upload a dataset or load a sample dataset first.")
        return

    df = st.session_state["df"]
    st.title("📋 Automatic Data Profiling")
    st.markdown("Comprehensive statistical inspection, data type classification, and anomaly detection.")

    profile = DataProfilerModel.profile_dataset(df)
    ds = profile["dataset_info"]

    # Top stats
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card("Total Rows", f"{ds['rows']:,}", "Records", "🔢")
    with c2:
        render_kpi_card("Total Columns", f"{ds['columns']}", "Features", "🗂️")
    with c3:
        render_kpi_card("Memory Footprint", f"{ds['memory_mb']} MB", "Deep memory usage", "💾")
    with c4:
        render_kpi_card("Duplicate Rows", f"{ds['duplicate_rows']}", f"{ds['duplicate_pct']}% of data", "🔁")
    with c5:
        render_kpi_card("Missing Cells", f"{ds['total_missing']:,}", f"{ds['missing_pct']}% overall", "❓")

    st.markdown("<br>", unsafe_allow_html=True)

    # Suspicious columns alert
    suspicious = profile.get("suspicious_columns", [])
    if suspicious:
        st.markdown("### ⚠️ Detected Suspicious Columns")
        for s in suspicious:
            st.warning(f"**Column `{s['column']}`**: {s['reason']}")

    # Column Details Table
    st.markdown("### 🔍 Column Classification & Summary")
    cols_data = profile["columns"]
    col_df = pd.DataFrame(cols_data)
    
    # Format table for display
    display_cols = ["name", "semantic_type", "dtype", "unique_count", "null_count", "null_pct", "mode"]
    numeric_extra = ["mean", "median", "std", "min", "max", "q25", "q75", "iqr"]
    for ne in numeric_extra:
        if ne in col_df.columns:
            display_cols.append(ne)

    st.dataframe(col_df[display_cols], use_container_width=True)

    with st.expander("🔎 Deep Inspect Individual Column"):
        chosen_col = st.selectbox("Select column to inspect:", options=list(df.columns))
        if chosen_col:
            c_info = next((c for c in cols_data if c["name"] == chosen_col), None)
            if c_info:
                st.write(c_info)
                if pd.api.types.is_numeric_dtype(df[chosen_col]):
                    st.line_chart(df[chosen_col].dropna().head(100))
                else:
                    st.bar_chart(df[chosen_col].value_counts().head(10))
