import streamlit as st
import pandas as pd
from core.models import SQLAnalyticsModel, ChartModel

def render_sql_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Please upload a dataset or load a sample dataset first.")
        return

    df = st.session_state["df"]
    st.title("💾 DuckDB SQL Analytics Workbench")
    st.markdown("Execute high-performance analytical queries directly on your dataset with built-in read-only security guardrails.")

    # Initialize SQL engine
    if "sql_engine" not in st.session_state:
        st.session_state["sql_engine"] = SQLAnalyticsModel()

    engine = st.session_state["sql_engine"]
    table_name = "dataset"
    engine.register_dataframe(table_name, df)

    # Sidebar / Schema drawer
    with st.expander("📖 Table Schema & Available Columns", expanded=False):
        schema_info = engine.get_table_schema(table_name)
        if "columns" in schema_info:
            st.dataframe(pd.DataFrame(schema_info["columns"]), use_container_width=True)

    # Preset query helper
    default_query = f'SELECT * FROM "{table_name}" LIMIT 10;'
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "id" not in str(c).lower()]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    if cat_cols and numeric_cols:
        c_col = cat_cols[0]
        n_col = numeric_cols[0]
        preset = f'SELECT "{c_col}", ROUND(SUM("{n_col}"), 2) AS total_{n_col}, COUNT(*) AS count\nFROM "{table_name}"\nGROUP BY "{c_col}"\nORDER BY total_{n_col} DESC\nLIMIT 10;'
    else:
        preset = default_query

    st.markdown("### Write SQL Query")
    st.caption("Allowed statements: `SELECT`, `WITH`, `GROUP BY`, `ORDER BY`. Destructive commands (`DROP`, `DELETE`, etc.) are blocked.")
    
    query = st.text_area("SQL Editor", value=preset, height=140, help="Press 'Run Query' below to execute against DuckDB.")

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("▶️ Run Query", type="primary")

    if run_btn:
        is_auth = st.session_state.get("authenticated", False)
        # In guest mode, allow previewing preset default/LIMIT query, but require auth for custom arbitrary analytical queries
        is_preset = (query.strip() == preset.strip() or query.strip() == default_query.strip())
        if not is_auth and not is_preset:
            st.warning("🔒 Custom SQL Workbench Execution is restricted in Guest Mode. Please Sign In or Register to execute arbitrary DuckDB queries.")
            if st.button("🔑 Sign In to Run Custom Queries"):
                st.session_state["active_nav"] = "Auth"
                st.rerun()
        else:
            with st.spinner("Executing query in DuckDB..."):
                res = engine.execute_query(query)
                
                if not res["success"]:
                    st.error(f"❌ {res['error']}")
                else:
                    res_df = res["dataframe"]
                    st.success(f" Query executed in **{res['execution_time_ms']} ms** — Returned **{res['row_count']:,} rows**.")


                tab_res, tab_viz = st.tabs(["📋 Result Table", "📊 Visualization"])
                
                with tab_res:
                    if res_df is not None and not res_df.empty:
                        st.dataframe(res_df, use_container_width=True)
                    else:
                        st.info("Query returned 0 rows.")

                with tab_viz:
                    chart_spec = res.get("recommended_chart")
                    if chart_spec and res_df is not None and not res_df.empty:
                        st.markdown(f"**Automated Visualization:** `{chart_spec['title']}`")
                        fig = ChartModel.create_plotly_figure(res_df, chart_spec)
                        st.plotly_chart(fig, use_container_width=True, key="sql_query_chart")
                    else:
                        st.info("Visual representation is not available for this query structure.")
