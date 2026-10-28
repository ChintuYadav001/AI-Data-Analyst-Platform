"""
ui.views.sql_view
=================
AI-Powered SQL Analytics Engine with:
  - Natural-language question → DuckDB SQL generation (AI or deterministic)
  - Smart preset query library based on detected schema
  - Live query editor with syntax guidance
  - Auto-visualization: bar, line, scatter, pie, heatmap
  - Query history & bookmarks
  - CSV/Excel export of results
  - Full schema explorer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core.models import SQLAnalyticsModel, ChartModel
from core.ai.llm_client import LLMClient


# ── Palette & Layout Constants ─────────────────────────────────────────
NEON_CYAN = "#00F0FF"
NEON_VIOLET = "#8B5CF6"
NEON_GREEN = "#10B981"
NEON_ROSE = "#F43F5E"
NEON_AMBER = "#F59E0B"
DARK_BG = "#0A0E1A"

CHART_PALETTE = [NEON_CYAN, NEON_VIOLET, NEON_GREEN, NEON_ROSE, NEON_AMBER, "#38BDF8", "#A855F7", "#06B6D4"]


# ── Helpers ─────────────────────────────────────────────────────────────
def _init_sql_state():
    """Ensure all session keys exist."""
    defaults = {
        "sql_engine": None,
        "sql_query_history": [],
        "sql_last_result_df": None,
        "sql_last_query": "",
        "sql_bookmarks": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _build_smart_presets(df: pd.DataFrame, table: str):
    """Generate contextual preset queries from schema."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "id" not in c.lower()]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 50]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or any(k in c.lower() for k in ["date", "time", "timestamp"])]
    all_cols = list(df.columns)

    presets = []

    # Always add a basic preview
    presets.append({
        "label": "👀 Preview First 20 Rows",
        "icon": "👀",
        "query": f'SELECT * FROM "{table}" LIMIT 20;',
        "category": "Basic"
    })

    # Row count
    presets.append({
        "label": "📊 Total Row Count",
        "icon": "📊",
        "query": f'SELECT COUNT(*) AS total_rows FROM "{table}";',
        "category": "Basic"
    })

    # Summary statistics
    if num_cols:
        stats_parts = ", ".join([
            f'ROUND(AVG("{c}"), 2) AS avg_{c.replace(" ", "_")}, '
            f'ROUND(MIN("{c}"), 2) AS min_{c.replace(" ", "_")}, '
            f'ROUND(MAX("{c}"), 2) AS max_{c.replace(" ", "_")}'
            for c in num_cols[:4]
        ])
        presets.append({
            "label": "📈 Summary Statistics",
            "icon": "📈",
            "query": f'SELECT {stats_parts}\nFROM "{table}";',
            "category": "Statistics"
        })

    # Top categories
    if cat_cols:
        c = cat_cols[0]
        presets.append({
            "label": f"🏆 Top {c} by Count",
            "icon": "🏆",
            "query": f'SELECT "{c}", COUNT(*) AS count\nFROM "{table}"\nGROUP BY "{c}"\nORDER BY count DESC\nLIMIT 10;',
            "category": "Segments"
        })

    # Aggregation by category
    if cat_cols and num_cols:
        c, n = cat_cols[0], num_cols[0]
        presets.append({
            "label": f"💰 Total {n} by {c}",
            "icon": "💰",
            "query": f'SELECT "{c}", ROUND(SUM("{n}"), 2) AS total_{n.replace(" ", "_")}, COUNT(*) AS records\nFROM "{table}"\nGROUP BY "{c}"\nORDER BY total_{n.replace(" ", "_")} DESC\nLIMIT 15;',
            "category": "Aggregation"
        })
        presets.append({
            "label": f"📐 Average {n} by {c}",
            "icon": "📐",
            "query": f'SELECT "{c}", ROUND(AVG("{n}"), 2) AS avg_{n.replace(" ", "_")}, COUNT(*) AS sample_size\nFROM "{table}"\nGROUP BY "{c}"\nHAVING COUNT(*) >= 2\nORDER BY avg_{n.replace(" ", "_")} DESC\nLIMIT 15;',
            "category": "Aggregation"
        })

    # Percentile / distribution
    if num_cols:
        n = num_cols[0]
        presets.append({
            "label": f"📊 Distribution of {n}",
            "icon": "📊",
            "query": f'SELECT\n  ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{n}"), 2) AS p25,\n  ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY "{n}"), 2) AS median,\n  ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{n}"), 2) AS p75,\n  ROUND(STDDEV("{n}"), 2) AS std_dev,\n  COUNT(*) AS total\nFROM "{table}";',
            "category": "Statistics"
        })

    # Time-series
    if date_cols and num_cols:
        d, n = date_cols[0], num_cols[0]
        presets.append({
            "label": f"📅 Monthly Trend of {n}",
            "icon": "📅",
            "query": f'SELECT DATE_TRUNC(\'month\', "{d}") AS month,\n  ROUND(SUM("{n}"), 2) AS total_{n.replace(" ", "_")},\n  COUNT(*) AS records\nFROM "{table}"\nGROUP BY month\nORDER BY month;',
            "category": "Time Series"
        })

    # Null analysis
    null_checks = " + ".join([f'CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END' for c in all_cols[:10]])
    presets.append({
        "label": "🕳️ Null Value Scan",
        "icon": "🕳️",
        "query": f'SELECT\n  COUNT(*) AS total_rows,\n  ' + ", ".join([f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS null_{c.replace(" ", "_")}' for c in all_cols[:10]]) + f'\nFROM "{table}";',
        "category": "Quality"
    })

    # Correlation proxy
    if len(num_cols) >= 2:
        n1, n2 = num_cols[0], num_cols[1]
        presets.append({
            "label": f"🔗 Correlation: {n1} vs {n2}",
            "icon": "🔗",
            "query": f'SELECT ROUND(CORR("{n1}", "{n2}"), 4) AS correlation,\n  COUNT(*) AS sample_size\nFROM "{table}"\nWHERE "{n1}" IS NOT NULL AND "{n2}" IS NOT NULL;',
            "category": "Statistics"
        })

    # Outlier detection
    if num_cols:
        n = num_cols[0]
        presets.append({
            "label": f"🚨 Outliers in {n}",
            "icon": "🚨",
            "query": f'WITH stats AS (\n  SELECT AVG("{n}") AS mu, STDDEV("{n}") AS sigma FROM "{table}"\n)\nSELECT *\nFROM "{table}", stats\nWHERE ABS("{n}" - mu) > 3 * sigma\nLIMIT 50;',
            "category": "Anomaly"
        })

    return presets


def _nl_to_sql(question: str, df: pd.DataFrame, table: str) -> str:
    """Convert a natural-language question into a DuckDB SQL query."""
    col_info = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        sample_vals = df[c].dropna().head(3).tolist()
        sample_str = ", ".join([str(v) for v in sample_vals])
        col_info.append(f'  - "{c}" ({dtype}) e.g. {sample_str}')
    schema_block = "\n".join(col_info)

    prompt = f"""You are a DuckDB SQL expert. Convert the user's question into a single valid DuckDB SELECT query.

TABLE: "{table}" ({len(df)} rows, {len(df.columns)} columns)
COLUMNS:
{schema_block}

RULES:
1. Output ONLY the SQL query. No explanation, no markdown, no backticks.
2. Use only SELECT/WITH statements. No DDL/DML.
3. Always quote column names with double quotes.
4. Use DuckDB syntax (not MySQL or PostgreSQL-specific).
5. Limit large results to 100 rows unless the user asks for all.
6. Use aggregation functions when the question implies totals, averages, or counts.

USER QUESTION: {question}

SQL:"""

    response = LLMClient.call_llm(
        prompt=prompt,
        system_instruction="You are a SQL generation engine. Output ONLY valid DuckDB SQL. No explanations.",
        max_tokens=400,
        temperature=0.1,
    )

    if response["success"] and response["text"]:
        sql = response["text"].strip()
        # Clean markdown fences if LLM wraps them
        if sql.startswith("```"):
            lines = sql.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            sql = "\n".join(lines).strip()
        if not sql.endswith(";"):
            sql += ";"
        return sql

    # ── Deterministic fallback: pattern-match common questions ──
    return _deterministic_nl_to_sql(question, df, table)


def _deterministic_nl_to_sql(question: str, df: pd.DataFrame, table: str) -> str:
    """Rule-based NL→SQL for offline/no-API scenarios."""
    q = question.lower().strip()
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "id" not in c.lower()]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 50]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower()]

    # Match specific column mentions
    mentioned_num = [c for c in num_cols if c.lower().replace("_", " ") in q]
    mentioned_cat = [c for c in cat_cols if c.lower().replace("_", " ") in q]

    n_col = mentioned_num[0] if mentioned_num else (num_cols[0] if num_cols else None)
    c_col = mentioned_cat[0] if mentioned_cat else (cat_cols[0] if cat_cols else None)
    d_col = date_cols[0] if date_cols else None

    if any(k in q for k in ["how many", "count", "total rows", "number of"]):
        if c_col and any(k in q for k in ["each", "per", "by", "group"]):
            return f'SELECT "{c_col}", COUNT(*) AS count\nFROM "{table}"\nGROUP BY "{c_col}"\nORDER BY count DESC\nLIMIT 20;'
        return f'SELECT COUNT(*) AS total_rows FROM "{table}";'

    if any(k in q for k in ["average", "avg", "mean"]) and n_col:
        if c_col:
            return f'SELECT "{c_col}", ROUND(AVG("{n_col}"), 2) AS avg_{n_col.replace(" ", "_")}\nFROM "{table}"\nGROUP BY "{c_col}"\nORDER BY avg_{n_col.replace(" ", "_")} DESC\nLIMIT 15;'
        return f'SELECT ROUND(AVG("{n_col}"), 2) AS average FROM "{table}";'

    if any(k in q for k in ["sum", "total"]) and n_col:
        if c_col:
            return f'SELECT "{c_col}", ROUND(SUM("{n_col}"), 2) AS total_{n_col.replace(" ", "_")}\nFROM "{table}"\nGROUP BY "{c_col}"\nORDER BY total_{n_col.replace(" ", "_")} DESC\nLIMIT 15;'
        return f'SELECT ROUND(SUM("{n_col}"), 2) AS total FROM "{table}";'

    if any(k in q for k in ["top", "highest", "best", "maximum", "largest"]) and n_col:
        return f'SELECT *\nFROM "{table}"\nORDER BY "{n_col}" DESC\nLIMIT 10;'

    if any(k in q for k in ["bottom", "lowest", "worst", "minimum", "smallest"]) and n_col:
        return f'SELECT *\nFROM "{table}"\nORDER BY "{n_col}" ASC\nLIMIT 10;'

    if any(k in q for k in ["trend", "over time", "monthly", "by month"]) and d_col and n_col:
        return f'SELECT DATE_TRUNC(\'month\', "{d_col}") AS month, ROUND(SUM("{n_col}"), 2) AS total\nFROM "{table}"\nGROUP BY month\nORDER BY month;'

    if any(k in q for k in ["distribution", "spread", "histogram"]) and n_col:
        return f'SELECT\n  ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{n_col}"), 2) AS p25,\n  ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY "{n_col}"), 2) AS median,\n  ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{n_col}"), 2) AS p75,\n  ROUND(STDDEV("{n_col}"), 2) AS std_dev\nFROM "{table}";'

    if any(k in q for k in ["null", "missing", "empty"]):
        checks = ", ".join([f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS null_{c.replace(" ", "_")}' for c in df.columns[:12]])
        return f'SELECT COUNT(*) AS total_rows, {checks}\nFROM "{table}";'

    if any(k in q for k in ["unique", "distinct"]):
        if c_col:
            return f'SELECT COUNT(DISTINCT "{c_col}") AS unique_values FROM "{table}";'
        return f'SELECT ' + ", ".join([f'COUNT(DISTINCT "{c}") AS distinct_{c.replace(" ", "_")}' for c in df.columns[:8]]) + f' FROM "{table}";'

    if any(k in q for k in ["correlat"]) and len(num_cols) >= 2:
        return f'SELECT ROUND(CORR("{num_cols[0]}", "{num_cols[1]}"), 4) AS correlation FROM "{table}";'

    if any(k in q for k in ["outlier", "anomal"]) and n_col:
        return f'WITH s AS (SELECT AVG("{n_col}") AS mu, STDDEV("{n_col}") AS sig FROM "{table}")\nSELECT * FROM "{table}", s WHERE ABS("{n_col}" - mu) > 3 * sig LIMIT 50;'

    # Fallback: show preview
    return f'SELECT * FROM "{table}" LIMIT 20;'


def _auto_visualize(result_df: pd.DataFrame):
    """Render smart auto-visualizations for any query result."""
    if result_df is None or result_df.empty or len(result_df.columns) < 1:
        st.info("No data available for visualization.")
        return

    num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
    str_cols = [c for c in result_df.columns if result_df[c].dtype == "object" or str(result_df[c].dtype) == "category"]
    date_cols = [c for c in result_df.columns if pd.api.types.is_datetime64_any_dtype(result_df[c])]

    layout_cfg = dict(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        margin=dict(l=20, r=20, t=50, b=30),
        font=dict(family="JetBrains Mono, sans-serif", size=11, color="#94A3B8"),
        xaxis=dict(showgrid=True, gridcolor="#172033", linecolor="#1E293B"),
        yaxis=dict(showgrid=True, gridcolor="#172033", linecolor="#1E293B"),
    )

    charts_rendered = 0
    max_charts = 3

    # Pattern 1: Category + Number → Bar Chart
    if str_cols and num_cols and charts_rendered < max_charts:
        x_col, y_col = str_cols[0], num_cols[0]
        plot_df = result_df.head(30)
        fig = px.bar(plot_df, x=x_col, y=y_col, color=x_col,
                     color_discrete_sequence=CHART_PALETTE,
                     title=f"<span style='color:{NEON_CYAN}'>{y_col} by {x_col}</span>")
        fig.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.15)", opacity=0.9)
        fig.update_layout(**layout_cfg)
        st.plotly_chart(fig, key=f"auto_bar_{x_col}_{y_col}", width="stretch")
        charts_rendered += 1

    # Pattern 2: Date + Number → Line/Area
    if date_cols and num_cols and charts_rendered < max_charts:
        x_col, y_col = date_cols[0], num_cols[0]
        plot_df = result_df.sort_values(x_col)
        fig = px.area(plot_df, x=x_col, y=y_col,
                      color_discrete_sequence=[NEON_CYAN],
                      title=f"<span style='color:{NEON_CYAN}'>Trend: {y_col} over {x_col}</span>")
        fig.update_traces(line=dict(width=2.5), fillcolor="rgba(0,240,255,0.1)")
        fig.update_layout(**layout_cfg)
        st.plotly_chart(fig, key=f"auto_trend_{x_col}_{y_col}", width="stretch")
        charts_rendered += 1

    # Pattern 3: Multiple number columns → Grouped bar or scatter
    if len(num_cols) >= 2 and charts_rendered < max_charts:
        if str_cols:
            x_col = str_cols[0]
            plot_df = result_df.head(20)
            fig = go.Figure()
            for i, nc in enumerate(num_cols[:4]):
                fig.add_trace(go.Bar(
                    name=nc, x=plot_df[x_col], y=plot_df[nc],
                    marker_color=CHART_PALETTE[i % len(CHART_PALETTE)],
                    opacity=0.9
                ))
            fig.update_layout(barmode="group",
                              title=f"<span style='color:{NEON_CYAN}'>Comparison across {x_col}</span>",
                              **layout_cfg)
            st.plotly_chart(fig, key=f"auto_grouped_{x_col}", width="stretch")
        else:
            x_col, y_col = num_cols[0], num_cols[1]
            fig = px.scatter(result_df.head(500), x=x_col, y=y_col,
                             color_discrete_sequence=[NEON_VIOLET],
                             title=f"<span style='color:{NEON_VIOLET}'>{y_col} vs {x_col}</span>")
            fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color=DARK_BG)))
            fig.update_layout(**layout_cfg)
            st.plotly_chart(fig, key=f"auto_scatter_{x_col}_{y_col}", width="stretch")
        charts_rendered += 1

    # Pattern 4: Single category column → Pie/Donut
    if str_cols and not num_cols and charts_rendered < max_charts:
        c = str_cols[0]
        counts = result_df[c].value_counts().head(10).reset_index()
        counts.columns = [c, "count"]
        fig = px.pie(counts, names=c, values="count", hole=0.55,
                     color_discrete_sequence=CHART_PALETTE,
                     title=f"<span style='color:{NEON_CYAN}'>Distribution of {c}</span>")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(**layout_cfg)
        st.plotly_chart(fig, key=f"auto_pie_{c}", width="stretch")
        charts_rendered += 1

    # Pattern 5: Single numeric → Histogram
    if num_cols and not str_cols and not date_cols and len(result_df) > 1 and charts_rendered < max_charts:
        n = num_cols[0]
        fig = px.histogram(result_df, x=n, color_discrete_sequence=[NEON_VIOLET],
                           title=f"<span style='color:{NEON_VIOLET}'>Distribution of {n}</span>")
        fig.update_traces(marker_line_width=1, marker_line_color=NEON_VIOLET, opacity=0.85)
        fig.update_layout(**layout_cfg)
        st.plotly_chart(fig, key=f"auto_hist_{n}", width="stretch")
        charts_rendered += 1

    if charts_rendered == 0:
        st.info("📊 Auto-visualization is not available for this result structure. View the data table above.")


def _render_schema_explorer(engine, table_name: str, df: pd.DataFrame):
    """Render a rich schema panel."""
    schema = engine.get_table_schema(table_name)
    cols_info = schema.get("columns", [])

    if not cols_info:
        st.info("Schema information unavailable.")
        return

    # Build enriched schema
    enriched = []
    for ci in cols_info:
        col_name = ci.get("column_name", ci.get("Field", ""))
        col_type = ci.get("column_type", ci.get("Type", ""))
        if col_name in df.columns:
            series = df[col_name]
            enriched.append({
                "Column": col_name,
                "Type": col_type,
                "Non-Null": f"{series.notna().sum():,}",
                "Null %": f"{(series.isna().sum() / len(df) * 100):.1f}%",
                "Unique": f"{series.nunique():,}",
                "Sample": str(series.dropna().head(1).values[0])[:40] if series.notna().any() else "—"
            })

    if enriched:
        st.dataframe(pd.DataFrame(enriched), width="stretch", hide_index=True)
    st.caption(f"📐 **{len(df):,}** rows × **{len(df.columns)}** columns  •  Table: `{table_name}`")


# ── Main View ───────────────────────────────────────────────────────────
def render_sql_view():
    _init_sql_state()

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ Please upload a dataset or load a sample dataset first.")
        return

    df = st.session_state["df"]

    # ── Header ──
    st.markdown(f"""
    <div style="padding: 18px 0 8px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, {NEON_CYAN} 0%, {NEON_VIOLET} 100%);
                        display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
                        box-shadow: 0 0 20px rgba(0,240,255,0.3);">⚡</div>
            <div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                    SQL Analytics Engine
                </div>
                <div style="font-size: 0.82rem; color: {NEON_CYAN}; font-weight: 600;">
                    Ask questions in plain English or write DuckDB SQL • Auto-visualize results
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Init engine ──
    if st.session_state["sql_engine"] is None:
        st.session_state["sql_engine"] = SQLAnalyticsModel()
    engine = st.session_state["sql_engine"]
    table_name = "dataset"
    engine.register_dataframe(table_name, df)

    # ── Layout: 3 tabs ──
    tab_ask, tab_editor, tab_history = st.tabs([
        "💬 Ask a Question",
        "⌨️ SQL Editor",
        "📜 Query History"
    ])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: Ask a Question (NL → SQL)
    # ═══════════════════════════════════════════════════════════════════
    with tab_ask:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0,240,255,0.06), rgba(139,92,246,0.06));
                    border: 1px solid rgba(0,240,255,0.15); border-radius: 12px; padding: 16px 20px; margin: 8px 0 16px 0;">
            <div style="font-size: 0.78rem; font-weight: 700; color: {NEON_CYAN}; letter-spacing: 0.05em; margin-bottom: 6px;">
                // ASK ANYTHING ABOUT YOUR DATA
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5;">
                Type a question in plain English. The AI engine will generate the SQL query,
                execute it on DuckDB, and auto-visualize the results.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Smart question suggestions
        presets = _build_smart_presets(df, table_name)
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "id" not in c.lower()]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 50]

        suggestion_questions = [
            "How many rows are in the dataset?",
            f"What are the top 10 {cat_cols[0]} by count?" if cat_cols else "Show the first 20 rows",
            f"What is the average {num_cols[0]}?" if num_cols else "List all unique values",
            f"Show total {num_cols[0]} by {cat_cols[0]}" if num_cols and cat_cols else "Summarize the data",
            f"Which {cat_cols[0]} has the highest {num_cols[0]}?" if cat_cols and num_cols else "Count distinct values",
            f"Show the distribution of {num_cols[0]}" if num_cols else "Detect null values",
        ]

        st.markdown("**💡 Suggested Questions** — click to auto-fill:")
        q_cols = st.columns(3)
        selected_suggestion = None
        for i, sq in enumerate(suggestion_questions[:6]):
            with q_cols[i % 3]:
                if st.button(f"❓ {sq}", key=f"sq_{i}", width="stretch"):
                    selected_suggestion = sq

        st.markdown("---")

        user_question = st.text_input(
            "🔍 Ask your data a question:",
            value=selected_suggestion or "",
            placeholder="e.g. What is the average revenue by region?",
            key="nl_question_input"
        )

        col_ask, col_schema = st.columns([1, 1])
        with col_ask:
            ask_btn = st.button("🚀 Generate & Run SQL", type="primary", width="stretch")
        with col_schema:
            show_schema = st.toggle("📖 Show Schema", value=False, key="toggle_schema_ask")

        if show_schema:
            with st.expander("📋 Table Schema", expanded=True):
                _render_schema_explorer(engine, table_name, df)

        if ask_btn and user_question.strip():
            with st.spinner("🧠 Generating SQL from your question..."):
                generated_sql = _nl_to_sql(user_question, df, table_name)

            st.markdown(f"""
            <div style="background: rgba(139,92,246,0.08); border: 1px solid rgba(139,92,246,0.25);
                        border-radius: 10px; padding: 12px 16px; margin: 8px 0;">
                <div style="font-size: 0.72rem; font-weight: 700; color: {NEON_VIOLET}; letter-spacing: 0.04em;">
                    // GENERATED SQL
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.code(generated_sql, language="sql")

            # Execute
            with st.spinner("⚡ Executing on DuckDB..."):
                res = engine.execute_query(generated_sql)

            if not res["success"]:
                st.error(f"❌ {res['error']}")
            else:
                res_df = res["dataframe"]
                st.session_state["sql_last_result_df"] = res_df
                st.session_state["sql_last_query"] = generated_sql

                # Add to history
                st.session_state["sql_query_history"].insert(0, {
                    "question": user_question,
                    "query": generated_sql,
                    "rows": res["row_count"],
                    "time_ms": res["execution_time_ms"],
                })
                # Keep last 30
                st.session_state["sql_query_history"] = st.session_state["sql_query_history"][:30]

                # Results
                st.success(f"✅ Returned **{res['row_count']:,} rows** in **{res['execution_time_ms']} ms**")

                res_tab, viz_tab, export_tab = st.tabs(["📋 Results", "📊 Visualization", "⬇️ Export"])

                with res_tab:
                    if res_df is not None and not res_df.empty:
                        st.dataframe(res_df, width="stretch", hide_index=True)
                    else:
                        st.info("Query returned 0 rows.")

                with viz_tab:
                    if res_df is not None and not res_df.empty:
                        _auto_visualize(res_df)
                    else:
                        st.info("No data to visualize.")

                with export_tab:
                    if res_df is not None and not res_df.empty:
                        col_csv, col_json = st.columns(2)
                        with col_csv:
                            st.download_button(
                                "📥 Download CSV",
                                data=res_df.to_csv(index=False),
                                file_name="sql_result.csv",
                                mime="text/csv",
                                width="stretch"
                            )
                        with col_json:
                            st.download_button(
                                "📥 Download JSON",
                                data=res_df.to_json(orient="records", indent=2),
                                file_name="sql_result.json",
                                mime="application/json",
                                width="stretch"
                            )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: SQL Editor (manual)
    # ═══════════════════════════════════════════════════════════════════
    with tab_editor:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(16,185,129,0.06), rgba(0,240,255,0.06));
                    border: 1px solid rgba(16,185,129,0.15); border-radius: 12px; padding: 16px 20px; margin: 8px 0 16px 0;">
            <div style="font-size: 0.78rem; font-weight: 700; color: {NEON_GREEN}; letter-spacing: 0.05em; margin-bottom: 6px;">
                // DUCKDB SQL WORKBENCH
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5;">
                Write and execute SQL queries directly. Supports <code>SELECT</code>, <code>WITH</code>,
                <code>GROUP BY</code>, <code>ORDER BY</code>, aggregate functions, window functions, and CTEs.
                Destructive statements (<code>DROP</code>, <code>DELETE</code>, etc.) are blocked.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Preset query selector
        presets = _build_smart_presets(df, table_name)
        preset_labels = ["— Select a preset query —"] + [p["label"] for p in presets]
        selected_preset_idx = st.selectbox("📋 Quick Query Presets:", preset_labels, key="preset_selector")

        preset_query = ""
        if selected_preset_idx != "— Select a preset query —":
            match = next((p for p in presets if p["label"] == selected_preset_idx), None)
            if match:
                preset_query = match["query"]

        default_query = f'SELECT * FROM "{table_name}" LIMIT 20;'
        query = st.text_area(
            "SQL Query Editor",
            value=preset_query if preset_query else default_query,
            height=180,
            key="sql_editor_textarea",
            help="Write your DuckDB SQL query. Press 'Run Query' to execute."
        )

        col_run, col_clear, col_schema2 = st.columns([1, 1, 1])
        with col_run:
            run_btn = st.button("▶️ Run Query", type="primary", key="run_editor_btn", width="stretch")
        with col_clear:
            if st.button("🗑️ Clear", key="clear_editor_btn", width="stretch"):
                st.rerun()
        with col_schema2:
            show_schema2 = st.toggle("📖 Schema", value=False, key="toggle_schema_editor")

        if show_schema2:
            with st.expander("📋 Table Schema", expanded=True):
                _render_schema_explorer(engine, table_name, df)

        if run_btn and query.strip():
            with st.spinner("⚡ Executing query on DuckDB..."):
                res = engine.execute_query(query)

            if not res["success"]:
                st.error(f"❌ {res['error']}")
            else:
                res_df = res["dataframe"]
                st.session_state["sql_last_result_df"] = res_df
                st.session_state["sql_last_query"] = query

                # Add to history
                st.session_state["sql_query_history"].insert(0, {
                    "question": "(Manual SQL)",
                    "query": query,
                    "rows": res["row_count"],
                    "time_ms": res["execution_time_ms"],
                })
                st.session_state["sql_query_history"] = st.session_state["sql_query_history"][:30]

                st.success(f"✅ Returned **{res['row_count']:,} rows** in **{res['execution_time_ms']} ms**")

                res_tab2, viz_tab2, export_tab2 = st.tabs(["📋 Results", "📊 Visualization", "⬇️ Export"])

                with res_tab2:
                    if res_df is not None and not res_df.empty:
                        st.dataframe(res_df, width="stretch", hide_index=True)
                    else:
                        st.info("Query returned 0 rows.")

                with viz_tab2:
                    if res_df is not None and not res_df.empty:
                        _auto_visualize(res_df)
                    else:
                        st.info("No data to visualize.")

                with export_tab2:
                    if res_df is not None and not res_df.empty:
                        col_csv2, col_json2 = st.columns(2)
                        with col_csv2:
                            st.download_button(
                                "📥 Download CSV",
                                data=res_df.to_csv(index=False),
                                file_name="sql_result.csv",
                                mime="text/csv",
                                key="dl_csv_editor",
                                width="stretch"
                            )
                        with col_json2:
                            st.download_button(
                                "📥 Download JSON",
                                data=res_df.to_json(orient="records", indent=2),
                                file_name="sql_result.json",
                                mime="application/json",
                                key="dl_json_editor",
                                width="stretch"
                            )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: Query History
    # ═══════════════════════════════════════════════════════════════════
    with tab_history:
        history = st.session_state.get("sql_query_history", [])

        if not history:
            st.info("📜 No queries executed yet. Ask a question or run a SQL query to build your history.")
        else:
            st.markdown(f"""
            <div style="font-size: 0.78rem; font-weight: 700; color: {NEON_CYAN}; letter-spacing: 0.05em; margin: 8px 0 12px 0;">
                // QUERY HISTORY ({len(history)} queries)
            </div>
            """, unsafe_allow_html=True)

            for i, entry in enumerate(history):
                with st.expander(f"{'💬' if entry['question'] != '(Manual SQL)' else '⌨️'} {entry['question']} — {entry['rows']:,} rows in {entry['time_ms']}ms", expanded=False):
                    st.code(entry["query"], language="sql")
                    if st.button(f"🔄 Re-run this query", key=f"rerun_{i}", width="stretch"):
                        with st.spinner("Re-executing..."):
                            res = engine.execute_query(entry["query"])
                        if res["success"]:
                            st.success(f"✅ {res['row_count']:,} rows in {res['execution_time_ms']}ms")
                            st.dataframe(res["dataframe"], width="stretch", hide_index=True)
                        else:
                            st.error(f"❌ {res['error']}")

            if st.button("🗑️ Clear History", key="clear_history_btn"):
                st.session_state["sql_query_history"] = []
                st.rerun()
