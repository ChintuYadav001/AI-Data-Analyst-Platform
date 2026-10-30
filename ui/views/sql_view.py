"""
ui.views.sql_view
=================
AI-Powered Enterprise SQL Analytics Engine with:
  - Natural-language question → High-accuracy DuckDB SQL generation via SmartSQLEngine
  - Direct, evidence-backed Executive Natural Language Answer with key numbers highlighted
  - Prominent interactive Plotly visualizations (Bar, Donut, Line, Area, Treemap, KPI Stat Cards)
  - Interactive Chart Switcher toolbar to toggle views on the fly
  - Smart preset query library based on detected schema
  - DuckDB SQL Workbench editor with syntax validation & self-healing execution
  - Query history & bookmarks with 1-click re-execution
  - Instant CSV & JSON export
  - Full schema explorer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

from core.models import SQLAnalyticsModel
from core.analytics.smart_sql_engine import SmartSQLEngine


# ── Palette & Layout Constants ─────────────────────────────────────────
NEON_CYAN = "#00F0FF"
NEON_VIOLET = "#8B5CF6"
NEON_GREEN = "#10B981"
NEON_ROSE = "#F43F5E"
NEON_AMBER = "#F59E0B"
DARK_BG = "#0A0E1A"

CHART_PALETTE = [NEON_CYAN, NEON_VIOLET, NEON_GREEN, NEON_ROSE, NEON_AMBER, "#38BDF8", "#A855F7", "#06B6D4"]


# ── State Initialization ───────────────────────────────────────────────
def _init_sql_state():
    """Ensure all session keys exist."""
    defaults = {
        "sql_engine": None,
        "sql_query_history": [],
        "sql_last_result_df": None,
        "sql_last_query": "",
        "sql_last_answer": "",
        "sql_last_kpis": [],
        "sql_last_chart_spec": None,
        "sql_bookmarks": [],
        "sql_chart_view_mode": "auto",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Presets Builder ───────────────────────────────────────────────────
# ── Presets Builder (Basic to Extreme Advanced) ─────────────────────
def _build_smart_presets(df: pd.DataFrame, table: str):
    """Generate comprehensive SQL templates from schema across Basic to Extreme Advanced levels."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "id" not in str(c).lower()]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 100 and "id" not in str(c).lower()]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or any(k in str(c).lower() for k in ["date", "time", "timestamp"])]
    id_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["id", "code", "ticket", "identifier"])]
    all_cols = list(df.columns)

    p_metric = num_cols[0] if num_cols else "Sales"
    sec_metric = num_cols[1] if len(num_cols) > 1 else p_metric
    p_dim = cat_cols[0] if cat_cols else "Category"
    sec_dim = cat_cols[1] if len(cat_cols) > 1 else (cat_cols[0] if cat_cols else "Region")
    d_col = date_cols[0] if date_cols else None
    id_col = id_cols[0] if id_cols else (cat_cols[0] if cat_cols else all_cols[0])

    clean_pm = p_metric.replace(" ", "_")
    clean_sm = sec_metric.replace(" ", "_")
    clean_dim = p_dim.replace(" ", "_")

    presets = []

    # ═══════════════════════════════════════════════════════════════════
    # LEVEL 1: BASIC SQL (Retrieval, Filtering, Scalar Aggregations)
    # ═══════════════════════════════════════════════════════════════════
    presets.append({
        "label": "🟢 [Basic] Preview First 25 Records",
        "icon": "👀",
        "level": "Level 1: Basic",
        "query": f'SELECT * FROM "{table}" LIMIT 25;',
        "category": "Basic"
    })

    presets.append({
        "label": "🟢 [Basic] Total Record Count & Summary",
        "icon": "🔢",
        "level": "Level 1: Basic",
        "query": f'SELECT COUNT(*) AS total_records FROM "{table}";',
        "category": "Basic"
    })

    if num_cols:
        stats_exprs = [
            f'ROUND(SUM("{p_metric}"), 2) AS total_{clean_pm}',
            f'ROUND(AVG("{p_metric}"), 2) AS avg_{clean_pm}',
            f'ROUND(MIN("{p_metric}"), 2) AS min_{clean_pm}',
            f'ROUND(MAX("{p_metric}"), 2) AS max_{clean_pm}',
            f'ROUND(STDDEV_SAMP("{p_metric}"), 2) AS std_{clean_pm}'
        ]
        presets.append({
            "label": f"🟢 [Basic] Descriptive Statistics ({p_metric})",
            "icon": "📈",
            "level": "Level 1: Basic",
            "query": f'SELECT\n  ' + ",\n  ".join(stats_exprs) + f'\nFROM "{table}";',
            "category": "Basic"
        })

    presets.append({
        "label": f"🟢 [Basic] Top 10 Highest {p_metric} Records",
        "icon": "🏆",
        "level": "Level 1: Basic",
        "query": f'SELECT * FROM "{table}"\nORDER BY "{p_metric}" DESC\nLIMIT 10;',
        "category": "Basic"
    })

    presets.append({
        "label": f"🟢 [Basic] Distinct Count of {p_dim}",
        "icon": "👥",
        "level": "Level 1: Basic",
        "query": f'SELECT COUNT(DISTINCT "{p_dim}") AS unique_{clean_dim}\nFROM "{table}";',
        "category": "Basic"
    })

    # ═══════════════════════════════════════════════════════════════════
    # LEVEL 2: INTERMEDIATE SQL (Group Aggregations, Having, Dates)
    # ═══════════════════════════════════════════════════════════════════
    if cat_cols and num_cols:
        presets.append({
            "label": f"🔵 [Intermediate] Total {p_metric} & {sec_metric} by {p_dim}",
            "icon": "📊",
            "level": "Level 2: Intermediate",
            "query": f'SELECT\n  "{p_dim}",\n  ROUND(SUM("{p_metric}"), 2) AS total_{clean_pm},\n  ROUND(SUM("{sec_metric}"), 2) AS total_{clean_sm},\n  COUNT(*) AS orders_count\nFROM "{table}"\nGROUP BY "{p_dim}"\nORDER BY total_{clean_pm} DESC\nLIMIT 15;',
            "category": "Intermediate"
        })

        if len(cat_cols) >= 2:
            presets.append({
                "label": f"🔵 [Intermediate] Cross-Breakdown: {p_dim} × {sec_dim}",
                "icon": "🗂️",
                "level": "Level 2: Intermediate",
                "query": f'SELECT\n  "{p_dim}",\n  "{sec_dim}",\n  ROUND(SUM("{p_metric}"), 2) AS total_{clean_pm},\n  COUNT(*) AS records\nFROM "{table}"\nGROUP BY "{p_dim}", "{sec_dim}"\nORDER BY "{p_dim}" ASC, total_{clean_pm} DESC\nLIMIT 20;',
                "category": "Intermediate"
            })

        presets.append({
            "label": f"🔵 [Intermediate] High-Volume Segments (HAVING Threshold)",
            "icon": "⚖️",
            "level": "Level 2: Intermediate",
            "query": f'SELECT\n  "{p_dim}",\n  ROUND(SUM("{p_metric}"), 2) AS total_{clean_pm},\n  COUNT(*) AS order_count\nFROM "{table}"\nGROUP BY "{p_dim}"\nHAVING COUNT(*) >= 10\nORDER BY total_{clean_pm} DESC;',
            "category": "Intermediate"
        })

    if d_col and num_cols:
        presets.append({
            "label": f"🔵 [Intermediate] Monthly Aggregation with DATE_TRUNC",
            "icon": "📅",
            "level": "Level 2: Intermediate",
            "query": f'SELECT\n  DATE_TRUNC(\'month\', TRY_CAST("{d_col}" AS TIMESTAMP)) AS month,\n  ROUND(SUM("{p_metric}"), 2) AS monthly_{clean_pm},\n  COUNT(*) AS orders_count\nFROM "{table}"\nWHERE "{d_col}" IS NOT NULL\nGROUP BY month\nORDER BY month ASC;',
            "category": "Intermediate"
        })

    presets.append({
        "label": f"🔵 [Intermediate] Benchmark Filtering (Above Average {p_metric})",
        "icon": "🎯",
        "level": "Level 2: Intermediate",
        "query": f'SELECT * FROM "{table}"\nWHERE "{p_metric}" > (SELECT AVG("{p_metric}") FROM "{table}")\nORDER BY "{p_metric}" DESC\nLIMIT 25;',
        "category": "Intermediate"
    })

    if len(date_cols) >= 2 and cat_cols:
        presets.append({
            "label": f"🔵 [Intermediate] Fulfillment Duration (DATEDIFF Days)",
            "icon": "🚚",
            "level": "Level 2: Intermediate",
            "query": f'SELECT\n  "{p_dim}",\n  ROUND(AVG(DATEDIFF(\'day\', TRY_CAST("{date_cols[0]}" AS TIMESTAMP), TRY_CAST("{date_cols[1]}" AS TIMESTAMP))), 1) AS avg_duration_days,\n  COUNT(*) AS total_shipments\nFROM "{table}"\nWHERE "{date_cols[0]}" IS NOT NULL AND "{date_cols[1]}" IS NOT NULL\nGROUP BY "{p_dim}"\nORDER BY avg_duration_days ASC;',
            "category": "Intermediate"
        })

    # ═══════════════════════════════════════════════════════════════════
    # LEVEL 3: ADVANCED SQL (Window Functions, Rankings, Moving Averages)
    # ═══════════════════════════════════════════════════════════════════
    if len(cat_cols) >= 2 and num_cols:
        presets.append({
            "label": f"🟣 [Advanced] Top 3 {sec_dim} in Each {p_dim} (PARTITION BY)",
            "icon": "🥇",
            "level": "Level 3: Advanced",
            "query": f'WITH ranked_items AS (\n  SELECT\n    "{p_dim}",\n    "{sec_dim}",\n    ROUND(SUM("{p_metric}"), 2) AS total_{clean_pm},\n    ROW_NUMBER() OVER (PARTITION BY "{p_dim}" ORDER BY SUM("{p_metric}") DESC) AS rank\n  FROM "{table}"\n  GROUP BY "{p_dim}", "{sec_dim}"\n)\nSELECT "{p_dim}", "{sec_dim}", total_{clean_pm}, rank\nFROM ranked_items\nWHERE rank <= 3\nORDER BY "{p_dim}" ASC, rank ASC;',
            "category": "Advanced"
        })

    if d_col and num_cols:
        presets.append({
            "label": f"🟣 [Advanced] Cumulative Running Total of {p_metric}",
            "icon": "📈",
            "level": "Level 3: Advanced",
            "query": f'WITH monthly_data AS (\n  SELECT\n    DATE_TRUNC(\'month\', TRY_CAST("{d_col}" AS TIMESTAMP)) AS month,\n    ROUND(SUM("{p_metric}"), 2) AS monthly_{clean_pm}\n  FROM "{table}"\n  WHERE "{d_col}" IS NOT NULL\n  GROUP BY 1\n)\nSELECT\n  month,\n  monthly_{clean_pm},\n  ROUND(SUM(monthly_{clean_pm}) OVER (ORDER BY month ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_{clean_pm}\nFROM monthly_data\nORDER BY month ASC;',
            "category": "Advanced"
        })

        presets.append({
            "label": f"🟣 [Advanced] 3-Period Smoothed Moving Average",
            "icon": "🌊",
            "level": "Level 3: Advanced",
            "query": f'WITH monthly_totals AS (\n  SELECT\n    DATE_TRUNC(\'month\', TRY_CAST("{d_col}" AS TIMESTAMP)) AS month,\n    ROUND(SUM("{p_metric}"), 2) AS monthly_{clean_pm}\n  FROM "{table}"\n  WHERE "{d_col}" IS NOT NULL\n  GROUP BY 1\n)\nSELECT\n  month,\n  monthly_{clean_pm},\n  ROUND(AVG(monthly_{clean_pm}) OVER (ORDER BY month ASC ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS rolling_3_period_avg\nFROM monthly_totals\nORDER BY month ASC;',
            "category": "Advanced"
        })

    if cat_cols and num_cols:
        presets.append({
            "label": f"🟣 [Advanced] Percentage Share of Total Revenue (SUM OVER)",
            "icon": "🥧",
            "level": "Level 3: Advanced",
            "query": f'SELECT\n  "{p_dim}",\n  ROUND(SUM("{p_metric}"), 2) AS total_{clean_pm},\n  ROUND(100.0 * SUM("{p_metric}") / NULLIF(SUM(SUM("{p_metric}")) OVER (), 0), 2) AS pct_of_total\nFROM "{table}"\nGROUP BY "{p_dim}"\nORDER BY total_{clean_pm} DESC\nLIMIT 15;',
            "category": "Advanced"
        })

    # ═══════════════════════════════════════════════════════════════════
    # LEVEL 4: EXTREME ADVANCED (MoM Growth, Pareto 80/20, RFM, Outliers)
    # ═══════════════════════════════════════════════════════════════════
    if d_col and num_cols:
        presets.append({
            "label": f"🔴 [Extreme] Month-over-Month (MoM) Growth Rate % (LAG)",
            "icon": "🚀",
            "level": "Level 4: Extreme",
            "query": f'WITH monthly_perf AS (\n  SELECT\n    DATE_TRUNC(\'month\', TRY_CAST("{d_col}" AS TIMESTAMP)) AS month,\n    ROUND(SUM("{p_metric}"), 2) AS current_{clean_pm}\n  FROM "{table}"\n  WHERE "{d_col}" IS NOT NULL\n  GROUP BY 1\n),\ngrowth_calc AS (\n  SELECT\n    month,\n    current_{clean_pm},\n    LAG(current_{clean_pm}, 1) OVER (ORDER BY month ASC) AS prior_{clean_pm},\n    ROUND(((current_{clean_pm} - LAG(current_{clean_pm}, 1) OVER (ORDER BY month ASC)) / NULLIF(LAG(current_{clean_pm}, 1) OVER (ORDER BY month ASC), 0)) * 100.0, 2) AS mom_growth_pct\n  FROM monthly_perf\n)\nSELECT month, current_{clean_pm}, prior_{clean_pm}, mom_growth_pct\nFROM growth_calc\nORDER BY month ASC;',
            "category": "Extreme"
        })

    if cat_cols and num_cols:
        presets.append({
            "label": f"🔴 [Extreme] Pareto 80/20 Revenue Concentration Analysis",
            "icon": "⚖️",
            "level": "Level 4: Extreme",
            "query": f'WITH item_volume AS (\n  SELECT\n    "{p_dim}",\n    ROUND(SUM("{p_metric}"), 2) AS total_{clean_pm}\n  FROM "{table}"\n  GROUP BY "{p_dim}"\n),\npareto_eval AS (\n  SELECT\n    "{p_dim}",\n    total_{clean_pm},\n    ROUND(100.0 * SUM(total_{clean_pm}) OVER (ORDER BY total_{clean_pm} DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) / NULLIF(SUM(total_{clean_pm}) OVER (), 0), 2) AS cumulative_pct\n  FROM item_volume\n)\nSELECT "{p_dim}", total_{clean_pm}, cumulative_pct\nFROM pareto_eval\nORDER BY total_{clean_pm} DESC\nLIMIT 25;',
            "category": "Extreme"
        })

    presets.append({
        "label": f"🔴 [Extreme] RFM Customer Segmentation (NTILE 5 & Tiers)",
        "icon": "💎",
        "level": "Level 4: Extreme",
        "query": f'WITH cust_totals AS (\n  SELECT\n    "{p_dim}" AS entity,\n    COUNT(DISTINCT "{id_col}") AS frequency,\n    ROUND(SUM("{p_metric}"), 2) AS monetary\n  FROM "{table}"\n  WHERE "{p_dim}" IS NOT NULL\n  GROUP BY 1\n),\nscored AS (\n  SELECT\n    entity,\n    frequency,\n    monetary,\n    NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,\n    NTILE(5) OVER (ORDER BY monetary ASC) AS m_score\n  FROM cust_totals\n)\nSELECT\n  entity,\n  frequency,\n  monetary,\n  f_score,\n  m_score,\n  CASE\n    WHEN m_score >= 4 AND f_score >= 4 THEN \'Champions / VIP\'\n    WHEN m_score >= 3 AND f_score >= 3 THEN \'Loyal Accounts\'\n    WHEN m_score >= 3 AND f_score <= 2 THEN \'High-Value / At Risk\'\n    ELSE \'Standard Tier\'\n  END AS segment_tier\nFROM scored\nORDER BY monetary DESC\nLIMIT 25;',
        "category": "Extreme"
    })

    if num_cols:
        presets.append({
            "label": f"🔴 [Extreme] Statistical Z-Score Outlier Radar (>2.0 σ)",
            "icon": "🚨",
            "level": "Level 4: Extreme",
            "query": f'WITH stats AS (\n  SELECT\n    AVG("{p_metric}") AS avg_val,\n    STDDEV_SAMP("{p_metric}") AS std_val\n  FROM "{table}"\n)\nSELECT\n  d."{id_col}",\n  d."{p_dim}",\n  d."{p_metric}",\n  ROUND(ABS(d."{p_metric}" - s.avg_val) / NULLIF(s.std_val, 0), 2) AS z_score\nFROM "{table}" d, stats s\nWHERE ABS(d."{p_metric}" - s.avg_val) / NULLIF(s.std_val, 0) > 2.0\nORDER BY z_score DESC\nLIMIT 25;',
            "category": "Extreme"
        })

    return presets


# ── KPI Stat Cards Renderer ───────────────────────────────────────────
def _render_kpi_cards(kpi_cards: List[Dict[str, str]]):
    """Render executive Neon KPI Stat Cards."""
    if not kpi_cards:
        return
    cols = st.columns(min(len(kpi_cards), 4))
    for i, card in enumerate(kpi_cards[:4]):
        with cols[i]:
            val_str = str(card.get('value', '—'))
            val_size = "1.25rem" if len(val_str) > 18 else "1.55rem"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.65));
                        border: 1px solid rgba(0, 240, 255, 0.22); border-radius: 12px;
                        padding: 12px 16px; margin-bottom: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);">
                <div style="font-size: 0.70rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em;">
                    {card.get('label', 'Metric')}
                </div>
                <div style="font-size: {val_size}; font-weight: 800; color: {NEON_CYAN}; margin: 3px 0 2px 0; font-family: 'JetBrains Mono', monospace; word-break: break-word;">
                    {val_str}
                </div>
                <div style="font-size: 0.72rem; color: #64748B;">
                    {card.get('subtext', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Smart Visualizer with Multi-Chart Toolbar ──────────────────────────
def _render_smart_visualization(
    result_df: pd.DataFrame,
    chart_spec: Optional[Dict[str, Any]] = None,
    key_prefix: str = "sql_chart"
):
    """
    Renders interactive Plotly visual graphs with on-the-fly chart type switching.
    Supports: Bar Chart, Donut Chart, Line Trend, Treemap, and Grouped Comparisons.
    """
    if result_df is None or result_df.empty:
        st.info("No data available for visualization.")
        return

    num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
    str_cols = [c for c in result_df.columns if not pd.api.types.is_numeric_dtype(result_df[c])]
    date_cols = [c for c in result_df.columns if pd.api.types.is_datetime64_any_dtype(result_df[c]) or any(k in str(c).lower() for k in ["month", "year", "date", "quarter"])]

    # Single Row / Scalar KPI Card
    if len(result_df) == 1 and num_cols:
        row = result_df.iloc[0]
        kpi_list = [{"label": c.replace("_", " ").title(), "value": f"{row[c]:,.2f}" if isinstance(row[c], (int, float)) else str(row[c]), "subtext": "Computed KPI"} for c in num_cols[:4]]
        _render_kpi_cards(kpi_list)
        return

    # Determine recommended view
    spec_type = (chart_spec or {}).get("type", "auto")

    # Interactive Chart-Type Toolbar
    col_t1, col_t2 = st.columns([3, 1])
    with col_t2:
        chart_options = ["📊 Bar Chart", "🍩 Donut Chart", "📈 Line Trend", "📦 Treemap", "📋 Table"]
        default_idx = 0
        if spec_type == "line" or (date_cols and num_cols):
            default_idx = 2
        selected_view = st.selectbox(
            "Visual Format:",
            chart_options,
            index=default_idx,
            key=f"{key_prefix}_view_mode",
            label_visibility="collapsed"
        )

    layout_cfg = dict(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        margin=dict(l=25, r=25, t=40, b=35),
        font=dict(family="JetBrains Mono, sans-serif", size=11, color="#94A3B8"),
        xaxis=dict(showgrid=True, gridcolor="#172033", linecolor="#1E293B"),
        yaxis=dict(showgrid=True, gridcolor="#172033", linecolor="#1E293B"),
    )

    # 1. Bar Chart View
    if "Bar" in selected_view:
        if str_cols and num_cols:
            x_col = str_cols[0]
            # Grouped bar if multiple metrics
            if len(num_cols) >= 2 and len(result_df) <= 25:
                fig = go.Figure()
                for i, nc in enumerate(num_cols[:3]):
                    fig.add_trace(go.Bar(
                        name=nc.replace("_", " ").title(),
                        x=result_df[x_col].astype(str).head(20),
                        y=result_df[nc].head(20),
                        marker_color=CHART_PALETTE[i % len(CHART_PALETTE)],
                        opacity=0.9
                    ))
                fig.update_layout(barmode="group", title=f"<span style='color:{NEON_CYAN}'>Comparison across {x_col}</span>", **layout_cfg)
                st.plotly_chart(fig, key=f"{key_prefix}_grouped_bar", width="stretch")
            else:
                y_col = num_cols[0]
                plot_df = result_df.head(25)
                fig = px.bar(
                    plot_df, x=x_col, y=y_col, color=y_col,
                    color_continuous_scale=[[0, NEON_VIOLET], [1, NEON_CYAN]],
                    title=f"<span style='color:{NEON_CYAN}'>{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}</span>",
                    text_auto=".2s"
                )
                fig.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.15)", opacity=0.9)
                fig.update_layout(**layout_cfg)
                st.plotly_chart(fig, key=f"{key_prefix}_bar", width="stretch")
        elif num_cols and not str_cols:
            # Single numeric histogram
            n = num_cols[0]
            fig = px.histogram(result_df, x=n, color_discrete_sequence=[NEON_CYAN], title=f"Distribution of {n}")
            fig.update_layout(**layout_cfg)
            st.plotly_chart(fig, key=f"{key_prefix}_hist", width="stretch")
        else:
            st.dataframe(result_df, width="stretch", hide_index=True)

    # 2. Donut Chart View
    elif "Donut" in selected_view:
        if str_cols and num_cols:
            names_col = str_cols[0]
            values_col = num_cols[0]
            plot_df = result_df.head(10)
            fig = px.pie(
                plot_df, names=names_col, values=values_col, hole=0.55,
                color_discrete_sequence=CHART_PALETTE,
                title=f"<span style='color:{NEON_CYAN}'>Share of {values_col.replace('_', ' ').title()} by {names_col}</span>"
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(**layout_cfg)
            st.plotly_chart(fig, key=f"{key_prefix}_donut", width="stretch")
        elif str_cols and not num_cols:
            c = str_cols[0]
            counts = result_df[c].value_counts().head(10).reset_index()
            counts.columns = [c, "count"]
            fig = px.pie(counts, names=c, values="count", hole=0.55, color_discrete_sequence=CHART_PALETTE)
            fig.update_layout(**layout_cfg)
            st.plotly_chart(fig, key=f"{key_prefix}_donut_cat", width="stretch")
        else:
            st.info("Donut chart requires at least 1 category column.")

    # 3. Line Trend View
    elif "Line" in selected_view:
        x_col = date_cols[0] if date_cols else (str_cols[0] if str_cols else result_df.columns[0])
        y_col = num_cols[0] if num_cols else result_df.columns[1]

        plot_df = result_df.sort_values(x_col) if x_col in result_df.columns else result_df
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df[x_col].astype(str),
            y=plot_df[y_col],
            mode="lines+markers",
            line=dict(color=NEON_CYAN, width=3),
            marker=dict(size=6, color=NEON_CYAN, line=dict(color="#FFFFFF", width=1)),
            fill="tozeroy",
            fillcolor="rgba(0, 240, 255, 0.08)",
            name=y_col.replace("_", " ").title()
        ))
        # If secondary numeric column exists
        if len(num_cols) >= 2:
            y2 = num_cols[1]
            fig.add_trace(go.Scatter(
                x=plot_df[x_col].astype(str),
                y=plot_df[y2],
                mode="lines+markers",
                line=dict(color=NEON_VIOLET, width=2.5, dash="dot"),
                marker=dict(size=5, color=NEON_VIOLET),
                name=y2.replace("_", " ").title()
            ))

        fig.update_layout(
            title=f"<span style='color:{NEON_CYAN}'>Trajectory: {y_col.replace('_', ' ').title()} over {x_col}</span>",
            hovermode="x unified",
            **layout_cfg
        )
        st.plotly_chart(fig, key=f"{key_prefix}_line", width="stretch")

    # 4. Treemap View
    elif "Treemap" in selected_view:
        if str_cols and num_cols:
            dim_cols = str_cols[:2] if len(str_cols) >= 2 else [str_cols[0]]
            val_col = num_cols[0]
            # Treemap requires positive values
            clean_df = result_df[result_df[val_col] > 0].head(30)
            if not clean_df.empty:
                fig = px.treemap(
                    clean_df, path=dim_cols, values=val_col,
                    color=val_col, color_continuous_scale=[[0, NEON_VIOLET], [1, NEON_CYAN]],
                    title=f"<span style='color:{NEON_CYAN}'>Hierarchical Breakdown: {val_col.replace('_', ' ').title()}</span>"
                )
                fig.update_layout(**layout_cfg)
                st.plotly_chart(fig, key=f"{key_prefix}_treemap", width="stretch")
            else:
                st.info("Treemap requires positive metric values.")
        else:
            st.info("Treemap requires at least one category and one numeric column.")

    # 5. Table View
    else:
        st.dataframe(result_df, width="stretch", hide_index=True)


# ── Schema Explorer Panel ──────────────────────────────────────────────
def _render_schema_explorer(engine, table_name: str, df: pd.DataFrame):
    """Render a rich schema panel."""
    schema = engine.get_table_schema(table_name)
    cols_info = schema.get("columns", [])

    if not cols_info:
        st.info("Schema information unavailable.")
        return

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
    <div style="padding: 16px 0 6px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, {NEON_CYAN} 0%, {NEON_VIOLET} 100%);
                        display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
                        box-shadow: 0 0 20px rgba(0,240,255,0.3);">⚡</div>
            <div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                    SQL Analytics & Natural Language Engine
                </div>
                <div style="font-size: 0.82rem; color: {NEON_CYAN}; font-weight: 600;">
                    Ask questions in plain English or write DuckDB SQL • Direct AI answers with interactive charts
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
        "💬 Ask a Question & Visual Answers",
        "⌨️ DuckDB SQL Workbench",
        "📜 Query History"
    ])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: Ask a Question (NL → SQL + Direct Answers + Graphs)
    # ═══════════════════════════════════════════════════════════════════
    with tab_ask:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0,240,255,0.06), rgba(139,92,246,0.06));
                    border: 1px solid rgba(0,240,255,0.18); border-radius: 12px; padding: 14px 18px; margin: 8px 0 14px 0;">
            <div style="font-size: 0.74rem; font-weight: 700; color: {NEON_CYAN}; letter-spacing: 0.05em; margin-bottom: 4px;">
                // ENTERPRISE NATURAL LANGUAGE ANALYTICS
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5;">
                Ask any analytical question in plain English. The high-accuracy engine translates it to 
                DuckDB SQL, executes with self-healing validation, generates a direct executive answer with key metrics, 
                and auto-renders interactive visual graphs.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Smart Suggested Questions
        meta = SmartSQLEngine.extract_metadata(df)
        num_cols = meta["numeric_cols"]
        cat_cols = meta["cat_cols"]
        date_cols = meta["date_cols"]

        c_col = cat_cols[0] if cat_cols else "category"
        n_col = num_cols[0] if num_cols else "value"
        sec_num = num_cols[1] if len(num_cols) > 1 else n_col
        d_col = date_cols[0] if date_cols else None

        # Sample entity value from dataset for filter suggestion
        sample_entity = None
        if cat_cols:
            for c in cat_cols:
                if df[c].nunique() <= 50 and df[c].notna().any():
                    sample_entity = (c, str(df[c].dropna().iloc[0]))
                    break

        suggested_chips = [
            f"Which {c_col} has the highest {n_col}?",
            f"Top 5 highest {n_col}",
            f"Total {n_col} and {sec_num} by {c_col}" if sec_num != n_col else f"Total {n_col} by {c_col}",
            f"What is the total {n_col} in {sample_entity[1]}?" if sample_entity else f"What is the total {n_col}?",
            f"Monthly {n_col} trend" if d_col else f"What is the total revenue?",
            f"Month over month {n_col} growth rate" if d_col else f"Percentage of total {n_col} by {c_col}",
        ]

        st.markdown("**💡 Quick Questions** — click to test accuracy:")
        q_cols = st.columns(3)
        selected_suggestion = None
        for i, sq in enumerate(suggested_chips[:6]):
            with q_cols[i % 3]:
                if st.button(f"❓ {sq}", key=f"sq_chip_{i}", width="stretch"):
                    selected_suggestion = sq

        # ── Interactive SQL Mastery Catalog (Basic to Extreme Advanced) ──
        with st.expander("📚 SQL Mastery Catalog: Basic to Extreme Advanced (Click any to run)", expanded=False):
            st.markdown("""
            <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 8px;">
                Explore real enterprise analytical queries trained across 4 levels of SQL sophistication. 
                Click any question to translate, execute on DuckDB, and visualize instantly.
            </div>
            """, unsafe_allow_html=True)
            cat_tabs = st.tabs(["🟢 Level 1: Basic", "🔵 Level 2: Intermediate", "🟣 Level 3: Advanced", "🔴 Level 4: Extreme Advanced"])

            with cat_tabs[0]:
                st.caption("Foundational SQL: Row retrieval, scalar metrics, and simple filtering.")
                b_cols = st.columns(2)
                l1_queries = [
                    f"What is the total {n_col}?",
                    f"Top 5 highest {n_col}",
                    f"How many unique {c_col} are there?",
                    f"What is the average {n_col} in the dataset?",
                ]
                for i, lq in enumerate(l1_queries):
                    with b_cols[i % 2]:
                        if st.button(f"🟢 {lq}", key=f"l1_btn_{i}", width="stretch"):
                            selected_suggestion = lq

            with cat_tabs[1]:
                st.caption("Intermediate SQL: Multi-column GROUP BY, HAVING thresholds, and date truncations.")
                i_cols = st.columns(2)
                l2_queries = [
                    f"Total {n_col} and {sec_num} by {c_col}",
                    f"Monthly {n_col} trend" if d_col else f"Total {n_col} by {c_col}",
                    f"Which {c_col} has the highest {n_col}?",
                    f"Products with {n_col} above average",
                ]
                for i, lq in enumerate(l2_queries):
                    with i_cols[i % 2]:
                        if st.button(f"🔵 {lq}", key=f"l2_btn_{i}", width="stretch"):
                            selected_suggestion = lq

            with cat_tabs[2]:
                st.caption("Advanced SQL: Window functions (ROW_NUMBER, RANK), running sums, and moving averages.")
                a_cols = st.columns(2)
                l3_queries = [
                    f"Top 3 {cat_cols[1] if len(cat_cols) > 1 else 'items'} in each {c_col}",
                    f"Running total of {n_col} over time" if d_col else f"Top 5 {n_col}",
                    f"3 month moving average of {n_col}" if d_col else f"Average {n_col} by {c_col}",
                    f"Percentage of total {n_col} by {c_col}",
                ]
                for i, lq in enumerate(l3_queries):
                    with a_cols[i % 2]:
                        if st.button(f"🟣 {lq}", key=f"l3_btn_{i}", width="stretch"):
                            selected_suggestion = lq

            with cat_tabs[3]:
                st.caption("Extreme Advanced SQL: Period-over-period lags, Pareto 80/20 distribution, RFM tiers, and Z-score outliers.")
                e_cols = st.columns(2)
                l4_queries = [
                    f"Month over month {n_col} growth rate" if d_col else f"Year over year {n_col} growth",
                    "Pareto top products driving 80% of revenue",
                    "RFM customer segmentation",
                    f"Outliers in {n_col} with high z score",
                ]
                for i, lq in enumerate(l4_queries):
                    with e_cols[i % 2]:
                        if st.button(f"🔴 {lq}", key=f"l4_btn_{i}", width="stretch"):
                            selected_suggestion = lq

        st.markdown("---")

        col_input, col_btn = st.columns([4, 1])
        with col_input:
            user_question = st.text_input(
                "🔍 Ask your data a question:",
                value=selected_suggestion or "",
                placeholder="e.g. Which category has the highest sales? or Month over month sales growth rate",
                key="nl_question_input",
                label_visibility="collapsed"
            )
        with col_btn:
            ask_btn = st.button("🚀 Analyze Now", type="primary", key="btn_ask_nl", width="stretch")

        col_schema_t, _ = st.columns([1, 3])
        with col_schema_t:
            show_schema = st.toggle("📖 Show Schema", value=False, key="toggle_schema_ask")
        if show_schema:
            with st.expander("📋 Table Schema & Data Dictionary", expanded=True):
                _render_schema_explorer(engine, table_name, df)

        # Execution trigger
        if (ask_btn or selected_suggestion) and user_question.strip():
            with st.spinner("🧠 Translating question, querying DuckDB & synthesizing answer..."):
                analysis_res = SmartSQLEngine.answer_question(
                    question=user_question,
                    df=df,
                    con=engine.con,
                    table=table_name
                )

            if not analysis_res.get("success"):
                st.error(f"❌ Analysis Error: {analysis_res.get('error')}")
            else:
                res_df = analysis_res["dataframe"]
                generated_sql = analysis_res["sql"]
                answer_md = analysis_res.get("answer_markdown", "")
                kpi_cards = analysis_res.get("kpi_cards", [])
                chart_spec = analysis_res.get("chart_spec")

                # Cache in session state
                st.session_state["sql_last_result_df"] = res_df
                st.session_state["sql_last_query"] = generated_sql
                st.session_state["sql_last_answer"] = answer_md
                st.session_state["sql_last_kpis"] = kpi_cards
                st.session_state["sql_last_chart_spec"] = chart_spec

                # Add to history
                st.session_state["sql_query_history"].insert(0, {
                    "question": user_question,
                    "query": generated_sql,
                    "rows": analysis_res.get("row_count", len(res_df) if res_df is not None else 0),
                    "time_ms": analysis_res.get("execution_time_ms", 0.0),
                })
                st.session_state["sql_query_history"] = st.session_state["sql_query_history"][:30]

        # ── Display Active Result Section ──
        if st.session_state.get("sql_last_result_df") is not None:
            active_df = st.session_state["sql_last_result_df"]
            active_sql = st.session_state.get("sql_last_query", "")
            active_ans = st.session_state.get("sql_last_answer", "")
            active_kpis = st.session_state.get("sql_last_kpis", [])
            active_spec = st.session_state.get("sql_last_chart_spec")

            st.markdown("---")

            # 1. Executive Answer Hero Card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(0, 240, 255, 0.08));
                        border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 14px;
                        padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 4px 25px rgba(0, 240, 255, 0.12);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="background: {NEON_CYAN}; color: #000000; font-weight: 800; font-size: 0.72rem;
                                     padding: 2px 8px; border-radius: 6px; letter-spacing: 0.04em;">EXECUTIVE ANSWER</span>
                        <span style="font-size: 0.78rem; color: #94A3B8;">Verified analytical computation</span>
                    </div>
                    <div style="font-size: 0.75rem; color: {NEON_GREEN}; font-weight: 600;">
                        ✓ DuckDB Execution
                    </div>
                </div>
                <div style="font-size: 1.05rem; color: #FFFFFF; font-weight: 500; line-height: 1.6;">
                    {active_ans}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 2. KPI Stat Cards (High-impact numbers)
            if active_kpis:
                _render_kpi_cards(active_kpis)

            # 3. Primary Visual Graphs & Analytical Breakdown
            res_tabs = st.tabs(["📊 Interactive Visual Graphs", "📋 Data Table", "🔍 Generated SQL & Logic", "⬇️ Export"])

            # Visual Graphs Tab
            with res_tabs[0]:
                st.markdown(f"""
                <div style="font-size: 0.75rem; font-weight: 700; color: {NEON_CYAN}; letter-spacing: 0.05em; margin: 4px 0 10px 0;">
                    // INTERACTIVE VISUAL EXPLORER
                </div>
                """, unsafe_allow_html=True)
                _render_smart_visualization(active_df, active_spec, key_prefix="nl_main")

            # Data Table Tab
            with res_tabs[1]:
                st.dataframe(active_df, width="stretch", hide_index=True)
                st.caption(f"Showing **{len(active_df):,}** analytical rows.")

            # SQL Logic Tab
            with res_tabs[2]:
                st.markdown(f"""
                <div style="font-size: 0.74rem; font-weight: 700; color: {NEON_VIOLET}; letter-spacing: 0.05em; margin-bottom: 6px;">
                    // EXECUTED DUCKDB SQL QUERY
                </div>
                """, unsafe_allow_html=True)
                st.code(active_sql, language="sql")
                st.info("💡 **DuckDB Engine**: This query is executed on an in-memory columnar database with zero data leakage.")

            # Export Tab
            with res_tabs[3]:
                col_csv, col_json = st.columns(2)
                with col_csv:
                    st.download_button(
                        "📥 Download Result as CSV",
                        data=active_df.to_csv(index=False),
                        file_name="sql_analytics_result.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                with col_json:
                    st.download_button(
                        "📥 Download Result as JSON",
                        data=active_df.to_json(orient="records", indent=2),
                        file_name="sql_analytics_result.json",
                        mime="application/json",
                        width="stretch"
                    )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: DuckDB SQL Workbench (manual editor)
    # ═══════════════════════════════════════════════════════════════════
    with tab_editor:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(16,185,129,0.06), rgba(0,240,255,0.06));
                    border: 1px solid rgba(16,185,129,0.18); border-radius: 12px; padding: 14px 18px; margin: 8px 0 14px 0;">
            <div style="font-size: 0.74rem; font-weight: 700; color: {NEON_GREEN}; letter-spacing: 0.05em; margin-bottom: 4px;">
                // DUCKDB SQL WORKBENCH
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5;">
                Write and execute raw SQL queries. Full support for <code>SELECT</code>, <code>WITH</code>,
                window functions, CTEs, and DuckDB analytical expressions.
            </div>
        </div>
        """, unsafe_allow_html=True)

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
            height=160,
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
            with st.spinner("⚡ Executing query on DuckDB with self-healing..."):
                res = SmartSQLEngine.execute_with_self_healing(engine.con, query, df, table=table_name)

            if not res["success"]:
                st.error(f"❌ {res['error']}")
            else:
                res_df = res["dataframe"]
                st.session_state["sql_editor_last_df"] = res_df

                # Add to history
                st.session_state["sql_query_history"].insert(0, {
                    "question": "(Manual SQL)",
                    "query": query,
                    "rows": res["row_count"],
                    "time_ms": res["execution_time_ms"],
                })
                st.session_state["sql_query_history"] = st.session_state["sql_query_history"][:30]

                st.success(f"✅ Returned **{res['row_count']:,} rows** in **{res['execution_time_ms']} ms**")

                ed_tabs = st.tabs(["📊 Visual Graphs", "📋 Data Table", "⬇️ Export"])

                with ed_tabs[0]:
                    _render_smart_visualization(res_df, key_prefix="editor_chart")

                with ed_tabs[1]:
                    st.dataframe(res_df, width="stretch", hide_index=True)

                with ed_tabs[2]:
                    col_csv2, col_json2 = st.columns(2)
                    with col_csv2:
                        st.download_button(
                            "📥 Download CSV",
                            data=res_df.to_csv(index=False),
                            file_name="sql_workbench_result.csv",
                            mime="text/csv",
                            key="dl_csv_editor",
                            width="stretch"
                        )
                    with col_json2:
                        st.download_button(
                            "📥 Download JSON",
                            data=res_df.to_json(orient="records", indent=2),
                            file_name="sql_workbench_result.json",
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
                icon = "💬" if entry["question"] != "(Manual SQL)" else "⌨️"
                with st.expander(f"{icon} {entry['question']} — {entry['rows']:,} rows ({entry['time_ms']}ms)", expanded=False):
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
