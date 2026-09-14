import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from core.cache import AnalyticsCache

class ChartModel:
    """Futuristic and enterprise chart recommendation and visual engine with

    neon glow and clean light themes, telemetry tooltips, and sub-millisecond caching.
    """

    FUTURISTIC_PALETTE = [
        "#00F0FF",  # Cyber Cyan
        "#8B5CF6",  # Electric Violet
        "#10B981",  # Neon Emerald
        "#F43F5E",  # Neon Rose
        "#F59E0B",  # Cyber Amber
        "#38BDF8",  # Sky Blue
        "#A855F7",  # Purple Glow
        "#06B6D4"   # Teal Neon
    ]

    CLASSIC_PALETTE = [
        "#2563EB",  # Royal Blue
        "#7C3AED",  # Purple
        "#059669",  # Emerald
        "#DC2626",  # Crimson
        "#D97706",  # Amber
        "#0284C7",  # Sky
        "#4F46E5",  # Indigo
        "#0D9488"   # Teal
    ]

    @classmethod
    def recommend_charts(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        cached = AnalyticsCache.get(df, "chart_recs")
        if cached is not None:
            return cached

        recommendations = []
        n_rows = len(df)
        if n_rows == 0:
            return recommendations

        date_cols = []
        num_cols = []
        cat_cols = []

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or any(k in str(col).lower() for k in ["date", "timestamp", "time"]):
                date_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]) and not any(k in str(col).lower() for k in ["_id", "id_", "identifier"]):
                num_cols.append(col)
            elif df[col].nunique() <= 35 and not any(k in str(col).lower() for k in ["_id", "id_", "identifier"]):
                cat_cols.append(col)

        # 1. Date + Numerical -> Neon Temporal Wave
        if date_cols and num_cols:
            d_col = date_cols[0]
            n_col = num_cols[0]
            recommendations.append({
                "id": "date_trend_line",
                "title": f"// TEMPORAL WAVE: {n_col.replace('_', ' ').upper()}",
                "type": "area",
                "x": str(d_col),
                "y": str(n_col),
                "badge": "QUANTUM TEMPORAL",
                "business_question": f"What is the chronological momentum, trajectory, and acceleration of {n_col}?",
                "requirement_rationale": f"Continuous temporal dimension ({d_col}) mapped to measure ({n_col}) rendered as a glowing cyber-area telemetry wave.",
                "kpi_alignment": f"Monitors velocity and cyclical peaks for {n_col}.",
                "usefulness_score": 98
            })

        # 2. Category + Numerical -> Holographic Segment Bar
        if cat_cols and num_cols:
            c_col = cat_cols[0]
            n_col = num_cols[0]
            recommendations.append({
                "id": "category_metric_bar",
                "title": f"// SEGMENT MATRIX: {n_col.replace('_', ' ').upper()} BY {c_col.replace('_', ' ').upper()}",
                "type": "bar",
                "x": str(c_col),
                "y": str(n_col),
                "aggregation": "sum",
                "badge": "PARETO SPECTRUM",
                "business_question": f"Which {c_col} clusters control the highest share of {n_col}?",
                "requirement_rationale": f"Discrete category nodes ({c_col}) ranked against cumulative values of {n_col} for instant hierarchy discovery.",
                "kpi_alignment": f"Directly monitors segment contribution and resource density across {c_col}.",
                "usefulness_score": 94
            })

        # 3. Two Numerical Variables -> Quantum Scatter Matrix
        if len(num_cols) >= 2:
            x_col = num_cols[0]
            y_col = num_cols[1]
            color_target = str(cat_cols[0]) if cat_cols else None
            recommendations.append({
                "id": "numeric_scatter",
                "title": f"// ORBITAL SCATTER: {y_col.replace('_', ' ').upper()} VS {x_col.replace('_', ' ').upper()}",
                "type": "scatter",
                "x": str(x_col),
                "y": str(y_col),
                "color": color_target,
                "badge": "CORRELATION ORBIT",
                "business_question": f"What bivariate attraction or clustering exists between {x_col} and {y_col}?",
                "requirement_rationale": f"Dual-continuous coordinate plane isolates elasticity, variance clusters, and orbital outliers.",
                "kpi_alignment": f"Evaluates unit economics and efficiency trade-offs between {x_col} and {y_col}.",
                "usefulness_score": 88
            })

        # 4. Single Categorical -> Neon Donut Hologram
        if cat_cols:
            c_col = cat_cols[0]
            recommendations.append({
                "id": "category_donut",
                "title": f"// HOLO-RING: PROPORTION OF {c_col.replace('_', ' ').upper()}",
                "type": "donut",
                "category": str(c_col),
                "badge": "QUANTUM SHARE",
                "business_question": f"What is the market share distribution across {c_col} segments?",
                "requirement_rationale": f"Part-to-whole categorical distribution ({c_col}) visualized as a glowing holographic HUD ring.",
                "kpi_alignment": f"Monitors portfolio diversification and concentration risks across {c_col}.",
                "usefulness_score": 82
            })

        # 5. Single Numerical -> Spectral Distribution Histogram
        if num_cols:
            n_col = num_cols[0]
            recommendations.append({
                "id": "numeric_histogram",
                "title": f"// SPECTRAL DENSITY: {n_col.replace('_', ' ').upper()}",
                "type": "histogram",
                "x": str(n_col),
                "badge": "FREQUENCY GAUSS",
                "business_question": f"What is the probability density, kurtosis, and tail dispersion of {n_col}?",
                "requirement_rationale": f"Spectral frequency distribution with marginal box HUD pinpoints skewness and volatility boundaries.",
                "kpi_alignment": f"Establishes variance limits and normal bounds for operational monitoring.",
                "usefulness_score": 78
            })

        # 6. Secondary Category / Normalized Benchmark
        if len(cat_cols) >= 2 and len(num_cols) >= 1:
            c_col = cat_cols[1]
            n_col = num_cols[0]
            recommendations.append({
                "id": "secondary_category_bar",
                "title": f"// MEAN VELOCITY: {n_col.replace('_', ' ').upper()} BY {c_col.replace('_', ' ').upper()}",
                "type": "bar",
                "x": str(c_col),
                "y": str(n_col),
                "aggregation": "mean",
                "badge": "NORMALIZED VECTOR",
                "business_question": f"How does mean performance of {n_col} compare across secondary dimension {c_col}?",
                "requirement_rationale": f"Unit-normalized bar chart standardizes categorical comparisons across {c_col}.",
                "kpi_alignment": f"Identifies highest average-value cohorts and operational leaders.",
                "usefulness_score": 75
            })

        sorted_recs = sorted(recommendations, key=lambda r: r.get("usefulness_score", 0), reverse=True)
        AnalyticsCache.set(df, "chart_recs", sorted_recs)
        return sorted_recs

    CLASSIC_PALETTE = ["#2563EB", "#10B981", "#8B5CF6", "#F59E0B", "#EC4899", "#06B6D4"]

    @classmethod
    def create_plotly_figure(cls, df: pd.DataFrame, spec: Dict[str, Any], theme: Optional[str] = None) -> go.Figure:
        if theme is None:
            try:
                import streamlit as st
                theme = st.session_state.get("theme", "dark_futuristic")
            except Exception:
                theme = "dark_futuristic"

        cache_key = f"fig_{spec.get('id', 'default')}_{spec.get('type', 'bar')}_{theme}"
        cached_fig = AnalyticsCache.get(df, cache_key)
        if cached_fig is not None:
            return cached_fig

        chart_type = spec.get("type", "bar")
        title = spec.get("title", "TELEMETRY MATRIX")
        
        is_light = (theme == "light_classic")
        palette = cls.CLASSIC_PALETTE if is_light else cls.FUTURISTIC_PALETTE

        if is_light:
            layout_config = dict(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                title=dict(
                    text=f"<b>{title.replace('//', '').strip()}</b>",
                    font=dict(size=14, color="#0F172A", family="Plus Jakarta Sans, sans-serif")
                ),
                margin=dict(l=24, r=24, t=54, b=30),
                hovermode="closest",
                font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#334155"),
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", tickfont=dict(size=10, color="#64748B")),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", tickfont=dict(size=10, color="#64748B"))
            )
        else:
            layout_config = dict(
                template="plotly_dark",
                paper_bgcolor="#0A0E1A",
                plot_bgcolor="#0A0E1A",
                title=dict(
                    text=f"<span style='color: #00F0FF; font-family: monospace; font-size: 13px;'>{title}</span>",
                    font=dict(size=14, color="#00F0FF", family="JetBrains Mono, monospace")
                ),
                margin=dict(l=24, r=24, t=54, b=30),
                hovermode="closest",
                font=dict(family="JetBrains Mono, Plus Jakarta Sans, sans-serif", size=11, color="#94A3B8"),
                xaxis=dict(showgrid=True, gridcolor="#172033", linecolor="#1E293B", tickfont=dict(size=10, color="#64748B"), zerolinecolor="#1E293B"),
                yaxis=dict(showgrid=True, gridcolor="#172033", linecolor="#1E293B", tickfont=dict(size=10, color="#64748B"), zerolinecolor="#1E293B"))

        if chart_type in ["line", "area"]:
            x_col = spec.get("x")
            y_col = spec.get("y")
            color_col = spec.get("color")
            temp = df.dropna(subset=[x_col, y_col]).sort_values(x_col)
            
            if temp[x_col].duplicated().any():
                agg_df = temp.groupby(x_col)[y_col].sum().reset_index()
                if len(agg_df) > 400:
                    step = max(1, len(agg_df) // 300)
                    agg_df = agg_df.iloc[::step]
                show_markers = (len(agg_df) <= 80)
                fig = px.area(agg_df, x=x_col, y=y_col, markers=show_markers, color_discrete_sequence=["#00F0FF"])
            else:
                if len(temp) > 400:
                    step = max(1, len(temp) // 300)
                    temp = temp.iloc[::step]
                show_markers = (len(temp) <= 80)
                fig = px.area(temp, x=x_col, y=y_col, color=color_col if color_col in temp.columns else None,
                              markers=show_markers, color_discrete_sequence=palette)

            fig.update_traces(
                line=dict(width=2.5, color="#00F0FF"),
                fillcolor="rgba(0, 240, 255, 0.12)"
            )
            if show_markers:
                fig.update_traces(marker=dict(size=6, color="#00F0FF", line=dict(width=2, color="#0A0E1A")))

        elif chart_type == "bar":
            x_col = spec.get("x")
            y_col = spec.get("y")
            color_col = spec.get("color")
            agg_method = spec.get("aggregation", "sum")
            
            if y_col:
                temp = df.dropna(subset=[x_col, y_col])
                if agg_method == "sum":
                    grouped = temp.groupby(x_col)[y_col].sum().reset_index()
                else:
                    grouped = temp.groupby(x_col)[y_col].mean().reset_index()
                grouped = grouped.sort_values(by=y_col, ascending=False).head(12)
                fig = px.bar(grouped, x=x_col, y=y_col, color=x_col, color_discrete_sequence=palette)
            else:
                counts = df[x_col].value_counts().reset_index().head(12)
                counts.columns = [x_col, "count"]
                fig = px.bar(counts, x=x_col, y="count", color=x_col, color_discrete_sequence=palette)

            fig.update_traces(
                marker_line_width=1,
                marker_line_color="rgba(255, 255, 255, 0.2)",
                opacity=0.9
            )

        elif chart_type == "scatter":
            x_col = spec.get("x")
            y_col = spec.get("y")
            color_col = spec.get("color")
            temp = df.dropna(subset=[x_col, y_col])
            
            # Subsample large scatter plots for instant rendering and zero browser lag
            if len(temp) > 1200:
                temp = temp.sample(n=1200, random_state=42)
            
            fig = px.scatter(temp, x=x_col, y=y_col, color=color_col if color_col in temp.columns else None,
                             color_discrete_sequence=palette,
                             render_mode="webgl" if len(temp) > 400 else "auto")

            fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=1.5, color="#0A0E1A")))

        elif chart_type == "histogram":
            x_col = spec.get("x")
            sample_df = df.dropna(subset=[x_col])
            if len(sample_df) > 4000:
                sample_df = sample_df.sample(n=4000, random_state=42)
            fig = px.histogram(sample_df, x=x_col, marginal="box" if len(sample_df) <= 1500 else None,
                               color_discrete_sequence=["#8B5CF6"])
            fig.update_traces(
                marker_line_width=1,
                marker_line_color="#A855F7",
                opacity=0.85
            )

        elif chart_type == "donut":
            c_col = spec.get("category", spec.get("x"))
            counts = df[c_col].value_counts().reset_index().head(8)
            counts.columns = [c_col, "count"]
            fig = px.pie(counts, names=c_col, values="count", hole=0.6,
                         color_discrete_sequence=palette)
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color="#0A0E1A", width=2))
            )

        elif chart_type == "heatmap":
            num_df = df.select_dtypes(include=[np.number])
            if len(num_df) > 5000:
                num_df = num_df.sample(n=5000, random_state=42)
            corr = num_df.corr().round(2)
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis")

        else:
            fig = go.Figure()
            fig.add_annotation(text="MATRIX OFFLINE", showarrow=False, font=dict(color="#00F0FF"))

        fig.update_layout(**layout_config)
        AnalyticsCache.set(df, cache_key, fig)
        return fig
