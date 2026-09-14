import streamlit as st
import plotly.graph_objects as go
from core.models import DataQualityModel

def render_quality_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Please upload a dataset or load a sample dataset first.")
        return

    df = st.session_state["df"]
    st.title("🛡️ Data Quality Score")
    st.markdown("Automated evaluation of dataset health (0–100) assessing missingness, duplicates, outliers, and data types.")

    quality = DataQualityModel.calculate_quality_score(df)
    score = quality["score"]
    grade = quality["grade"]

    col1, col2 = st.columns([1, 2])

    with col1:
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Dataset Health ({grade})", 'font': {'size': 20, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                'bar': {'color': "#1E3A8A"},
                'steps': [
                    {'range': [0, 50], 'color': "#FEE2E2"},
                    {'range': [50, 75], 'color': "#FEF3C7"},
                    {'range': [75, 100], 'color': "#DCFCE7"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#0A0E1A",
            plot_bgcolor="#0A0E1A",
            font=dict(color="#00F0FF", family="JetBrains Mono, monospace")
        )
        st.plotly_chart(fig, use_container_width=True, key="quality_gauge_chart")

    with col2:
        st.markdown("### 📋 Deductions Breakdown")
        breakdown = quality.get("breakdown", [])
        if not breakdown:
            st.success("🎉 Excellent! Zero data quality defects detected.")
        else:
            for item in breakdown:
                sev = item.get("severity", "Medium")
                color_map = {"High": "🔴", "Medium": "🟡", "Low": "🔵"}
                icon = color_map.get(sev, "⚪")
                
                with st.expander(f"{icon} **{item['category']}** (-{item['deduction']} pts) — {sev} Severity", expanded=True):
                    st.write(item["explanation"])
                    st.caption(f"Impact on health score: Deducted {item['deduction']} points.")

    st.markdown("---")
    st.info("💡 **Next Step:** Navigate to **Data Cleaning** to automatically resolve these issues with one click and recover points!")
