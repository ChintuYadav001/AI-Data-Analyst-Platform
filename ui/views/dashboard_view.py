import streamlit as st
import pandas as pd
from ui.components import (
    render_saas_hero,
    render_workflow_stepper,
    render_dataset_header_card,
    render_quick_actions,
    render_recent_projects_cards,
    render_ai_analyst_card,
    render_kpi_card,
    render_requirement_spec_box
)
from core.models import DashboardModel, ChartModel, DataQualityModel, ProjectModel
from core.sample_data import (
    generate_sales_dataset,
    generate_customer_churn_dataset,
    generate_ecommerce_dataset,
    generate_finance_dataset,
    generate_marketing_dataset,
    generate_healthcare_dataset
)

def render_dashboard_view():
    theme = st.session_state.get("theme", "dark_futuristic")
    df = st.session_state.get("df")
    project_name = st.session_state.get("current_project_name", "Current Dataset")
    project_goal = st.session_state.get("project_goal", "General Business Intelligence")
    project_goal_explanation = st.session_state.get("project_goal_explanation", "")

    is_auth = bool(st.session_state.get("authenticated", False))

    # =========================================================================
    # STATE A: NO DATASET LOADED (PRIMARY UPLOAD & LANDING WORKFLOW)
    # =========================================================================
    if df is None:
        # Hero Section
        render_saas_hero()

        if not is_auth:
            st.markdown("""
            <div style="background: rgba(255, 0, 85, 0.08); border: 1px solid rgba(255, 0, 85, 0.4); border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 24px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">🔒</div>
                <h3 style="margin: 0 0 6px 0; font-size: 1.35rem; color: #FFFFFF; font-weight: 800;">
                    Dataset Ingestion Locked
                </h3>
                <p style="color: #94A3B8; font-size: 0.95rem; margin: 0 0 16px 0;">
                    Authentication is required to upload datasets or initialize telemetry pipelines.<br>
                    Please sign in or create an analyst account to access the platform.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔑 Proceed to Sign In / Register", key="dash_lock_login_btn", type="primary", use_container_width=True):
                st.session_state["active_nav"] = "Auth"
                st.rerun()
            return

        # Primary Upload Area (Most Important Component)
        st.markdown("""
        <div style="background: rgba(12, 18, 34, 0.7); border: 2px dashed rgba(0, 240, 255, 0.35); border-radius: 14px; padding: 32px 24px; text-align: center; margin-bottom: 20px;">
            <div style="width: 48px; height: 48px; margin: 0 auto 12px auto; border-radius: 12px; background: rgba(0, 240, 255, 0.1); border: 1px solid rgba(0, 240, 255, 0.3); display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                ☁️
            </div>
            <h3 style="margin: 0 0 6px 0; font-size: 1.45rem; color: #FFFFFF; font-weight: 800;">
                Upload Your Dataset
            </h3>
            <p style="color: #94A3B8; font-size: 0.95rem; margin: 0 0 16px 0;">
                Drag & drop your file here, or click to browse
            </p>
            <div style="display: inline-flex; align-items: center; gap: 8px; font-size: 0.78rem; color: #64748B; background: rgba(255, 255, 255, 0.04); padding: 4px 14px; border-radius: 9999px; margin-bottom: 16px;">
                <span>Supported formats: <b style="color: #CBD5E1;">CSV • Excel • JSON • Parquet • SQL</b></span>
                <span>•</span>
                <span>Max size: <b style="color: #00F0FF;">100MB</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Your Dataset",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            label_visibility="collapsed",
            key="dash_primary_file_uploader",
            help="Supported formats: CSV, Excel (.xlsx, .xls), JSON, Parquet. Max size: 100MB."
        )

        if uploaded_file is not None:
            try:
                fname = uploaded_file.name
                if fname.endswith(".csv"):
                    new_df = pd.read_csv(uploaded_file)
                elif fname.endswith((".xlsx", ".xls")):
                    new_df = pd.read_excel(uploaded_file)
                elif fname.endswith(".json"):
                    new_df = pd.read_json(uploaded_file)
                elif fname.endswith(".parquet"):
                    new_df = pd.read_parquet(uploaded_file)
                else:
                    new_df = pd.read_csv(uploaded_file)

                p_name = fname.rsplit(".", 1)[0].replace("_", " ").title()
                project = ProjectModel.create_project(p_name, new_df, goal="General Business Intelligence")
                st.session_state["current_project_id"] = project.project_id
                st.session_state["current_project_name"] = p_name
                st.session_state["df"] = new_df
                st.session_state["raw_df"] = new_df.copy()
                st.success(f"✓ Successfully ingested **{p_name}** ({len(new_df):,} rows, {len(new_df.columns)} columns) into Quantum pipeline.")
                st.rerun()
            except Exception as e:
                st.error(f"Error parsing dataset: {str(e)}")

        # Sample Datasets Bar
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin: 18px 0 10px 0;">
            <span style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.04em;">
                OR TRY A PRE-LOADED SAMPLE DATASET:
            </span>
            <span style="font-size: 0.74rem; color: #64748B;">Click any vector to instantly trigger full AI analytics</span>
        </div>
        """, unsafe_allow_html=True)

        s1, s2, s3, s4, s5, s6 = st.columns(6)
        
        with s1:
            if st.button("🛒 Sales", use_container_width=True, key="btn_sample_sales"):
                with st.spinner("Synthesizing Sales dataset..."):
                    df_sample = generate_sales_dataset(800)
                    p = ProjectModel.create_project("Sales Analysis", df_sample, goal="Revenue Maximization & Growth")
                    st.session_state["current_project_id"] = p.project_id
                    st.session_state["current_project_name"] = "Sales Analysis"
                    st.session_state["df"] = df_sample
                    st.session_state["raw_df"] = df_sample.copy()
                    st.rerun()

        with s2:
            if st.button("👥 Customer", use_container_width=True, key="btn_sample_customer"):
                with st.spinner("Synthesizing Customer dataset..."):
                    df_sample = generate_customer_churn_dataset(400)
                    p = ProjectModel.create_project("Customer Churn", df_sample, goal="Customer Churn Prevention & Retention")
                    st.session_state["current_project_id"] = p.project_id
                    st.session_state["current_project_name"] = "Customer Churn"
                    st.session_state["df"] = df_sample
                    st.session_state["raw_df"] = df_sample.copy()
                    st.rerun()

        with s3:
            if st.button("📦 E-commerce", use_container_width=True, key="btn_sample_ecom"):
                with st.spinner("Synthesizing E-commerce dataset..."):
                    df_sample = generate_ecommerce_dataset(500)
                    p = ProjectModel.create_project("E-Commerce Orders", df_sample, goal="Revenue Maximization & Growth")
                    st.session_state["current_project_id"] = p.project_id
                    st.session_state["current_project_name"] = "E-Commerce Orders"
                    st.session_state["df"] = df_sample
                    st.session_state["raw_df"] = df_sample.copy()
                    st.rerun()

        with s4:
            if st.button("💰 Finance", use_container_width=True, key="btn_sample_fin"):
                with st.spinner("Synthesizing Finance dataset..."):
                    df_sample = generate_finance_dataset(600)
                    p = ProjectModel.create_project("Financial Risk Portfolio", df_sample, goal="Cost Reduction & Margin Optimization")
                    st.session_state["current_project_id"] = p.project_id
                    st.session_state["current_project_name"] = "Financial Risk Portfolio"
                    st.session_state["df"] = df_sample
                    st.session_state["raw_df"] = df_sample.copy()
                    st.rerun()

        with s5:
            if st.button("📣 Marketing", use_container_width=True, key="btn_sample_mkt"):
                with st.spinner("Synthesizing Marketing dataset..."):
                    df_sample = generate_marketing_dataset(500)
                    p = ProjectModel.create_project("Marketing Campaigns", df_sample, goal="Conversion Velocity & Efficiency")
                    st.session_state["current_project_id"] = p.project_id
                    st.session_state["current_project_name"] = "Marketing Campaigns"
                    st.session_state["df"] = df_sample
                    st.session_state["raw_df"] = df_sample.copy()
                    st.rerun()

        with s6:
            if st.button("🏥 Healthcare", use_container_width=True, key="btn_sample_health"):
                with st.spinner("Synthesizing Healthcare dataset..."):
                    df_sample = generate_healthcare_dataset(500)
                    p = ProjectModel.create_project("Healthcare Vitals", df_sample, goal="Anomaly & Operational Optimization")
                    st.session_state["current_project_id"] = p.project_id
                    st.session_state["current_project_name"] = "Healthcare Vitals"
                    st.session_state["df"] = df_sample
                    st.session_state["raw_df"] = df_sample.copy()
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Analytics Workflow Stepper (Ready status)
        render_workflow_stepper(df_loaded=False)

        # Quick Actions Cards
        render_quick_actions()

        # Meet Your AI Data Analyst Card
        render_ai_analyst_card()

        # Recent Projects
        render_recent_projects_cards()
        return

    # =========================================================================
    # STATE B: DATASET-AWARE DASHBOARD (LOGGED IN & ACTIVE DATASET)
    # =========================================================================
    quality = DataQualityModel.calculate_quality_score(df)
    quality_score = quality.get("score", 94)

    # 1. Prominent Current Dataset Header Card
    render_dataset_header_card(df, project_name=project_name, quality_score=quality_score)

    # 2. Dynamic Workflow Progress Bar
    render_workflow_stepper(df_loaded=True)

    # 3. High-Speed Sub-Millisecond KPI Cards
    kpis = DashboardModel.compute_kpis(df)
    if kpis:
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

    # 4. Multi-Dimension Slicing Filter Engine
    dimensions = DashboardModel.detect_filter_dimensions(df)
    active_filters = {}
    if dimensions:
        with st.expander(f"🎛️ Multi-Dimension Slicing & Filter Engine ({len(dimensions)} Dimensions Detected)", expanded=False):
            st.caption("Filter and segment data across multiple categorical dimensions simultaneously:")
            num_dims = min(8, len(dimensions))
            for row_start in range(0, num_dims, 4):
                row_dims = dimensions[row_start:row_start + 4]
                f_cols = st.columns(len(row_dims))
                for idx, dim in enumerate(row_dims):
                    with f_cols[idx]:
                        sel = st.multiselect(
                            f"Filter {dim['column'].replace('_', ' ').title()}:",
                            options=dim["options"],
                            default=[],
                            key=f"dash_multi_filter_{dim['column']}"
                        )
                        if sel:
                            active_filters[dim["column"]] = sel

    # Fast filtered view with sub-millisecond cache
    active_df = DashboardModel.apply_filters(df, active_filters) if active_filters else df
    if len(active_df) < len(df):
        st.info(f"Filtering active: displaying **{len(active_df):,}** of {len(df):,} records ({round((len(active_df)/len(df))*100, 1)}% slice).")

    # 5. Visual Matrix, Architecture & Data Preview Tabs
    tab_charts, tab_spec, tab_data = st.tabs([
        "📈 Executive Visual Matrix",
        "📋 Requirement Architecture & Lineage",
        "👁️ Data Preview & Schema"
    ])

    with tab_charts:
        recs = ChartModel.recommend_charts(active_df)
        if recs:
            c1, c2 = st.columns(2)
            for i, rec in enumerate(recs[:4]):
                col = c1 if i % 2 == 0 else c2
                with col:
                    fig = ChartModel.create_plotly_figure(active_df, rec, theme=theme)
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"dash_chart_{i}_{rec.get('id', i)}",
                        config={"displayModeBar": False, "responsive": True, "scrollZoom": False}
                    )
                    with st.expander(f"ℹ️ Telemetry Rationale: {rec['title']}"):
                        st.markdown(f"**Business Question:** {rec.get('business_question', 'N/A')}")
                        st.markdown(f"**Why this chart:** {rec.get('requirement_rationale', 'N/A')}")
                        st.markdown(f"**Target KPI:** {rec.get('kpi_alignment', 'N/A')}")
        else:
            st.info("Ingest dataset to view automated executive charts.")

    with tab_spec:
        spec = DashboardModel.generate_dashboard_specification(df)
        st.markdown("#### 🎯 Architectural Rationale & Lineage")
        st.markdown("The platform automatically synthesized this dashboard layout based on analytical rules & column semantics:")
        c_kpi_spec, c_arch_spec = st.columns(2)
        with c_kpi_spec:
            st.markdown("##### 📌 Selected KPI Rationale:")
            for k in kpis:
                st.markdown(f"- **{k['label']}** ({k.get('badge', 'Metric')}): {k.get('rationale', 'Calculated from dataset distribution')}")

        with c_arch_spec:
            st.markdown("##### 🏛️ Architectural Principles:")
            for p in spec.get("architecture_principles", []):
                st.markdown(f"- {p}")

    with tab_data:
        st.markdown("#### // Active Data Table View (Top 10 Rows)")
        st.dataframe(active_df.head(10), use_container_width=True)

    # 6. Quick Actions & Meet AI Analyst at Bottom
    render_quick_actions()
    render_ai_analyst_card()

