import streamlit as st
import pandas as pd
import io
from core.models import ProjectModel
from core.sample_data import generate_ecommerce_dataset, generate_customer_churn_dataset
from core.data_ingestion import DataIngestion
from core.parquet_manager import ParquetManager
from core.profile_engine import ProfileEngine
from core.quality_engine import QualityEngine
from core.eda_planner import EDAPlanner
from core.eda_engine import EDAEngine
from core.duckdb_engine import DuckDBEngine
from ui.views.neon_card import render_neon_cockpit

def render_upload_view():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #00F0FF; letter-spacing: 0.1em;">// TELEMETRY INGESTION PORTAL</div>
        <h2 style="margin: 4px 0 8px 0;">📁 DATASET INGESTION & CLOUD STORAGE</h2>
        <p style="color: #94A3B8; font-size: 0.95rem;">Ingest structured tabular datasets via local files or stream/fetch directly from Neon Serverless PostgreSQL.</p>
    </div>
    """, unsafe_allow_html=True)

    is_auth = st.session_state.get("authenticated", False)

    # If guest / unauthenticated: show a clean static banner stating login is required to upload, but do NOT ask popup modal
    if not is_auth:
        st.markdown("""
        <div style="background: rgba(255, 0, 85, 0.08); border: 1px solid rgba(255, 0, 85, 0.4); border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 1.5rem;">🔒</span>
                    <div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #FF0055; font-weight: 700; letter-spacing: 0.1em;">
                            // INGESTION DISABLED &bull; AUTHENTICATION REQUIRED
                        </div>
                        <div style="font-size: 1rem; font-weight: 700; color: #FFFFFF; margin-top: 2px;">
                            You must be logged in to upload datasets or connect to Neon cloud databases.
                        </div>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">
                            Without login or sign up, dataset upload is locked. Navigate to the Account tab in the sidebar whenever you are ready to sign in.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    tab_local, tab_neon = st.tabs([
        "📁 Local File Stream Ingestion",
        "🐘 Neon Serverless PostgreSQL (Store & Fetch)"
    ])

    with tab_local:
        col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### // UPLOAD LOCAL DATASTREAM")
        if not is_auth:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px dashed rgba(255, 0, 85, 0.4); border-radius: 8px; padding: 32px 20px; text-align: center; margin-top: 10px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">🔒</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #FFFFFF; font-size: 1.05rem;">
                    File Ingestion Locked
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">
                    Authentication is required to upload CSV or Excel datasets.<br>
                    Please sign in or register via the <b style="color: #00F0FF;">🔑 Account / Auth</b> section in the sidebar.
                </div>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = None
        else:
            uploaded_file = st.file_uploader(
                "Select file to ingest",
                type=["csv", "xlsx", "xls", "parquet"],
                help="Supported formats: CSV, XLSX, XLS, PARQUET. Enterprise scale: supports datasets up to 2,000,000+ records (up to 4 GB)."
            )

        if uploaded_file is not None and is_auth:
            file_name = uploaded_file.name
            file_size_mb = uploaded_file.size / (1024 * 1024)

            st.markdown(f"""
            <div style="background: rgba(0, 240, 255, 0.08); border: 1px solid #00F0FF; border-radius: 8px; padding: 12px; margin: 12px 0;">
                <span style="color: #00F0FF; font-family: monospace; font-weight: 700;">[INGESTION BUFFER READY]</span><br>
                <b>File:</b> <code>{file_name}</code> | <b>Payload Size:</b> {file_size_mb:.2f} MB
            </div>
            """, unsafe_allow_html=True)

            try:
                ingest_res = DataIngestion.ingest_file(uploaded_file, filename=file_name)
                p_rows = ingest_res["rows"]
                p_cols = ingest_res["columns"]
                parquet_path = ingest_res["parquet_path"]
                ds_hash = ingest_res["dataset_hash"]

                if p_rows >= 2000000:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 10px; padding: 14px 18px; margin: 12px 0;">
                        <span style="font-family: monospace; font-size: 0.82rem; color: #10B981; font-weight: 700;">⚡ 2M+ ULTRA SCALE ENGINE ACTIVATED &bull; {p_rows:,} ROWS DETECTED</span>
                        <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 4px;">
                            ✓ Vectorized DuckDB OLAP Pipeline &bull; ✓ Snappy Parquet Zero-Copy Storage &bull; ✓ Progressive Analytics &bull; ✓ WebGL Adaptive Sampling
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif p_rows >= 250000:
                    st.markdown(f"""
                    <div style="background: rgba(0, 240, 255, 0.12); border: 1px solid #00F0FF; border-radius: 10px; padding: 14px 18px; margin: 12px 0;">
                        <span style="font-family: monospace; font-size: 0.82rem; color: #00F0FF; font-weight: 700;">⚡ LARGE DATASET MODE ACTIVATED &bull; {p_rows:,} ROWS DETECTED</span>
                        <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 4px;">
                            ✓ DuckDB Analytical Engine &bull; ✓ Snappy Parquet Zero-Copy Storage &bull; ✓ Progressive Analytics &bull; ✓ Smart Sampled Visuals
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✓ Neural parsing successful: **{p_rows:,} records** across **{p_cols} feature nodes**.")

                st.markdown(f"""
                <div style="background: #0A0E1A; border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 10px 14px; font-family: monospace; font-size: 0.78rem; margin-bottom: 12px;">
                    <div style="color: #10B981;">✓ STAGE 1: File Ingestion & Validation ({ingest_res.get('ingestion_time_ms', 0):.1f} ms)</div>
                    <div style="color: #10B981;">✓ STAGE 2: Parquet Conversion & SHA-256 Hash ({ds_hash})</div>
                    <div style="color: #10B981;">✓ STAGE 3: Dataset Registration & Version Lineage (dataset_v1)</div>
                    <div style="color: #00F0FF;">⚡ STAGE 4–9: DuckDB Fast Profiling & Priority Analytics Ready</div>
                </div>
                """, unsafe_allow_html=True)

                project_name = st.text_input("Designate Workspace Name", value=ingest_res["workspace_name"])
                
                st.markdown("#### 🎯 Strategic Project Goal")
                goal_options = [
                    "Revenue Maximization & Growth",
                    "Customer Churn Prevention & Retention",
                    "Cost Reduction & Margin Optimization",
                    "Anomaly & Financial Risk Mitigation",
                    "Conversion Velocity & Efficiency",
                    "Custom Objective..."
                ]
                selected_goal_option = st.selectbox(
                    "Select Primary Analytical Objective",
                    options=goal_options,
                    index=0,
                    help="AI engines and dashboard dimensions will calibrate their models and recommendations to this goal."
                )
                
                DEFAULT_EXPLANATIONS = {
                    "Revenue Maximization & Growth": "Identify high-velocity customer segments and product categories, analyze margin leakage across regions, and evaluate price sensitivity to accelerate top-line transaction volume.",
                    "Customer Churn Prevention & Retention": "Audit behavioral indicators of customer attrition, isolate balance and tenure thresholds triggering churn, and detect early warning signals to implement targeted retention workflows.",
                    "Cost Reduction & Margin Optimization": "Detect operational cost bottlenecks, review low-margin or negative-profit product SKUs, and eliminate process friction points across operational cohorts.",
                    "Anomaly & Financial Risk Mitigation": "Surface statistical outliers exceeding 3 standard deviations, analyze high-volatility tail transactions, and evaluate potential fraud or systemic compliance risks.",
                    "Conversion Velocity & Efficiency": "Evaluate pipeline funnel velocity, analyze cross-dimensional conversion rates across channels, and identify factors causing drop-off or slowdown.",
                    "Custom Objective...": "State your specific business hypotheses, key metrics of interest, and target outcomes for this dataset."
                }

                if selected_goal_option == "Custom Objective...":
                    final_goal = st.text_input("Define Custom Goal / KPI", value="Optimize operations and mitigate systemic risk")
                else:
                    final_goal = selected_goal_option

                default_expl = DEFAULT_EXPLANATIONS.get(selected_goal_option, "")
                goal_explanation = st.text_area(
                    "📝 Explain Goal & Hypotheses (Detailed Analytical Brief)",
                    value=default_expl,
                    height=85,
                    help="Detail what you specifically want to discover, diagnose, or optimize. AI engines and dashboard cross-filters will directly test this hypothesis against the dataset."
                )

                if st.button("⚡ INITIALIZE QUANTUM PIPELINE", type="primary", disabled=not is_auth):
                    with st.spinner("⚡ Initializing DuckDB Fast Profiling & Priority Analytics..."):
                        profile = ProfileEngine.profile_dataset_fast(parquet_path, dataset_hash=ds_hash)
                        quality = QualityEngine.run_quality_scan(parquet_path, profile)
                        plan = EDAPlanner.create_eda_plan(profile, goal=final_goal)
                        quick_eda = EDAEngine.run_quick_eda(parquet_path, profile, plan)

                        # Load full dataset so all records (e.g. 250,000+ rows) are fully preserved
                        active_df = pd.read_parquet(parquet_path)

                        project = ProjectModel.create_project(project_name, active_df, goal=final_goal, goal_explanation=goal_explanation, parquet_source=parquet_path)
                        
                        st.session_state["current_project_id"] = project.project_id
                        st.session_state["current_project_name"] = project.name
                        st.session_state["dataset_hash"] = ds_hash
                        st.session_state["parquet_path"] = str(parquet_path)
                        st.session_state["dataset_profile"] = profile
                        st.session_state["data_quality"] = quality
                        st.session_state["eda_plan"] = plan
                        st.session_state["quick_eda"] = quick_eda
                        st.session_state["project_goal"] = final_goal
                        st.session_state["project_goal_explanation"] = goal_explanation
                        st.session_state["df"] = active_df
                        st.session_state["raw_df"] = active_df
                        st.session_state["cleaning_log"] = []
                        st.session_state["active_nav"] = "EDA"
                        st.rerun()

                preview_df = ParquetManager.read_sample(parquet_path, n=5)
                with st.expander("👁️ Telemetry Buffer Preview (First 5 Rows)", expanded=True):
                    st.dataframe(preview_df, width="stretch")

            except Exception as e:
                st.error(f"Ingestion parsing failure: {str(e)}")

    with col2:
        st.markdown("### // SYNTHETIC QUANTUM TEST VECTORS")
        if not is_auth:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px dashed rgba(255, 0, 85, 0.4); border-radius: 8px; padding: 32px 20px; text-align: center; margin-top: 10px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">🔒</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #FFFFFF; font-size: 1.05rem;">
                    Test Matrices Locked
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">
                    Pre-loaded sample vectors are restricted to logged-in users.<br>
                    Authenticate first to initialize synthetic test matrices.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption("Instantly initialize end-to-end telemetry workflows with verified sample matrices:")

            with st.container():
                st.markdown("""
                <div style="background: rgba(10, 14, 26, 0.9); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 10px; padding: 16px; margin-bottom: 14px;">
                    <div style="font-family: monospace; color: #00F0FF; font-size: 0.75rem;">// DATASET VECTOR A &bull; TARGET: REVENUE MAXIMIZATION</div>
                    <h4 style="margin: 4px 0 6px 0; color: #FFFFFF;">🛒 E-Commerce Velocity & Profit Matrix</h4>
                    <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 12px;">500 transaction nodes with regional clusters, product categories, margin distributions, and outlier injections.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("⚡ LOAD E-COMMERCE MATRIX", key="btn_sample_ecom"):
                    with st.spinner("Synthesizing e-commerce vector..."):
                        df = generate_ecommerce_dataset(500)
                        goal = "Revenue Maximization & Growth"
                        goal_explanation = "Identify high-velocity customer segments and product categories, analyze margin leakage across regions, and evaluate price sensitivity to accelerate top-line transaction volume."
                        project = ProjectModel.create_project("E-Commerce Sales", df, goal=goal, goal_explanation=goal_explanation)
                        st.session_state["current_project_id"] = project.project_id
                        st.session_state["current_project_name"] = "E-Commerce Sales"
                        st.session_state["project_goal"] = goal
                        st.session_state["project_goal_explanation"] = goal_explanation
                        st.session_state["df"] = df
                        st.session_state["raw_df"] = df.copy()
                        st.session_state["cleaning_log"] = []
                        st.session_state["active_nav"] = "Dashboard"
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            with st.container():
                st.markdown("""
                <div style="background: rgba(10, 14, 26, 0.9); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 10px; padding: 16px; margin-bottom: 14px;">
                    <div style="font-family: monospace; color: #A855F7; font-size: 0.75rem;">// DATASET VECTOR B &bull; TARGET: CHURN PRESERVATION</div>
                    <h4 style="margin: 4px 0 6px 0; color: #FFFFFF;">👥 Customer Churn & Attrition Matrix</h4>
                    <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 12px;">400 user nodes with demographic profiles, balance ratios, activity levels, and retention signals.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("⚡ LOAD CUSTOMER RETENTION MATRIX", key="btn_sample_churn"):
                    with st.spinner("Synthesizing retention vector..."):
                        df = generate_customer_churn_dataset(400)
                        goal = "Customer Churn Prevention & Retention"
                        goal_explanation = "Audit behavioral indicators of customer attrition, isolate balance and tenure thresholds triggering churn, and detect early warning signals to implement targeted retention workflows."
                        project = ProjectModel.create_project("Customer Churn", df, goal=goal, goal_explanation=goal_explanation)
                        st.session_state["current_project_id"] = project.project_id
                        st.session_state["current_project_name"] = "Customer Churn"
                        st.session_state["project_goal"] = goal
                        st.session_state["project_goal_explanation"] = goal_explanation
                        st.session_state["df"] = df
                        st.session_state["raw_df"] = df.copy()
                        st.session_state["cleaning_log"] = []
                        st.session_state["active_nav"] = "Dashboard"
                        st.rerun()


