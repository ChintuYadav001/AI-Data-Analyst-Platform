"""
ui.views.neon_card
==================
Interactive Neon Serverless PostgreSQL Ingestion & Cloud Storage Cockpit.
Empowers users to:
1. Connect securely to Neon cloud databases using SSL connection URIs
2. Live ping and latency testing
3. Fetch any table from Neon directly into the analytics workspace pipeline
4. Store / upload the current active workspace dataset into a Neon table
"""

import os
import time
import streamlit as st
import pandas as pd
from typing import Optional

from core.neon_db import (
    get_neon_url,
    test_neon_connection,
    list_neon_tables,
    fetch_table_from_neon,
    store_dataframe_to_neon,
    sanitize_table_name,
)
from core.models import ProjectModel
from core.parquet_manager import ParquetManager
from core.profile_engine import ProfileEngine
from core.quality_engine import QualityEngine
from core.eda_planner import EDAPlanner
from core.eda_engine import EDAEngine
from core.config import PARQUET_DIR, BASE_DIR


def render_neon_cockpit(is_auth: bool = True):
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(0, 240, 255, 0.08) 0%, rgba(16, 185, 129, 0.04) 100%);
                border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 12px; padding: 18px 22px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="background: rgba(0, 240, 255, 0.15); color: #00F0FF; font-size: 0.72rem; font-weight: 800; padding: 3px 10px; border-radius: 20px; border: 1px solid rgba(0, 240, 255, 0.3);">
                    🐘 NEON SERVERLESS POSTGRESQL CLOUD CONNECTOR
                </span>
                <h3 style="margin: 6px 0 2px 0; color: #FFFFFF; font-size: 1.35rem;">
                    Cloud Storage & Live Ingestion Hub
                </h3>
                <p style="color: #94A3B8; font-size: 0.84rem; margin: 0;">
                    Fetch cloud PostgreSQL tables directly into memory or persist processed analytical datasets into your Neon database.
                </p>
            </div>
            <div>
                <span style="font-family: monospace; font-size: 0.75rem; color: #10B981; background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; padding: 4px 12px; border-radius: 16px;">
                    ● SSL ENCRYPTED (sslmode=require)
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 1. CONNECTION CONFIGURATION
    # -------------------------------------------------------------------------
    env_neon_url = os.getenv("NEON_DATABASE_URL", os.getenv("DATABASE_URL", ""))
    saved_url = st.session_state.get("neon_active_conn_str", env_neon_url)

    c_url1, c_url2 = st.columns([3, 1])
    with c_url1:
        neon_input_url = st.text_input(
            "🔑 Neon PostgreSQL Connection String (DATABASE_URL)",
            value=saved_url,
            type="password",
            placeholder="postgresql://user:password@ep-xyz.us-east-2.aws.neon.tech/neondb?sslmode=require",
            help="Found in your Neon Console under Dashboard -> Connection Details -> Connection String (Direct or Pooled)."
        )
    with c_url2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            test_clicked = st.button("⚡ Test Ping", use_container_width=True)
        with c_btn2:
            save_clicked = st.button("💾 Save URL", use_container_width=True)

    if save_clicked:
        if neon_input_url.strip():
            st.session_state["neon_active_conn_str"] = neon_input_url.strip()
            os.environ["NEON_DATABASE_URL"] = neon_input_url.strip()
            # Append/update to .env file
            try:
                env_path = BASE_DIR / ".env"
                lines = []
                found = False
                if env_path.exists():
                    with open(env_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    for i, l in enumerate(lines):
                        if l.startswith("NEON_DATABASE_URL=") or l.startswith("DATABASE_URL="):
                            lines[i] = f"NEON_DATABASE_URL={neon_input_url.strip()}\n"
                            found = True
                            break
                if not found:
                    lines.append(f"\nNEON_DATABASE_URL={neon_input_url.strip()}\n")
                with open(env_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                st.success("✓ Neon Connection URL saved to environment and active session!")
            except Exception as e:
                st.info(f"Saved to active session: {str(e)}")
        else:
            st.warning("Please enter a valid Neon connection string.")

    # Connection Test Result
    if test_clicked:
        with st.spinner("Pinging Neon Cloud Database over TLS..."):
            test_res = test_neon_connection(neon_input_url)
            if test_res["success"]:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-radius: 8px; padding: 12px 16px; margin: 10px 0;">
                    <div style="color: #10B981; font-weight: 700; font-size: 0.88rem;">
                        ✓ Connection to Neon Established Successfully!
                    </div>
                    <div style="font-size: 0.8rem; color: #E2E8F0; margin-top: 4px;">
                        • Roundtrip Latency: <b>{test_res['latency_ms']} ms</b> | Database: <code>{test_res['database_name']}</code> | Active User: <code>{test_res['current_user']}</code> | Public Tables: <b>{test_res['public_tables_count']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Connection Failed: {test_res.get('error')}")

    active_url = neon_input_url.strip() or saved_url

    st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.08); margin: 20px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. STORE & FETCH TABS
    # -------------------------------------------------------------------------
    tab_fetch, tab_store = st.tabs([
        "📥 Fetch Dataset from Neon (Database → Workspace)",
        "💾 Store Dataset to Neon (Workspace → Database)"
    ])

    # =========================================================================
    # TAB: FETCH FROM NEON
    # =========================================================================
    with tab_fetch:
        st.markdown("#### 📥 Ingest Cloud Table into Analytical Workspace")
        st.caption("Select any table from your Neon PostgreSQL instance to stream it into DuckDB and Scikit-Learn pipelines.")

        if not active_url:
            st.info("👆 Please provide and save your Neon connection string above to explore and fetch cloud tables.")
        else:
            with st.spinner("Querying Neon information_schema..."):
                discovered_tables = list_neon_tables(active_url)

            if not discovered_tables:
                st.warning("No public user tables discovered in this Neon database.")
                st.markdown("""
                *Tips:*
                1. Make sure your tables reside in the default `public` schema.
                2. Use the **Store Dataset to Neon** tab to upload your first table!
                """)
            else:
                table_names = [t["table_name"] for t in discovered_tables]
                c_f1, c_f2 = st.columns([2, 1])

                with c_f1:
                    selected_table = st.selectbox(
                        "Select Table to Ingest:",
                        options=table_names,
                        format_func=lambda name: next(
                            (f"📊 {t['table_name']} (~{t['estimated_rows']:,} rows, {t['size_formatted']})" for t in discovered_tables if t['table_name'] == name),
                            name
                        )
                    )

                with c_f2:
                    limit_choice = st.selectbox(
                        "Row Ingestion Limit:",
                        options=["All Records", "5,000 Rows", "25,000 Rows", "100,000 Rows", "250,000 Rows"],
                        index=0
                    )
                    limit_val = None
                    if limit_choice != "All Records":
                        limit_val = int(limit_choice.split()[0].replace(",", ""))

                col_goal1, col_goal2 = st.columns(2)
                with col_goal1:
                    workspace_label = st.text_input("Workspace Name", value=f"{selected_table.replace('_', ' ').title()}")
                with col_goal2:
                    goal_options = [
                        "Revenue Maximization & Growth",
                        "Customer Churn Prevention & Retention",
                        "Cost Reduction & Margin Optimization",
                        "Anomaly & Financial Risk Mitigation",
                        "Conversion Velocity & Efficiency",
                    ]
                    selected_goal = st.selectbox("Strategic Analysis Goal", options=goal_options, index=0)

                c_act1, c_act2 = st.columns([1, 2])
                with c_act1:
                    preview_clicked = st.button("👁️ Quick Preview (5 Rows)", use_container_width=True)
                with c_act2:
                    fetch_clicked = st.button("🚀 Fetch & Ingest into Platform Workspace", type="primary", use_container_width=True)

                if preview_clicked:
                    with st.spinner(f"Fetching preview of '{selected_table}'..."):
                        try:
                            df_prev, p_meta = fetch_table_from_neon(selected_table, active_url, limit=5)
                            st.markdown(f"**Previewing {selected_table} (5 sample rows):**")
                            st.dataframe(df_prev, use_container_width=True)
                        except Exception as e:
                            st.error(f"Preview failed: {str(e)}")

                if fetch_clicked:
                    with st.spinner(f"Streaming '{selected_table}' from Neon Cloud to Vectorized DuckDB..."):
                        try:
                            fetched_df, meta = fetch_table_from_neon(selected_table, active_url, limit=limit_val)
                            n_rows = len(fetched_df)
                            n_cols = len(fetched_df.columns)

                            # Save to local Snappy Parquet zero-copy cache
                            ds_hash = f"neon_{selected_table}_{int(time.time())}"
                            parquet_file = PARQUET_DIR / f"{ds_hash}.parquet"
                            ParquetManager.write_parquet(fetched_df, parquet_file)

                            # Run profile & quality scan
                            profile = ProfileEngine.profile_dataset_fast(parquet_file, dataset_hash=ds_hash)
                            quality = QualityEngine.run_quality_scan(parquet_file, profile)
                            plan = EDAPlanner.create_eda_plan(profile, goal=selected_goal)
                            quick_eda = EDAEngine.run_quick_eda(parquet_file, profile, plan)

                            project = ProjectModel.create_project(
                                workspace_label,
                                fetched_df,
                                goal=selected_goal,
                                goal_explanation=f"Ingested directly from Neon cloud table '{selected_table}'.",
                                parquet_source=parquet_file
                            )

                            st.session_state["current_project_id"] = project.project_id
                            st.session_state["current_project_name"] = project.name
                            st.session_state["dataset_hash"] = ds_hash
                            st.session_state["parquet_path"] = str(parquet_file)
                            st.session_state["dataset_profile"] = profile
                            st.session_state["data_quality"] = quality
                            st.session_state["eda_plan"] = plan
                            st.session_state["quick_eda"] = quick_eda
                            st.session_state["project_goal"] = selected_goal
                            st.session_state["df"] = fetched_df
                            st.session_state["raw_df"] = fetched_df.copy()
                            st.session_state["cleaning_log"] = []
                            st.session_state["active_nav"] = "Dashboard"

                            st.toast(f"🎉 Successfully ingested '{selected_table}' ({n_rows:,} rows) from Neon!", icon="🐘")
                            st.success(f"✓ Ingested **{n_rows:,} records** across **{n_cols} columns** in {meta['fetch_time_seconds']}s.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to fetch table from Neon: {str(e)}")

    # =========================================================================
    # TAB: STORE TO NEON
    # =========================================================================
    with tab_store:
        st.markdown("#### 💾 Persist Workspace Dataset to Neon Cloud Table")
        st.caption("Upload your active dataset, cleaned transformations, or synthesized vectors directly into a Neon PostgreSQL table.")

        active_df = st.session_state.get("df")
        curr_name = st.session_state.get("current_project_name", "dataset")

        if active_df is None or active_df.empty:
            st.warning("⚠️ No active dataset found in the current workspace. Please upload a file or load a sample dataset first.")
        elif not active_url:
            st.info("👆 Please provide and save your Neon connection string above to store tables to Neon.")
        else:
            st.markdown(f"""
            <div style="background: rgba(14, 20, 36, 0.85); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 8px; padding: 12px 16px; margin-bottom: 14px;">
                <div style="font-size: 0.72rem; color: #00F0FF; font-weight: 700;">// ACTIVE WORKSPACE PAYLOAD READY FOR EXPORT</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #FFFFFF; margin-top: 2px;">{curr_name}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">
                    {len(active_df):,} records &bull; {len(active_df.columns)} columns &bull; Memory: ~{round(active_df.memory_usage(deep=True).sum() / (1024 * 1024), 2)} MB
                </div>
            </div>
            """, unsafe_allow_html=True)

            c_s1, c_s2 = st.columns(2)
            with c_s1:
                target_table_name = st.text_input(
                    "Target Neon Table Name:",
                    value=sanitize_table_name(curr_name),
                    help="Table names are automatically sanitized into valid PostgreSQL lowercase identifiers."
                )
            with c_s2:
                mode_choice = st.radio(
                    "Table Storage Mode:",
                    options=["replace", "append"],
                    format_func=lambda m: "Overwrite / Replace Table" if m == "replace" else "Append to Existing Table",
                    horizontal=True
                )

            chunk_size = st.slider("Bulk Insert Chunk Size (Rows / Batch):", min_value=500, max_value=5000, value=2000, step=500)

            if st.button("☁️ Upload & Store to Neon Database", type="primary", use_container_width=True):
                with st.spinner(f"Streaming {len(active_df):,} rows to Neon table '{target_table_name}'..."):
                    try:
                        res = store_dataframe_to_neon(
                            df=active_df,
                            table_name=target_table_name,
                            conn_str=active_url,
                            if_exists=mode_choice,
                            chunksize=chunk_size,
                        )
                        st.toast("🎉 Successfully stored dataset into Neon PostgreSQL!", icon="🐘")
                        st.markdown(f"""
                        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 10px; padding: 16px; margin-top: 14px;">
                            <div style="color: #10B981; font-weight: 800; font-size: 1rem;">
                                ✓ Dataset Successfully Persisted to Neon Cloud!
                            </div>
                            <div style="font-size: 0.85rem; color: #E2E8F0; margin-top: 6px;">
                                • Table Name: <code>{res['table_name']}</code><br>
                                • Records Uploaded: <b>{res['rows_stored']:,} rows</b><br>
                                • Total Columns: <b>{res['columns_stored']} features</b><br>
                                • Cloud Write Duration: <b>{res['execution_time_seconds']} seconds</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Upload to Neon failed: {str(e)}")
