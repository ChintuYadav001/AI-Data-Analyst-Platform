import streamlit as st
import os
from core.config import OPENAI_API_KEY, GEMINI_API_KEY, BASE_DIR

def render_settings_view():
    st.title("⚙️ Platform Settings & Configuration")
    st.markdown("Manage AI reasoning providers, API credentials, privacy constraints, and database settings.")

    st.markdown("### 🤖 AI Reasoning Configuration")
    
    current_openai = os.getenv("OPENAI_API_KEY", "")
    current_gemini = os.getenv("GEMINI_API_KEY", "")

    openai_key = st.text_input("OpenAI API Key", value=current_openai, type="password", help="Used for GPT-4o / GPT-4o-mini reasoning")
    gemini_key = st.text_input("Google Gemini API Key", value=current_gemini, type="password", help="Used for Gemini 2.5 Flash reasoning")

    model_choice = st.selectbox(
        "Active Reasoning Model",
        options=[
            "Analytical Synthesis Engine (Offline Grounded - Default)",
            "Google Gemini 2.5 Flash",
            "OpenAI GPT-4o-mini"
        ],
        help="Even without external API keys, the system operates with 100% deterministic, grounded analytical engines."
    )

    def _update_env_key(key: str, value: str):
        env_file = BASE_DIR / ".env"
        lines = []
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        found = False
        new_lines = []
        for line in lines:
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={value}\n")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"{key}={value}\n")
            
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    if st.button("💾 Save AI Settings", type="primary"):
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["GEMINI_API_KEY"] = gemini_key
        _update_env_key("OPENAI_API_KEY", openai_key)
        _update_env_key("GEMINI_API_KEY", gemini_key)
        _update_env_key("DEFAULT_MODEL", model_choice)
        st.success("AI Reasoning settings saved successfully!")

    st.markdown("---")
    st.markdown("### 🐘 Neon Serverless PostgreSQL Database")
    st.markdown("Connect to [Neon.tech](https://neon.tech) Serverless PostgreSQL to store datasets, query historical telemetry, and synchronize workspaces across cloud environments.")

    current_neon = os.getenv("NEON_DATABASE_URL", os.getenv("DATABASE_URL", ""))
    
    neon_input = st.text_input(
        "Neon PostgreSQL Connection URI (with SSL)",
        value=current_neon,
        type="password",
        placeholder="postgresql://user:pass@ep-xyz.us-east-2.aws.neon.tech/neondb?sslmode=require",
        help="Paste your Neon connection URI from neon.tech dashboard. Requires SSL (sslmode=require)."
    )

    c_neon1, c_neon2 = st.columns([1, 1])
    with c_neon1:
        if st.button("⚡ Test Neon Ping & Connection", key="btn_settings_test_neon", use_container_width=True):
            if not neon_input.strip():
                st.warning("Please provide a Neon Connection URI to test.")
            else:
                with st.spinner("Testing handshake and latency with Neon cloud..."):
                    try:
                        from core.neon_db import test_neon_connection
                        res = test_neon_connection(neon_input.strip())
                        if res["success"]:
                            st.success(f"✓ Connected! Ping: **{res['latency_ms']:.1f} ms** | Database: `{res['database']}` | Tables: **{res['table_count']}**")
                        else:
                            st.error(f"❌ Connection failed: {res.get('error')}")
                    except Exception as err:
                        st.error(f"Connection error: {str(err)}")

    with c_neon2:
        if st.button("💾 Save Neon URI to Environment", key="btn_settings_save_neon", type="primary", use_container_width=True):
            if neon_input.strip():
                os.environ["NEON_DATABASE_URL"] = neon_input.strip()
                _update_env_key("NEON_DATABASE_URL", neon_input.strip())
                st.session_state["neon_url"] = neon_input.strip()
                st.toast("✓ Neon Database URI saved securely!", icon="🐘")
                st.success("✓ Neon Connection String saved to environment and `.env`!")
            else:
                st.warning("Connection URI cannot be blank.")

    st.markdown("---")
    st.markdown("### 🐙 GitHub Repository & Automatic Sync")
    
    st.markdown("""
    <div style="background: rgba(14, 20, 36, 0.85); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="font-size: 0.72rem; font-weight: 800; color: #10B981; letter-spacing: 0.06em;">CONNECTED REPOSITORY</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-top: 2px;">
                    ChintuYadav001 / AI-Data-Analyst-Platform
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 3px;">
                    Branch: <code style="color: #00F0FF;">main</code> &nbsp;|&nbsp; Target: <a href="https://github.com/ChintuYadav001/AI-Data-Analyst-Platform" target="_blank" style="color: #38BDF8; text-decoration: underline;">github.com/ChintuYadav001/AI-Data-Analyst-Platform</a>
                </div>
            </div>
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; color: #10B981; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">
                ● LIVE AUTO-SYNC ACTIVE
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_sync1, col_sync2 = st.columns([1, 1])
    with col_sync1:
        if st.button("🚀 Sync & Push to GitHub Now", key="btn_sync_github_now", type="primary", use_container_width=True):
            with st.spinner("Staging changes, committing, and pushing to GitHub..."):
                try:
                    from scripts.auto_git_sync import sync_once
                    success, msg = sync_once()
                    if success:
                        st.toast("✓ Successfully synced changes to GitHub!", icon="🚀")
                        st.success(f"✓ {msg}")
                    else:
                        st.info(f"ℹ️ {msg}")
                except Exception as e:
                    st.error(f"Sync error: {str(e)}")

    with col_sync2:
        st.link_button(
            "🌐 Open Repository on GitHub",
            url="https://github.com/ChintuYadav001/AI-Data-Analyst-Platform",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 🧹 Cache & Storage Maintenance")
    st.markdown("Purge stale parquet streams, experiment snapshots, and analytical buffers to optimize RAM and speed up boot times.")

    from core.cache_cleaner import get_cache_size_mb, clean_all_cache
    cache_sizes = get_cache_size_mb()

    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Total Cache Footprint", f"{cache_sizes.get('total', 0.0):.2f} MB")
    col_c2.metric("Project Storage", f"{cache_sizes.get('projects', 0.0):.2f} MB")
    col_c3.metric("Parquet & ML Cache", f"{(cache_sizes.get('parquet', 0.0) + cache_sizes.get('experiments', 0.0)):.2f} MB")

    if st.button("🧹 Purge All Temporary Cache Now", key="btn_clean_cache_now", type="primary"):
        with st.spinner("Purging stale temporary cache and optimizing workspace buffers..."):
            active_p = st.session_state.get("current_project_id")
            clean_res = clean_all_cache(keep_active_project_id=active_p)
            st.toast(f"✓ Purged {clean_res['files_deleted']} files! Freed {clean_res['freed_mb']} MB", icon="🧹")
            st.success(f"✓ Cache cleaned successfully: **{clean_res['files_deleted']} files deleted**, **{clean_res['freed_mb']} MB disk space recovered**.")
            st.rerun()

    st.markdown("""
    <div style="background: rgba(14, 20, 36, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 9px; padding: 12px 16px; margin-top: 12px; font-size: 0.85rem; color: #94A3B8;">
        <b style="color: #38BDF8;">⚡ Why does Render take time to load on first click?</b><br>
        On Render's free tier, servers <b>spin down to 0 after 15 minutes of idle time</b>. When someone visits, Render cold-boots the container (takes ~45–60s).<br>
        To prevent cold boots and keep your Render web app <b>permanently active 24/7</b>, set up a free 10-minute HTTP ping using <a href="https://cron-job.org" target="_blank" style="color: #00F0FF;">cron-job.org</a> or <a href="https://uptimerobot.com" target="_blank" style="color: #00F0FF;">UptimeRobot</a> pointing to your Render deployment URL.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔒 Security & Privacy Guarantees")
    st.markdown("""
    - **No Arbitrary Code Execution:** The AI model cannot execute arbitrary Python or shell code. It calls controlled, sandboxed backend tools.
    - **DuckDB Analytical Guardrail:** All destructive SQL commands (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`) are strictly blocked before execution.
    - **Data Isolation:** Uploaded data is stored locally in `./data/`. Raw datasets are never overwritten.
    - **Secrets Protection:** API credentials, SMTP passwords, and `.env` files are permanently ignored and never pushed to GitHub.
    - **Neon SSL Encryption:** All connections to Neon Serverless PostgreSQL enforce `sslmode=require` in-flight encryption.
    """)

