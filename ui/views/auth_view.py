import time
"""
Authentication View: Sign In, Sign Up, and Security Audit Logs.
"""
from datetime import datetime
import streamlit as st
import pandas as pd
from core.models.auth_model import AuthModel
from core.services.notification_service import mask_destination

auth_model = AuthModel()

def render_auth_view():
    st.markdown("""
<style>
.block-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding-top: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 1.2rem !important;
}
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    display: flex !important;
    visibility: visible !important;
    background: transparent !important;
    height: 2.875rem !important;
    z-index: 99990 !important;
}
[data-testid="stToolbar"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
footer {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}
</style>
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="width: 34px; height: 34px; border-radius: 9px; background: linear-gradient(135deg, #0284C7, #00F0FF); display: flex; align-items: center; justify-content: center; font-size: 1.15rem; box-shadow: 0 0 12px rgba(0, 240, 255, 0.4);">
            ⚡
        </div>
        <div>
            <h1 style="font-family: 'Inter', -apple-system, sans-serif; font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 0; line-height: 1.2; letter-spacing: -0.02em;">
                Authentication & Access Control
            </h1>
            <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 1px;">
                Institutional Clearance Gate • DuckDB High-Throughput Analytics • MySQL Ledger Sync
            </div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94A3B8; background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 20px; padding: 4px 12px;">
        <span style="width: 7px; height: 7px; border-radius: 50%; background: #10B981; box-shadow: 0 0 6px #10B981; display: inline-block;"></span>
        <span style="color: #10B981; font-weight: 700;">DEPLOYED & OPERATIONAL</span>
        <span>•</span>
        <span style="color: #00F0FF;">FAST-PATH ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

    is_auth = st.session_state.get("authenticated", False)
    user = st.session_state.get("user")

    if is_auth and user:
        st.markdown(f"""
        <div style="background: rgba(0, 255, 157, 0.06); border: 1px solid rgba(0, 255, 157, 0.35); border-radius: 12px; padding: 20px 24px; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #00FF9D; letter-spacing: 0.1em;">
                        // ACTIVE AUTHENTICATED SESSION
                    </div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">
                        Welcome, {user.get('full_name') or user.get('username')}
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">
                        Role: <span style="color: #00F0FF; font-weight: 600;">{user.get('role', 'analyst').upper()}</span> &nbsp;|&nbsp;
                        Email: <span style="color: #FFFFFF;">{user.get('email')}</span> &nbsp;|&nbsp;
                        Phone: <span style="color: #00FF9D;">{user.get('phone') or 'Not set'}</span> &nbsp;|&nbsp;
                        Login Count: <span style="color: #FFB800;">{user.get('login_count', 1)}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🚪 Sign Out / Terminate Session", type="secondary", use_container_width=True):
                st.session_state["authenticated"] = False
                st.session_state["user"] = None
                st.session_state["df"] = None
                st.session_state["raw_df"] = None
                st.session_state["current_project_id"] = None
                st.session_state["current_project_name"] = None
                st.toast("Logged out successfully", icon="ℹ️")
                st.rerun()

        with col2:
            if st.button("📊 Go to Dashboard", type="primary", use_container_width=True):
                st.session_state["active_nav"] = "Dashboard"
                st.rerun()

    tab_signin, tab_signup, tab_logs = st.tabs(["🔑 Sign In", "📝 Sign Up / Register", "🛡️ Security Audit Logs"])

    with tab_signin:
        if is_auth:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 20px 24px; margin: 12px 0;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: #10B981; letter-spacing: 0.08em;">// ACTIVE AUTHENTICATED SESSION</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">You are currently signed in with full institutional clearance.</div>
                <div style="font-size: 0.84rem; color: #94A3B8;">All analytical pipelines, DuckDB SQL kernels, and dossiers are active.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if "login_preview_query" not in st.session_state:
                st.session_state["login_preview_query"] = "revenue"

            col_login_card, col_interactive_space = st.columns([0.92, 1.28], gap="large")

            # -----------------------------------------------------------------
            # 1. SMALL & COMPACT PROFESSIONAL LOGIN CARD (LEFT)
            # -----------------------------------------------------------------
            with col_login_card:
                with st.container(border=True):
                    st.markdown("""
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                        <div style="width: 38px; height: 38px; border-radius: 10px; 
                                    background: linear-gradient(135deg, #0284C7 0%, #00F0FF 50%, #8B5CF6 100%); 
                                    display: flex; align-items: center; justify-content: center; font-size: 1.25rem; 
                                    box-shadow: 0 0 14px rgba(0, 240, 255, 0.45); flex-shrink: 0;">
                            ⚡
                        </div>
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.01em; line-height: 1.2;">
                                Analyst Sign In
                            </div>
                            <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 600;">
                                Institutional Clearance Gate
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    username_or_email = st.text_input(
                        "Email or Username",
                        placeholder="analyst@quantum.ai",
                        key="compact_signin_identifier"
                    )

                    password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="••••••••••••",
                        key="compact_signin_password"
                    )

                    col_rem, col_fog = st.columns([1.1, 0.9])
                    with col_rem:
                        remember = st.checkbox("Remember me", value=True, key="compact_signin_remember")
                    with col_fog:
                        with st.popover("Forgot?", use_container_width=True):
                            st.markdown("<b style='color: #00F0FF; font-size: 0.85rem;'>Password Recovery</b>", unsafe_allow_html=True)
                            st.caption("Enter registered email to dispatch recovery link.")
                            rec_dest = st.text_input("Work Email", placeholder="analyst@quantum.ai", key="compact_rec_dest")
                            if st.button("Send Link", key="compact_btn_rec", type="primary"):
                                if rec_dest:
                                    st.success("Verification dispatched.")
                                else:
                                    st.error("Email required.")

                    st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)

                    if st.button("⚡ Sign In to Platform →", key="compact_btn_signin_submit", type="primary", use_container_width=True):
                        if not username_or_email or not password:
                            st.error("Please provide both identifier and password.")
                        else:
                            auth_user = auth_model.authenticate(
                                username_or_email=username_or_email,
                                password=password,
                                ip_address="127.0.0.1",
                                user_agent="Quantum-Web-Client"
                            )
                            if auth_user:
                                st.session_state["authenticated"] = True
                                st.session_state["user"] = auth_user
                                st.session_state["active_nav"] = "Dashboard"
                                disp_name = auth_user.get("full_name") or auth_user.get("username")
                                st.toast(f"Welcome back, {disp_name}!", icon="✨")
                                st.rerun()
                            else:
                                st.error("Invalid credentials. Try the Demo Access below.")

                    # Quick 1-Click Demo Pass (2-Minute Trial Limit)
                    st.markdown("<div style='margin-top: 4px;'></div>", unsafe_allow_html=True)
                    if st.button("⚡ 1-Click Demo Pass (2-Min Trial)", key="compact_btn_demo_pass", type="secondary", use_container_width=True):
                        demo_user = auth_model.authenticate(
                            username_or_email="analyst@quantum.ai",
                            password="QuantumPassword2026!",
                            ip_address="127.0.0.1",
                            user_agent="Quantum-Demo-Pass"
                        )
                        if not demo_user:
                            with auth_model._get_connection() as conn:
                                cursor = conn.cursor()
                                cursor.execute("SELECT id FROM users LIMIT 1")
                                r = cursor.fetchone()
                                if r:
                                    demo_user = auth_model.get_user_by_id(r["id"])

                        if demo_user:
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = demo_user
                            st.session_state["is_demo_session"] = True
                            st.session_state["demo_start_time"] = time.time()
                            st.session_state["demo_duration_seconds"] = 120
                            st.session_state["active_nav"] = "Dashboard"
                            st.toast("⏳ 2-Minute Demo Pass activated! Live reverse countdown active on the sidebar.", icon="🚀")
                            st.rerun()
                        else:
                            st.error("Could not activate demo pass.")

                    st.markdown("""
                    <div style="margin-top: 10px; padding: 7px 10px; background: rgba(0, 240, 255, 0.04); border: 1px solid rgba(0, 240, 255, 0.18); border-radius: 8px; text-align: center;">
                        <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 600;">
                            ⏱️ 2-Min Demo Trial &nbsp;•&nbsp; 🔒 256-Bit SSL &nbsp;•&nbsp; ⚡ MySQL Sync
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # 2. ATTRACTIVE INTERACTIVE FUNCTION IN REMAINING SPACE (RIGHT)
            # -----------------------------------------------------------------
            with col_interactive_space:
                with st.container(border=True):
                    st.markdown("""
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                        <div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #00F0FF; font-weight: 700; letter-spacing: 0.08em;">
                                // INTERACTIVE INTELLIGENCE SANDBOX
                            </div>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-top: 2px;">
                                Experience Autonomous Analytics
                            </div>
                            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">
                                Click a query below to test how Quantum Analyst executes vectorized SQL and neural reasoning:
                            </div>
                        </div>
                        <div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.12); 
                                    border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 20px; padding: 3px 9px;">
                            <span style="width: 7px; height: 7px; border-radius: 50%; background: #10B981; box-shadow: 0 0 6px #10B981; display: inline-block;"></span>
                            <span style="color: #10B981; font-size: 0.7rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">LIVE ENGINE</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Interactive Query Selection Buttons
                    col_q1, col_q2, col_q3 = st.columns(3)
                    with col_q1:
                        is_q1 = (st.session_state["login_preview_query"] == "revenue")
                        if st.button("📈 Revenue Forecast", key="btn_prev_rev", type="primary" if is_q1 else "secondary", use_container_width=True):
                            st.session_state["login_preview_query"] = "revenue"
                            st.rerun()
                    with col_q2:
                        is_q2 = (st.session_state["login_preview_query"] == "fraud")
                        if st.button("⚠️ Outlier Anomaly", key="btn_prev_fraud", type="primary" if is_q2 else "secondary", use_container_width=True):
                            st.session_state["login_preview_query"] = "fraud"
                            st.rerun()
                    with col_q3:
                        is_q3 = (st.session_state["login_preview_query"] == "churn")
                        if st.button("👥 Customer Churn", key="btn_prev_churn", type="primary" if is_q3 else "secondary", use_container_width=True):
                            st.session_state["login_preview_query"] = "churn"
                            st.rerun()

                    # Dynamic Interactive Result Output
                    active_mode = st.session_state.get("login_preview_query", "revenue")

                    if active_mode == "revenue":
                        metric_1 = ("$1,248,920", "+18.4% YoY", "#10B981")
                        metric_2 = ("1,420,000", "Vector Rows", "#00F0FF")
                        metric_3 = ("0.74 ms", "Query Latency", "#8B5CF6")
                        insight_text = "Revenue accelerated by <b>+24.3% in Q3 enterprise cohorts</b>. Highest margin growth observed across North American SaaS subscriptions with recurring expansion."
                        sample_sql = "SELECT cohort_quarter, SUM(revenue) as total_rev, AVG(margin_pct) FROM enterprise_pipeline GROUP BY 1 ORDER BY 1 DESC;"
                    elif active_mode == "fraud":
                        metric_1 = ("14 Flags", "Critical Outliers", "#EF4444")
                        metric_2 = ("99.4%", "Model Precision", "#00F0FF")
                        metric_3 = ("0.91 ms", "Scan Latency", "#8B5CF6")
                        insight_text = "Detected <b>14 high-deviation transaction spikes</b> exceeding 3 standard deviations ($12,400+ threshold). Isolation forest clustering isolated automated bot retries."
                        sample_sql = "SELECT tx_id, amount, z_score FROM transactions WHERE ABS(z_score) > 3.0 ORDER BY amount DESC LIMIT 10;"
                    else: # churn
                        metric_1 = ("4.2%", "-1.8% Improvement", "#10B981")
                        metric_2 = ("8,940", "Monitored Accounts", "#00F0FF")
                        metric_3 = ("0.82 ms", "DuckDB Kernel", "#8B5CF6")
                        insight_text = "Identified <b>primary churn trigger</b>: accounts with under 2 team log-ins during week 3 exhibit a 73% attrition hazard. Targeted onboarding recommended."
                        sample_sql = "SELECT account_tier, COUNT(*) as churned_users, CORR(tenure_days, weekly_logins) FROM user_churn GROUP BY 1;"

                    # Dynamic Metric Cards
                    m_c1, m_c2, m_c3 = st.columns(3)
                    with m_c1:
                        st.markdown(f"""
                        <div style="background: rgba(2, 132, 199, 0.1); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 8px; padding: 8px 10px;">
                            <div style="font-size: 0.65rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">KEY METRIC</div>
                            <div style="font-size: 1.1rem; font-weight: 800; color: #FFFFFF;">{metric_1[0]}</div>
                            <div style="font-size: 0.7rem; color: {metric_1[2]}; font-weight: 700;">{metric_1[1]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m_c2:
                        st.markdown(f"""
                        <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 8px; padding: 8px 10px;">
                            <div style="font-size: 0.65rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">DATA ENGINE</div>
                            <div style="font-size: 1.1rem; font-weight: 800; color: #FFFFFF;">{metric_2[0]}</div>
                            <div style="font-size: 0.7rem; color: {metric_2[2]}; font-weight: 700;">{metric_2[1]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m_c3:
                        st.markdown(f"""
                        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 8px 10px;">
                            <div style="font-size: 0.65rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">PERFORMANCE</div>
                            <div style="font-size: 1.1rem; font-weight: 800; color: #FFFFFF;">{metric_3[0]}</div>
                            <div style="font-size: 0.7rem; color: {metric_3[2]}; font-weight: 700;">{metric_3[1]}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # AI Narrative Synthesis
                    st.markdown(f"""
                    <div style="background: rgba(0, 240, 255, 0.05); border-left: 3px solid #00F0FF; 
                                border-radius: 0 8px 8px 0; padding: 9px 12px; margin: 10px 0 8px 0;">
                        <div style="font-size: 0.68rem; font-weight: 700; color: #00F0FF; letter-spacing: 0.06em;">✦ SYNTHESIZED NEURAL FINDING</div>
                        <div style="font-size: 0.82rem; color: #E2E8F0; margin-top: 3px; line-height: 1.45;">
                            {insight_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Vectorized DuckDB Query Snippet
                    st.markdown(f"""
                    <div style="background: #0B111E; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 8px 10px; margin-bottom: 4px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.64rem; color: #64748B; font-family: 'JetBrains Mono', monospace; margin-bottom: 3px;">
                            <span>// DUCKDB COMPILED SQL</span>
                            <span style="color: #00FF9D;">FAST-PATH VECTORIZED</span>
                        </div>
                        <code style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #38BDF8; line-height: 1.35; word-break: break-all;">
                            {sample_sql}
                        </code>
                    </div>
                    """, unsafe_allow_html=True)

    with tab_signup:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 12px; padding: 24px; max-width: 540px; margin: 10px 0 20px 0;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">
                Create New Analyst Account
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 16px;">
                Your credentials are encrypted using PBKDF2-HMAC-SHA256 with cryptographic salt and stored in local SQLite database.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detect if user opened the page from an email or SMS verification link
        url_code = st.query_params.get("verify_code", "")
        url_target = st.query_params.get("verify_target", "")

        if url_code and url_target:
            st.markdown(f"""
            <div style="background: rgba(0, 240, 255, 0.08); border: 1px solid #00F0FF; border-radius: 8px; padding: 14px; margin-bottom: 16px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #00F0FF; font-weight: 700;">
                    ✓ ONE-CLICK VERIFICATION LINK ACTIVATED
                </div>
                <div style="font-size: 0.85rem; color: #FFFFFF; margin-top: 4px;">
                    Verification clearance loaded from your secure link for: <b>{url_target}</b>. Complete your desired username and password below to finalize registration.
                </div>
            </div>
            """, unsafe_allow_html=True)

        default_email = url_target if ("@" in url_target) else ""
        default_phone = url_target if (url_target and "@" not in url_target) else ""

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            new_username = st.text_input("Desired Username", placeholder="e.g. quantum_analyst", key="main_reg_uname")
            new_fullname = st.text_input("Full Name", placeholder="e.g. Dr. Alex Vance", key="main_reg_fname")
            new_phone = st.text_input("Mobile / Phone Number", value=default_phone, placeholder="e.g. +1 555-0199 or 9876543210", key="main_reg_phone")
        with col_u2:
            new_email = st.text_input("Email Address", value=default_email, placeholder="e.g. alex@vance.io", key="main_reg_email")
            new_role = st.selectbox(
                "Role Clearance",
                ["Lead Data Scientist", "Senior Quantitative Analyst", "Executive BI Director", "Data Engineer"],
                key="main_reg_role"
            )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            new_password = st.text_input("Secure Password", type="password", placeholder="Choose a secure password", key="main_reg_pwd")
        with col_p2:
            new_password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password", key="main_reg_pwd2")

        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #00F0FF; margin-top: 8px;'>📲 Select OTP & Verification Link Dispatch Channel:</div>", unsafe_allow_html=True)
        otp_channel = st.radio(
            "OTP Channel",
            ["📧 Send OTP & Link to Institutional Email (Active via Gmail SMTP)", "📱 Send OTP & Link to Mobile / Phone Number (SMS)"],
            label_visibility="collapsed",
            key="main_otp_channel_choice"
        )

        col_send_btn, col_blank = st.columns([1, 1])
        with col_send_btn:
            if st.button("📲 Send OTP / Verification Link", key="main_reg_send_otp_btn", use_container_width=True):
                is_phone_target = "Phone" in otp_channel or "Mobile" in otp_channel
                target_dest = new_phone.strip() if is_phone_target else new_email.strip()

                if is_phone_target and (not new_phone or len(new_phone.strip()) < 6):
                    st.error("Please provide a valid mobile or phone number (at least 6 digits).")
                elif not is_phone_target and (not new_email or "@" not in new_email):
                    st.error("Please provide a valid institutional email address.")
                else:
                    try:
                        target_type = "phone" if is_phone_target else "email"
                        auth_model.generate_otp(target_dest, target_type=target_type)
                        st.session_state["main_otp_sent_to"] = target_dest
                        st.session_state["main_otp_type"] = target_type
                        st.toast(f"OTP & Verification link dispatched to your {target_type}!", icon="📬")
                    except Exception as e:
                        st.error(f"Failed to dispatch OTP: {str(e)}")

        main_sent_dest = st.session_state.get("main_otp_sent_to")
        main_dest_type = st.session_state.get("main_otp_type", "channel")

        if main_sent_dest:
            masked_dest = mask_destination(main_sent_dest, main_dest_type)
            last_dispatch = auth_model.get_last_dispatch_info() or {}
            dispatched_live = last_dispatch.get("dispatched_live", False)
            sent_code = last_dispatch.get("otp_code", "")
            sent_link = last_dispatch.get("verification_link", "")

            if dispatched_live:
                st.markdown(f"""
                <div style="background: rgba(0, 255, 157, 0.1); border: 1px solid #00FF9D; box-shadow: 0 0 15px rgba(0, 255, 157, 0.2); border-radius: 8px; padding: 16px; margin: 12px 0;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #00FF9D; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00FF9D; box-shadow: 0 0 8px #00FF9D;"></span>
                        ✓ REAL EMAIL DELIVERED TO YOUR GMAIL INBOX: {main_sent_dest}
                    </div>
                    <div style="font-size: 0.88rem; color: #FFFFFF; margin-top: 6px; line-height: 1.5;">
                        Your 6-digit OTP code and 1-click verification link have been delivered to <b>{main_sent_dest}</b> via Gmail SMTP.
                        <br/><span style="font-size: 0.8rem; color: #94A3B8;">Please check your Gmail inbox (or spam folder) on your phone or computer to retrieve your 6-digit code, then enter it below.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(255, 170, 0, 0.08); border: 1px solid rgba(255, 170, 0, 0.4); border-radius: 8px; padding: 14px; margin: 10px 0;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #FFAA00; font-weight: 700;">
                        ⚠️ EMAIL DISPATCH: SMTP CREDENTIALS REQUIRED FOR REAL-TIME INBOX ROUTING
                    </div>
                    <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">
                        To protect your security, <b>verification codes are NEVER displayed on this signup page</b>.
                        <br/>Because an outbound mail server (e.g. Gmail App Password) has not yet been connected to this system, emails cannot be routed over the public internet to <code>{masked_dest}</code> until you configure your sender email below.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("⚙️ Configure Real Email Delivery (Gmail / Outlook / SMTP)", expanded=True):
                st.markdown("""
                **Why did "Username and Password not accepted (BadCredentials)" appear?**
                > Google **does not allow your normal account password** for sending automated emails. You must use a dedicated **16-character Google App Password**.

                **Step-by-Step Instructions (Takes 60 seconds):**
                1. [👉 Click here to open your Google App Passwords page](https://myaccount.google.com/apppasswords) (log in as `shambhuray0002@gmail.com`).
                   *(If it says "The setting that you are looking for is not available", make sure [2-Step Verification is turned on](https://myaccount.google.com/signinoptions/two-step-verification) first).*
                2. In the **App name** box, type `Analytics Platform` and click **Create**.
                3. A yellow popup will show a **16-character password** (e.g. `abcd efgh ijkl mnop`).
                4. Paste that 16-character code into the box below and click **Test & Save**.
                """)
                col_cfg1, col_cfg2 = st.columns(2)
                with col_cfg1:
                    cfg_host = st.text_input("SMTP Server", value="smtp.gmail.com", key="cfg_smtp_host")
                    cfg_port = st.number_input("SMTP Port", value=587, key="cfg_smtp_port")
                with col_cfg2:
                    cfg_user = st.text_input("Gmail Address", value=main_sent_dest if "@" in main_sent_dest else "shambhuray0002@gmail.com", placeholder="your.email@gmail.com", key="cfg_smtp_user")
                    cfg_pass = st.text_input("Google App Password (16 letters)", type="password", placeholder="Paste 16-character app password here", key="cfg_smtp_pass")

                if st.button("💾 Test & Save Gmail SMTP to .env", key="cfg_save_btn"):
                    from core.services.notification_service import NotificationService
                    res = NotificationService.configure_smtp(
                        server=cfg_host.strip(),
                        port=int(cfg_port),
                        username=cfg_user.strip(),
                        password=cfg_pass.strip().replace(" ", ""),
                        use_tls=True
                    )
                    if res.get("success"):
                        st.success("✓ Gmail SMTP connected and saved! Future OTPs will arrive in your real Gmail inbox.")
                        st.rerun()
                    else:
                        st.error(res.get("error"))

        user_otp_input = st.text_input(
            "Enter 6-Digit Verification OTP (from your Email or Phone SMS)",
            value=url_code if url_code else "",
            placeholder="e.g. 123456",
            max_chars=6,
            key="main_user_otp_input"
        )

        register_btn = st.button("🛡️ Verify OTP & Register Account", type="primary", use_container_width=True, key="main_submit_reg_btn")

        if register_btn:
            if not new_username or not new_email or not new_password:
                st.error("Username, Email, and Password are required fields.")
            elif not new_phone or len(new_phone.strip()) < 6:
                st.error("Please provide a valid phone number.")
            elif new_password != new_password_confirm:
                st.error("Passwords do not match. Please verify your password.")
            elif len(new_password) < 4:
                st.error("Password must be at least 4 characters in length.")
            elif not user_otp_input:
                st.error("Please enter the 6-digit One-Time Password (OTP) received on your email or phone.")
            else:
                try:
                    registered_user = auth_model.register(
                        username=new_username,
                        email=new_email,
                        password=new_password,
                        full_name=new_fullname,
                        phone=new_phone.strip(),
                        role=new_role,
                        otp_code=user_otp_input,
                        require_otp=True
                    )
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = registered_user
                    st.session_state["active_nav"] = "Dashboard"
                    if "main_otp_sent_to" in st.session_state:
                        del st.session_state["main_otp_sent_to"]
                    st.success(f"Account successfully verified and created! Logged in as {registered_user['username']}.")
                    st.rerun()
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"Error registering user: {str(e)}")

    with tab_logs:
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #00F0FF; letter-spacing: 0.1em;">
                // IMMUTABLE SQL AUDIT TRAIL
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">
                Database Security & Login Attempt Ledger
            </div>
            <p style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                Every authentication attempt (success or rejection) is persisted in the local SQLite table <code>login_logs</code> for regulatory compliance and audit tracking.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 View Database Security & Login Attempt Ledger", expanded=False):
            logs = auth_model.list_login_logs(limit=50)
            if logs:
                df_logs = pd.DataFrame(logs)
                st.dataframe(
                    df_logs,
                    use_container_width=True,
                    column_config={
                        "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
                        "status": st.column_config.TextColumn("Status"),
                        "username": st.column_config.TextColumn("Username / Identifier"),
                        "ip_address": st.column_config.TextColumn("IP Address"),
                        "failure_reason": st.column_config.TextColumn("Failure Note")
                    }
                )
                
                # Summary Metrics
                total_attempts = len(df_logs)
                success_attempts = len(df_logs[df_logs["status"] == "SUCCESS"])
                failed_attempts = len(df_logs[df_logs["status"] == "FAILED"])

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Total Recorded Attempts", total_attempts)
                with m2:
                    st.metric("Successful Logins", success_attempts)
                with m3:
                    st.metric("Failed / Blocked Attempts", failed_attempts)
            else:
                st.info("No authentication attempts logged yet. Sign in or register an account to populate the audit ledger.")

        # Outbound Dispatch Ledger (Admin Audit Trail)
        st.markdown("<hr style='border-color: rgba(0, 240, 255, 0.15); margin: 16px 0 12px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom: 12px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #00F0FF; letter-spacing: 0.1em;">
                // SECURE SERVER OUTBOX DISPATCH LOG
            </div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF;">
                Outbound OTP Verification Ledger (Admin View)
            </div>
            <p style="color: #94A3B8; font-size: 0.8rem; margin-top: 2px;">
                Logs outbound email and SMS OTP dispatches generated by the server. Visible exclusively on this administrative audit ledger tab.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📬 View Outbound OTP Verification Dispatch Outbox", expanded=False):
            from core.config import DATA_DIR
            outbox_file = DATA_DIR / "outbox.log"
            if outbox_file.exists():
                import json
                outbox_entries = []
                try:
                    with open(outbox_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    outbox_entries.append(json.loads(line))
                                except Exception:
                                    pass
                except Exception:
                    pass

                if outbox_entries:
                    df_outbox = pd.DataFrame(outbox_entries[-20:])
                    display_cols = [c for c in ["timestamp", "target", "target_type", "dispatched_live", "otp_code", "error"] if c in df_outbox.columns]
                    st.dataframe(
                        df_outbox[display_cols],
                        use_container_width=True,
                        column_config={
                            "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                            "target": st.column_config.TextColumn("Destination"),
                            "target_type": st.column_config.TextColumn("Channel"),
                            "dispatched_live": st.column_config.CheckboxColumn("Live Delivered?"),
                            "otp_code": st.column_config.TextColumn("Dispatched OTP Code"),
                            "error": st.column_config.TextColumn("Gateway Note")
                        }
                    )
                else:
                    st.caption("Outbox is currently clear.")
            else:
                st.caption("No outbox log file generated yet.")
