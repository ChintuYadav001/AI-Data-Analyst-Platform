"""
Institutional Authentication & OTP Verification Modal Component.
Surfaces a secure interactive modal dialog when guests attempt restricted operations.
"""
from typing import Optional
import streamlit as st
from core.models.auth_model import AuthModel
from core.services.notification_service import mask_destination

auth_model = AuthModel()


def render_auth_modal(title: str = "Identity & Access Clearance Required", key_prefix: str = "modal"):
    """
    Renders an inline or modal authentication interface with OTP verification.
    """
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid #00F0FF; box-shadow: 0 0 25px rgba(0, 240, 255, 0.25); border-radius: 12px; padding: 24px; margin: 16px 0;">
        <div style="display: flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #00F0FF;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: #00F0FF; box-shadow: 0 0 8px #00F0FF; display: inline-block;"></span>
            // CLEARANCE REQUIRED &bull; INSTITUTIONAL ACCESS CONTROL
        </div>
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 6px 0;">
            {title}
        </div>
        <div style="font-size: 0.85rem; color: #94A3B8;">
            Sign in to your analyst account or register with One-Time Password (OTP) verification to upload datasets and access quantum analytics pipelines.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_signin, tab_signup, tab_reset = st.tabs(["🔑 Sign In", "📝 Sign Up with OTP Verification", "🔄 Reset Password"])

    # 1. Sign In Tab
    with tab_signin:
        with st.form(f"{key_prefix}_signin_form", clear_on_submit=False):
            username_or_email = st.text_input("Username or Email", placeholder="e.g. analyst or analyst@company.com", key=f"{key_prefix}_login_id")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key=f"{key_prefix}_login_pwd")
            login_btn = st.form_submit_button("🔓 Authenticate & Access Platform", type="primary", width="stretch")

            if login_btn:
                if not username_or_email or not password:
                    st.error("Please enter both username/email and password.")
                else:
                    user = auth_model.authenticate(
                        username_or_email=username_or_email,
                        password=password,
                        ip_address="127.0.0.1",
                        user_agent="Streamlit-Modal"
                    )
                    if user:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user
                        st.success(f"Clearance granted! Welcome, {user.get('full_name') or user.get('username')}.")
                        st.rerun()
                    else:
                        st.error("Authentication failed. Invalid credentials. Event logged to security audit ledger.")

    # 2. Sign Up Tab with OTP Verification
    with tab_signup:
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            reg_username = st.text_input("Username", placeholder="e.g. quantum_analyst", key=f"{key_prefix}_reg_uname")
            reg_fullname = st.text_input("Full Name", placeholder="e.g. Dr. Alex Vance", key=f"{key_prefix}_reg_fname")
        with col_reg2:
            reg_email = st.text_input("Institutional Email", placeholder="e.g. alex@vance.io", key=f"{key_prefix}_reg_email")
            reg_role = st.selectbox(
                "Role Clearance",
                ["Lead Data Scientist", "Senior Quantitative Analyst", "Executive BI Director", "Data Engineer"],
                key=f"{key_prefix}_reg_role"
            )

        col_pwd1, col_pwd2 = st.columns(2)
        with col_pwd1:
            reg_password = st.text_input("Password", type="password", placeholder="Minimum 4 characters", key=f"{key_prefix}_reg_pwd")
        with col_pwd2:
            reg_password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key=f"{key_prefix}_reg_pwd2")

        # OTP Dispatch Subsystem
        otp_state_key = f"{key_prefix}_active_otp"
        otp_email_key = f"{key_prefix}_otp_sent_to"

        col_otp_btn, col_otp_inp = st.columns([1, 1])
        with col_otp_btn:
            if st.button("📲 Send OTP / Verification Link", key=f"{key_prefix}_btn_send_otp", width="stretch"):
                if not reg_email:
                    st.error("Please provide an institutional email address first.")
                elif "@" not in reg_email or "." not in reg_email:
                    st.error("Please provide a valid email format.")
                else:
                    try:
                        auth_model.generate_otp(reg_email, target_type="email")
                        st.session_state[otp_email_key] = reg_email
                        st.toast(f"OTP code and 1-click link sent directly to {reg_email}!", icon="📬")
                    except Exception as e:
                        st.error(f"Failed to generate OTP: {str(e)}")

        sent_email = st.session_state.get(otp_email_key)

        if sent_email and sent_email == reg_email:
            masked_email = mask_destination(sent_email, "email")
            last_dispatch = auth_model.get_last_dispatch_info() or {}
            dispatched_live = last_dispatch.get("dispatched_live", False)
            sent_code = last_dispatch.get("otp_code", "")

            if dispatched_live:
                st.markdown(f"""
                <div style="background: rgba(0, 255, 157, 0.08); border: 1px solid rgba(0, 255, 157, 0.35); border-radius: 8px; padding: 14px; margin: 10px 0;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #00FF9D; font-weight: 700;">
                        ✓ LIVE MESSAGE DISPATCHED TO: {masked_email}
                    </div>
                    <div style="font-size: 0.85rem; color: #FFFFFF; margin-top: 4px; line-height: 1.4;">
                        Your verification code was delivered to your email inbox. Please check your inbox. (Valid for 10 minutes)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(255, 170, 0, 0.08); border: 1px solid rgba(255, 170, 0, 0.4); border-radius: 8px; padding: 12px; margin: 8px 0;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #FFAA00; font-weight: 700;">
                        ⚠️ LIVE SMTP SENDER NOT YET CONFIGURED
                    </div>
                    <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 4px; line-height: 1.4;">
                        Verification codes are protected and never displayed on screen. Please configure your SMTP server credentials in the Auth settings to receive emails in your real inbox.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        user_entered_otp = st.text_input("Enter 6-Digit Verification OTP (from Email)", placeholder="e.g. 847291", max_chars=6, key=f"{key_prefix}_input_otp")

        if st.button("🛡️ Verify OTP & Complete Registration", type="primary", width="stretch", key=f"{key_prefix}_btn_register"):
            if not reg_username or not reg_email or not reg_password:
                st.error("Username, Email, and Password are required.")
            elif reg_password != reg_password_confirm:
                st.error("Passwords do not match.")
            elif len(reg_password) < 4:
                st.error("Password must be at least 4 characters.")
            elif not user_entered_otp:
                st.error("Please enter the 6-digit One-Time Password (OTP).")
            else:
                try:
                    registered_user = auth_model.register(
                        username=reg_username,
                        email=reg_email,
                        password=reg_password,
                        full_name=reg_fullname,
                        role=reg_role,
                        otp_code=user_entered_otp,
                        require_otp=True
                    )
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = registered_user
                    # Clean up OTP state
                    if otp_state_key in st.session_state:
                        del st.session_state[otp_state_key]
                    st.success(f"Account verified and registered! Welcome, {registered_user['username']}.")
                    st.rerun()
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"Registration error: {str(e)}")

    # 3. Reset Password Tab with OTP Verification
    with tab_reset:
        from ui.views.auth_view import render_password_reset_form
        render_password_reset_form(auth_model, key_prefix=f"{key_prefix}_reset", show_back_btn=False)

