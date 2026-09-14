import time
import streamlit as st
import pandas as pd
from ui.components import apply_custom_css, render_top_navbar

# Page Config
st.set_page_config(
    page_title="AI Data Analyst Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme state (default: dark_futuristic)
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark_futuristic"

# Apply theme styling dynamically based on current theme state
apply_custom_css(theme=st.session_state["theme"])

# Initialize authentication state - strict security gate (unauthenticated by default)
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# Navigation definition matching exact product spec
WORKSPACE_ITEMS = [
    ("🏠 Dashboard", "Dashboard"),
    ("📂 Upload Dataset", "Upload Dataset")
]

ANALYSIS_ITEMS = [
    ("📋 Data Profile", "Data Profile"),
    ("🛡️ Data Quality", "Data Quality"),
    ("🧹 Data Cleaning", "Data Cleaning"),
    ("🔬 EDA Engine", "EDA"),
    ("⚙️ Feature Engineering", "Feature Engineering"),
    ("🤖 ML & Predictive Analytics", "Machine Learning")
]

INTELLIGENCE_ITEMS = [
    ("⚡ SQL Analytics", "SQL Analytics"),
    ("✨ AI Insights", "AI Insights"),
    ("💬 Chat with Data", "AI Chat")
]

OUTPUT_ITEMS = [
    ("📈 Dashboard Builder", "Dashboard Builder"),
    ("📄 Reports", "Reports"),
    ("⬇ Downloads", "Downloads")
]

MANAGE_ITEMS = [
    ("📁 Projects", "Projects"),
    ("⚙ Settings", "Settings")
]

ALL_NAV_MAP = {}
for group in [WORKSPACE_ITEMS, ANALYSIS_ITEMS, INTELLIGENCE_ITEMS, OUTPUT_ITEMS, MANAGE_ITEMS]:
    for label, route in group:
        ALL_NAV_MAP[label] = route

# Check authentication state
is_auth = bool(st.session_state.get("authenticated", False))
user = st.session_state.get("user") or {}

# Demo session 2-minute expiration gate
if is_auth and st.session_state.get("is_demo_session", False):
    now_ts = time.time()
    demo_start = st.session_state.get("demo_start_time", now_ts)
    demo_duration = st.session_state.get("demo_duration_seconds", 120)
    elapsed = now_ts - demo_start
    rem_secs = max(0, int(demo_duration - elapsed))

    if rem_secs <= 0:
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.session_state["is_demo_session"] = False
        st.session_state["df"] = None
        st.session_state["raw_df"] = None
        st.session_state["current_project_id"] = None
        st.session_state["current_project_name"] = None
        st.session_state["active_nav"] = "Auth"
        st.toast("⏳ Demo session expired (2 minutes completed). Please log in again.", icon="⚠️")
        st.rerun()

# Initialize session state defaults
if not is_auth:
    st.session_state["active_nav"] = "Auth"
elif "active_nav" not in st.session_state or st.session_state["active_nav"] == "Auth":
    st.session_state["active_nav"] = "Dashboard"

def render_sidebar_demo_countdown():
    if not st.session_state.get("is_demo_session", False):
        return

    now_ts = time.time()
    demo_start = st.session_state.get("demo_start_time", now_ts)
    duration = st.session_state.get("demo_duration_seconds", 120)
    elapsed = now_ts - demo_start
    rem_secs = max(0, int(duration - elapsed))

    if rem_secs <= 0:
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.session_state["is_demo_session"] = False
        st.session_state["demo_start_time"] = None
        st.session_state["df"] = None
        st.session_state["raw_df"] = None
        st.session_state["current_project_id"] = None
        st.session_state["current_project_name"] = None
        st.session_state["active_nav"] = "Auth"
        st.toast("⏳ Demo session expired (2 minutes completed). Automatically logged out.", icon="⚠️")
        st.rerun()

    m_cnt = rem_secs // 60
    s_cnt = rem_secs % 60
    progress_pct = max(0.0, min(1.0, rem_secs / duration))
    color = "#EF4444" if rem_secs <= 30 else "#F59E0B"

    timer_html = (
        f'<div id="demo-pass-timer-box" style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(245, 158, 11, 0.16) 100%); '
        f'border: 1px solid {color}; border-radius: 10px; padding: 10px 12px; margin-bottom: 12px; '
        f'box-shadow: 0 0 16px rgba(239, 68, 68, 0.22);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
        f'<span style="font-size: 0.72rem; font-weight: 800; color: {color}; letter-spacing: 0.05em; display: flex; align-items: center; gap: 5px;">'
        f'<span style="width: 7px; height: 7px; border-radius: 50%; background: {color}; box-shadow: 0 0 8px {color}; display: inline-block;"></span>'
        f'DEMO PASS ACTIVE'
        f'</span>'
        f'<span id="demo-timer-display" style="font-family: monospace; font-size: 1rem; font-weight: 800; color: #FFFFFF; background: rgba(0,0,0,0.55); padding: 2px 8px; border-radius: 6px; border: 1px solid {color};">'
        f'{m_cnt:02d}:{s_cnt:02d}'
        f'</span>'
        f'</div>'
        f'<div style="width: 100%; height: 4px; background: rgba(255,255,255,0.1); border-radius: 99px; overflow: hidden; margin-top: 8px;">'
        f'<div id="demo-timer-progress" style="width: {int(progress_pct * 100)}%; height: 100%; background: {color}; transition: width 1s linear;"></div>'
        f'</div>'
        f'<div style="font-size: 0.72rem; color: #FCA5A5; margin-top: 5px; line-height: 1.3;">'
        f'⏳ Auto-logout in <b>{rem_secs}s</b> (2-minute limit).'
        f'</div>'
        f'</div>'
    )
    st.markdown(timer_html, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 4px 14px 4px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, #00F0FF 0%, #8B5CF6 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 0 14px rgba(0, 240, 255, 0.4); font-size: 1.1rem;">
                ⚡
            </div>
            <div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; line-height: 1.15;">
                    QUANTUM ANALYST
                </div>
                <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 600; letter-spacing: 0.02em;">
                    AI Powered Data Analytics
                </div>
            </div>
        </div>
    </div>
    <hr style='border-color: rgba(255, 255, 255, 0.08); margin: 6px 0 12px 0;'>
    """, unsafe_allow_html=True)

    # Reverse Countdown Timer on side of page (live 1s auto-updating fragment)
    if is_auth and st.session_state.get("is_demo_session", False):
        render_sidebar_demo_countdown()

    # Current dataset badge
    curr_proj = st.session_state.get("current_project_name")
    df = st.session_state.get("df")
    if is_auth and curr_proj and df is not None:
        rows_cnt = len(df)
        st.markdown(f"""
        <div style="background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 9px 12px; margin-bottom: 12px;">
            <div style="font-size: 0.68rem; font-weight: 700; color: #00F0FF; letter-spacing: 0.05em;">// ACTIVE WORKSPACE</div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.88rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{curr_proj}</div>
            <div style="font-size: 0.74rem; color: #94A3B8;">{rows_cnt:,} rows loaded</div>
        </div>
        """, unsafe_allow_html=True)

    if is_auth:
        active_route = st.session_state.get("active_nav", "Dashboard")

        # WORKSPACE
        st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; margin: 8px 0 4px 4px;'>WORKSPACE</div>", unsafe_allow_html=True)
        for label, route in WORKSPACE_ITEMS:
            is_active = (active_route == route)
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"{label}", key=f"nav_{route}", use_container_width=True, type=btn_type):
                st.session_state["active_nav"] = route
                st.rerun()

        # DATA ANALYSIS
        st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.08em; margin: 12px 0 5px 6px;'>DATA ANALYSIS</div>", unsafe_allow_html=True)
        for label, route in ANALYSIS_ITEMS:
            is_active = (active_route == route)
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"{label}", key=f"nav_{route}", use_container_width=True, type=btn_type):
                st.session_state["active_nav"] = route
                st.rerun()

        # INTELLIGENCE
        st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; margin: 10px 0 4px 4px;'>INTELLIGENCE</div>", unsafe_allow_html=True)
        for label, route in INTELLIGENCE_ITEMS:
            is_active = (active_route == route)
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"{label}", key=f"nav_{route}", use_container_width=True, type=btn_type):
                st.session_state["active_nav"] = route
                st.rerun()

        # OUTPUT
        st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; margin: 10px 0 4px 4px;'>OUTPUT</div>", unsafe_allow_html=True)
        for label, route in OUTPUT_ITEMS:
            is_active = (active_route == route)
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"{label}", key=f"nav_{route}", use_container_width=True, type=btn_type):
                st.session_state["active_nav"] = route
                st.rerun()

        # MANAGE
        st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; margin: 10px 0 4px 4px;'>MANAGE</div>", unsafe_allow_html=True)
        for label, route in MANAGE_ITEMS:
            is_active = (active_route == route)
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"{label}", key=f"nav_{route}", use_container_width=True, type=btn_type):
                st.session_state["active_nav"] = route
                st.rerun()

        st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.08); margin: 14px 0 10px 0;'>", unsafe_allow_html=True)

        # Authenticated user card at bottom
        user_name = user.get("full_name") or user.get("username") or "Analyst"
        user_role = user.get("role") or "Analyst"
        
        st.markdown(f"""
        <div style="background: rgba(14, 20, 36, 0.85); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 9px; padding: 10px 12px; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 700; color: #10B981;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #10B981; box-shadow: 0 0 6px #10B981; display: inline-block;"></span>
                AUTHENTICATED
            </div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.88rem; margin-top: 2px;">{user_name}</div>
            <div style="font-size: 0.74rem; color: #94A3B8;">Role: {user_role}</div>
        </div>
        """, unsafe_allow_html=True)

        col_signout, col_profile = st.columns([1, 1])
        with col_signout:
            if st.button("🚪 Sign Out", key="sb_btn_signout", type="secondary", use_container_width=True):
                st.session_state["authenticated"] = False
                st.session_state["user"] = None
                st.session_state["is_demo_session"] = False
                st.session_state["demo_start_time"] = None
                st.session_state["df"] = None
                st.session_state["raw_df"] = None
                st.session_state["current_project_id"] = None
                st.session_state["current_project_name"] = None
                st.session_state["active_nav"] = "Auth"
                st.rerun()
        with col_profile:
            if st.button("⚙ Account", key="sb_btn_acct", type="secondary", use_container_width=True):
                st.session_state["active_nav"] = "Settings"
                st.rerun()

        # PRO PLAN Box
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(2, 132, 199, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%); 
                    border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 9px; padding: 10px 12px; margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.75rem; font-weight: 800; color: #38BDF8; letter-spacing: 0.04em;">PRO PLAN</span>
                <span style="background: rgba(0, 240, 255, 0.2); color: #00F0FF; font-size: 0.65rem; font-weight: 700; padding: 1px 6px; border-radius: 4px;">ACTIVE</span>
            </div>
            <div style="font-size: 0.74rem; color: #CBD5E1; margin: 3px 0 6px 0;">Unlock advanced analytics & neural reasoning</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Upgrade Plan", key="sb_btn_upgrade", type="secondary", use_container_width=True):
            st.toast("You are currently on the Enterprise Pro tier with unlimited DuckDB in-memory executions.")
    else:
        # Unauthenticated state in sidebar
        st.markdown("""
        <div style="background: rgba(255, 0, 85, 0.08); border: 1px solid rgba(255, 0, 85, 0.3); border-radius: 10px; padding: 12px 14px; margin-bottom: 16px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #FF0055; font-weight: 700;">
                🔒 ACCESS RESTRICTED
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">
                Authentication is required to enter workspace and upload datasets.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔑 Account Sign In / Sign Up", key="sb_btn_auth_login", type="primary", use_container_width=True):
            st.session_state["active_nav"] = "Auth"
            st.rerun()


    st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.15); margin: 12px 0;'>", unsafe_allow_html=True)
    
    # System Status & Info
    with st.expander("System Architecture & Engine", expanded=False):
        st.markdown("""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94A3B8;">
            <b>Theme:</b> <span style="color: #00F0FF;">Futuristic Cyber Dark</span><br>
            <b>Engine:</b> DuckDB In-Memory<br>
            <b>Cache:</b> Sub-Millisecond Active<br>
            <b>Mode:</b> High Performance (Fast-Path)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<small style='color: #64748B;'>Enterprise Grade Analytics Platform<br>DuckDB • Scikit-Learn • FastAPI</small>", unsafe_allow_html=True)

# High-Performance Lazy View Dispatcher
def dispatch_active_view(route: str):
    if route == "Dashboard":
        from ui.views.dashboard_view import render_dashboard_view
        render_dashboard_view()
    elif route == "Upload Dataset":
        from ui.views.upload_view import render_upload_view
        render_upload_view()
    elif route == "Data Profile":
        from ui.views.profile_view import render_profile_view
        render_profile_view()
    elif route == "Data Quality":
        from ui.views.quality_view import render_quality_view
        render_quality_view()
    elif route == "Data Cleaning":
        from ui.views.cleaning_view import render_cleaning_view
        render_cleaning_view()
    elif route == "EDA":
        from ui.views.eda_view import render_eda_view
        render_eda_view()
    elif route == "Feature Engineering":
        from ui.views.feature_view import render_feature_view
        render_feature_view()
    elif route == "Machine Learning":
        from ui.views.ml_view import render_ml_view
        render_ml_view()
    elif route == "SQL Analytics":
        from ui.views.sql_view import render_sql_view
        render_sql_view()
    elif route == "AI Insights":
        from ui.views.insights_view import render_insights_view
        render_insights_view()
    elif route == "Dashboard Builder":
        from ui.views.builder_view import render_builder_view
        render_builder_view()
    elif route == "AI Chat":
        from ui.views.chat_view import render_chat_view
        render_chat_view()
    elif route == "Reports":
        from ui.views.reports_view import render_reports_view
        render_reports_view()
    elif route == "Downloads":
        from ui.views.downloads_view import render_downloads_view
        render_downloads_view()
    elif route == "Projects":
        from ui.views.projects_view import render_projects_view
        render_projects_view()
    elif route == "Settings":
        from ui.views.settings_view import render_settings_view
        render_settings_view()
    else:
        from ui.views.auth_view import render_auth_view
        render_auth_view()

# View Dispatcher: Strict Auth Gate
# If not authenticated, ALWAYS render only the Auth view regardless of any other parameter
if not is_auth:
    from ui.views.auth_view import render_auth_view
    render_auth_view()
else:
    active = st.session_state.get("active_nav", "Dashboard")
    curr_proj = st.session_state.get("current_project_name")
    df = st.session_state.get("df")

    # Clean SaaS top navbar with interactive Dark/Light toggle, Notifications & Profile
    render_top_navbar(
        breadcrumb=f"Workspace / {active}",
        active_dataset=curr_proj if df is not None else None
    )

    dispatch_active_view(active)

