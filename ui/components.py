import streamlit as st
from typing import Dict, Any, List, Optional

def apply_custom_css(theme: str = "dark_futuristic", *args, **kwargs):
    """Clean, premium, modern SaaS interface with dark and light theme palettes."""
    is_light = (theme == "light_classic")

    if is_light:
        st.markdown("""
        <style>
        /* Zero-Latency Native Font Stack */

        /* Full-Screen Edge-to-Edge Responsive Layout */
        .block-container {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            padding-top: 3.5rem !important;
            padding-bottom: 2rem !important;
        }

        /* Streamlit Header & Deployment Toolbar - Native & Non-Overlapping */
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

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            background-color: #F8FAFC !important;
            background-image: 
                radial-gradient(at 15% 10%, rgba(2, 132, 199, 0.03) 0px, transparent 40%),
                radial-gradient(at 85% 15%, rgba(99, 102, 241, 0.03) 0px, transparent 45%),
                linear-gradient(rgba(0, 0, 0, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 0, 0, 0.03) 1px, transparent 1px) !important;
            background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px !important;
            color: #0F172A !important;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
            font-weight: 700 !important;
            color: #0F172A !important;
            letter-spacing: -0.025em !important;
        }

        code, pre, .mono-code {
            font-family: 'JetBrains Mono', monospace !important;
        }

        p, span, label, .stMarkdown {
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        /* Streamlit Sidebar Styling - Light Clean Enterprise */
        section[data-testid="stSidebar"] {
            background: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            box-shadow: 4px 0 20px rgba(0, 0, 0, 0.03) !important;
            padding-top: 0.5rem;
        }
        
        section[data-testid="stSidebar"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        section[data-testid="stSidebar"] .stRadio label {
            color: #475569 !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            padding: 9px 12px !important;
            border-radius: 8px !important;
            border: 1px solid transparent !important;
            transition: all 0.18s ease-in-out !important;
            margin-bottom: 2px !important;
            cursor: pointer !important;
        }

        section[data-testid="stSidebar"] .stRadio label:hover {
            background: #F1F5F9 !important;
            color: #0F172A !important;
        }

        section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div,
        section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
            background: #E0F2FE !important;
            border: 1px solid #0284C7 !important;
            color: #0284C7 !important;
            font-weight: 600 !important;
        }

        /* Light Buttons */
        .stButton > button {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.88rem !important;
            border-radius: 9px !important;
            padding: 8px 16px !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }

        .stButton > button[kind="primary"],
        .stButton > button:not([kind="secondary"]):not([kind="tertiary"]) {
            background: linear-gradient(135deg, #0284C7 0%, #0EA5E9 100%) !important;
            border: 1px solid #0284C7 !important;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.25) !important;
        }

        .stButton > button[kind="primary"] *,
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        .stButton > button:not([kind="secondary"]) * {
            color: #FFFFFF !important;
            font-weight: 700 !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
        }

        .stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }

        .stButton > button[kind="secondary"] *,
        .stButton > button[kind="secondary"] p,
        .stButton > button[kind="secondary"] span {
            color: #0F172A !important;
            font-weight: 600 !important;
        }

        .stButton > button[kind="secondary"]:hover {
            background: #F0F9FF !important;
            border-color: #0284C7 !important;
        }

        .stButton > button[kind="secondary"]:hover * {
            color: #0284C7 !important;
        }

        /* Popovers - Clean Light Dropdown */
        div[data-testid="stPopoverBody"] {
            background: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.12) !important;
            color: #0F172A !important;
            padding: 16px !important;
            z-index: 999999 !important;
        }

        div[data-testid="stPopover"] > button,
        div[data-testid="stPopover"] button {
            background: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 9px !important;
            padding: 8px 14px !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stPopover"] > button *,
        div[data-testid="stPopover"] > button p,
        div[data-testid="stPopover"] > button span {
            color: #0F172A !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }

        div[data-testid="stPopover"] > button:hover,
        div[data-testid="stPopover"] button:hover {
            border-color: #0284C7 !important;
            background: #F0F9FF !important;
            transform: translateY(-1px) !important;
        }

        div[data-testid="stPopover"] > button:hover * {
            color: #0284C7 !important;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            background: #F1F5F9 !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 4px !important;
        }

        .stTabs [data-baseweb="tab"] {
            color: #64748B !important;
            font-weight: 500 !important;
            font-size: 0.86rem !important;
            border-radius: 7px !important;
            padding: 7px 16px !important;
            border: 1px solid transparent !important;
        }

        .stTabs [aria-selected="true"] {
            background: #FFFFFF !important;
            border: 1px solid #0284C7 !important;
            color: #0284C7 !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06) !important;
        }

        /* Dataframe & Tables */
        [data-testid="stDataFrame"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
        }

        /* Inputs */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            font-size: 0.88rem !important;
        }

        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #0284C7 !important;
            box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.15) !important;
        }

        /* Cards and Elements in Light Mode */
        .saas-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.2s ease;
        }

        .saas-card:hover {
            border-color: #0284C7 !important;
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(2, 132, 199, 0.1) !important;
        }

        .hero-container {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%) !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
        }

        .upload-box {
            background: #FFFFFF !important;
            border: 2px dashed #0284C7 !important;
            border-radius: 14px;
        }

        .upload-box:hover {
            background: rgba(2, 132, 199, 0.03) !important;
        }

        .workflow-step {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
        }

        .workflow-step.active {
            border-color: #0284C7 !important;
            background: #E0F2FE !important;
            box-shadow: 0 0 12px rgba(2, 132, 199, 0.15) !important;
        }

        .kpi-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-top: 3px solid #0284C7 !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important;
        }

        .kpi-val {
            color: #0F172A !important;
            font-size: 1.85rem;
            font-weight: 800;
        }

        /* Color Overrides for Inline Text in Light Theme */
        .saas-card [style*="color: #FFFFFF"], 
        .hero-container [style*="color: #FFFFFF"], 
        .kpi-card [style*="color: #FFFFFF"],
        [style*="color: #F8FAFC"] {
            color: #0F172A !important;
        }
        [style*="color: #F1F5F9"] {
            color: #1E293B !important;
        }
        [style*="color: #CBD5E1"] {
            color: #475569 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        /* Zero-Latency Native Font Stack */

        /* Full-Screen Edge-to-Edge Responsive Layout */
        .block-container {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            padding-top: 3.5rem !important;
            padding-bottom: 2rem !important;
        }

        /* Streamlit Header & Deployment Toolbar - Native & Non-Overlapping */
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

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            background-color: #080C16 !important;
            background-image: 
                radial-gradient(at 15% 10%, rgba(0, 240, 255, 0.04) 0px, transparent 40%),
                radial-gradient(at 85% 15%, rgba(139, 92, 246, 0.05) 0px, transparent 45%),
                radial-gradient(at 50% 90%, rgba(6, 182, 212, 0.03) 0px, transparent 45%),
                linear-gradient(rgba(255, 255, 255, 0.012) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.012) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 100% 100%, 36px 36px, 36px 36px;
            color: #E2E8F0 !important;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
            font-weight: 700 !important;
            color: #F8FAFC !important;
            letter-spacing: -0.025em !important;
        }

        code, pre, .mono-code {
            font-family: 'JetBrains Mono', monospace !important;
        }

        p, span, label, .stMarkdown {
            color: #94A3B8;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        /* Streamlit Sidebar Styling - ~250px Fixed Feel */
        section[data-testid="stSidebar"] {
            background: #0B0F1C !important;
            border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
            box-shadow: 6px 0 28px rgba(0, 0, 0, 0.4);
            padding-top: 0.5rem;
        }
        
        section[data-testid="stSidebar"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        section[data-testid="stSidebar"] .stRadio > div {
            gap: 2px !important;
        }

        section[data-testid="stSidebar"] .stRadio label {
            color: #94A3B8 !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            padding: 9px 12px !important;
            border-radius: 8px !important;
            border: 1px solid transparent !important;
            transition: all 0.18s ease-in-out !important;
            margin-bottom: 2px !important;
            cursor: pointer !important;
        }

        section[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(0, 240, 255, 0.05) !important;
            color: #F8FAFC !important;
            border-color: rgba(0, 240, 255, 0.15) !important;
        }

        section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div,
        section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.16) 0%, rgba(37, 99, 235, 0.22) 100%) !important;
            border: 1px solid rgba(0, 240, 255, 0.4) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            box-shadow: 0 0 12px rgba(0, 240, 255, 0.12) !important;
        }

        /* High-Contrast, Crystal-Clear Buttons */
        .stButton > button {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-size: 0.88rem !important;
            border-radius: 9px !important;
            padding: 8px 16px !important;
            letter-spacing: -0.01em !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }

        /* Active / Primary Button: Bright Electric Cyan with BOLD CRISP Black Text */
        .stButton > button[kind="primary"],
        .stButton > button:not([kind="secondary"]):not([kind="tertiary"]) {
            background: linear-gradient(135deg, #00F0FF 0%, #0284C7 100%) !important;
            border: 1px solid #00F0FF !important;
            box-shadow: 0 0 14px rgba(0, 240, 255, 0.35) !important;
        }

        .stButton > button[kind="primary"] *,
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        .stButton > button[kind="primary"] div,
        .stButton > button:not([kind="secondary"]) * {
            color: #030816 !important;
            font-weight: 800 !important;
            font-size: 0.88rem !important;
            text-shadow: none !important;
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.6) !important;
        }

        /* Inactive / Secondary Button: High-Contrast Solid Dark Navy with BOLD White Text */
        .stButton > button[kind="secondary"] {
            background: rgba(18, 26, 46, 0.95) !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        }

        .stButton > button[kind="secondary"] *,
        .stButton > button[kind="secondary"] p,
        .stButton > button[kind="secondary"] span,
        .stButton > button[kind="secondary"] div {
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            text-shadow: none !important;
        }

        .stButton > button[kind="secondary"]:hover {
            background: rgba(0, 240, 255, 0.14) !important;
            border-color: rgba(0, 240, 255, 0.5) !important;
            transform: translateY(-1px) !important;
        }

        .stButton > button[kind="secondary"]:hover * {
            color: #00F0FF !important;
        }

        /* Preserve Material Icons font ligatures (prevent arrow_down / icon text collision) */
        [data-testid="stIconMaterial"], 
        .material-symbols-rounded, 
        .material-icons,
        [data-testid="stExpanderToggleIcon"] {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        }

        /* Popovers - Dark Cyber */
        div[data-testid="stPopoverBody"] {
            background: #0E1424 !important;
            border: 1px solid rgba(0, 240, 255, 0.3) !important;
            border-radius: 12px !important;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.75) !important;
            color: #E2E8F0 !important;
            padding: 16px !important;
            z-index: 999999 !important;
        }

        div[data-testid="stPopover"] > button,
        div[data-testid="stPopover"] button {
            background: rgba(18, 26, 46, 0.95) !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            border-radius: 9px !important;
            padding: 8px 14px !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        }

        div[data-testid="stPopover"] > button *,
        div[data-testid="stPopover"] > button p,
        div[data-testid="stPopover"] > button span,
        div[data-testid="stPopover"] > button div {
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            text-shadow: none !important;
        }

        div[data-testid="stPopover"] > button:hover,
        div[data-testid="stPopover"] button:hover {
            border-color: rgba(0, 240, 255, 0.5) !important;
            background: rgba(0, 240, 255, 0.14) !important;
            transform: translateY(-1px) !important;
        }

        div[data-testid="stPopover"] > button:hover * {
            color: #00F0FF !important;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            background: #0D1322 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 4px !important;
        }

        .stTabs [data-baseweb="tab"] {
            color: #94A3B8 !important;
            font-weight: 500 !important;
            font-size: 0.86rem !important;
            border-radius: 7px !important;
            padding: 7px 16px !important;
            border: 1px solid transparent !important;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(0, 240, 255, 0.12) !important;
            border: 1px solid rgba(0, 240, 255, 0.35) !important;
            color: #00F0FF !important;
            font-weight: 600 !important;
        }

        /* Dataframe & Tables */
        [data-testid="stDataFrame"] {
            background: #0C1120 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
        }

        /* Inputs */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background: #0D1322 !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 8px !important;
            font-size: 0.88rem !important;
        }

        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #00F0FF !important;
            box-shadow: 0 0 0 1px #00F0FF !important;
        }

        /* Modern Clean SaaS Card */
        .saas-card {
            background: rgba(14, 20, 36, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }

        .saas-card:hover {
            border-color: rgba(0, 240, 255, 0.28);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }

        /* Top Navigation Bar */
        .top-navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(11, 17, 30, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 12px;
            padding: 10px 18px;
            margin-bottom: 22px;
        }

        /* Hero Section */
        .hero-container {
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 24px;
            background: linear-gradient(135deg, rgba(14, 22, 40, 0.95) 0%, rgba(10, 16, 30, 0.95) 100%);
            border: 1px solid rgba(0, 240, 255, 0.2);
            border-radius: 16px;
            padding: 32px 36px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        }

        @media (max-width: 960px) {
            .hero-container {
                grid-template-columns: 1fr;
            }
        }

        .gradient-text {
            background: linear-gradient(135deg, #00F0FF 0%, #38BDF8 50%, #818CF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }

        /* Upload Drag & Drop Area */
        .upload-box {
            background: rgba(12, 18, 34, 0.7);
            border: 2px dashed rgba(0, 240, 255, 0.35);
            border-radius: 14px;
            padding: 36px 24px;
            text-align: center;
            transition: all 0.25s ease;
            margin-bottom: 24px;
        }

        .upload-box:hover {
            border-color: #00F0FF;
            background: rgba(0, 240, 255, 0.03);
        }

        /* Workflow Horizontal Stepper */
        .workflow-track {
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            gap: 10px;
            margin: 16px 0 28px 0;
        }

        @media (max-width: 1024px) {
            .workflow-track {
                grid-template-columns: repeat(4, 1fr);
            }
        }
        @media (max-width: 640px) {
            .workflow-track {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        .workflow-step {
            background: rgba(14, 20, 36, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 12px 10px;
            text-align: center;
            transition: all 0.2s ease;
        }

        .workflow-step.active {
            border-color: rgba(0, 240, 255, 0.5);
            background: rgba(0, 240, 255, 0.08);
            box-shadow: 0 0 16px rgba(0, 240, 255, 0.15);
        }

        .workflow-step.completed {
            border-color: rgba(16, 185, 129, 0.35);
            background: rgba(16, 185, 129, 0.06);
        }

        /* KPI Cards */
        .kpi-card {
            background: rgba(14, 20, 36, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-top: 2px solid #00F0FF;
            border-radius: 12px;
            padding: 18px 20px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }

        .kpi-card:hover {
            border-color: rgba(0, 240, 255, 0.35);
            transform: translateY(-2px);
        }

        .kpi-val {
            font-family: 'Inter', -apple-system, sans-serif;
            font-size: 1.85rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.02em;
            margin: 6px 0 8px 0;
        }

        /* Authentication View Specific Styling matching Mockup */
        .auth-glass-card {
            background: linear-gradient(145deg, rgba(14, 20, 36, 0.95) 0%, rgba(10, 15, 28, 0.98) 100%);
            border: 1px solid rgba(0, 240, 255, 0.2);
            border-radius: 18px;
            padding: 32px 30px;
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.5);
        }

        .auth-analyst-card {
            background: linear-gradient(145deg, rgba(12, 18, 34, 0.92) 0%, rgba(8, 12, 24, 0.96) 100%);
            border: 1px solid rgba(0, 240, 255, 0.24);
            border-radius: 18px;
            padding: 28px 26px;
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.5);
        }

        .auth-feature-card {
            background: rgba(13, 20, 36, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 14px 16px;
            transition: all 0.2s ease;
        }

        .auth-feature-card:hover {
            border-color: rgba(0, 240, 255, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 240, 255, 0.12);
        }

        /* Anti-Overlap & Robust Flex Layout Guarantees */
        [data-testid="column"] {
            min-width: 0 !important;
            overflow-wrap: break-word !important;
        }

        [data-testid="stHorizontal"] {
            align-items: stretch !important;
            flex-wrap: wrap !important;
        }

        /* Container with border styling */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(11, 17, 30, 0.85) !important;
            border: 1px solid rgba(0, 240, 255, 0.22) !important;
            border-radius: 14px !important;
            padding: 18px 20px !important;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45) !important;
            margin-bottom: 12px !important;
        }

        /* Prevent BaseWeb tabs underline / highlight from slicing text */
        div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
            display: none !important;
        }

        /* Headings line height guarantee */
        h1, h2, h3, h4, h5, h6 {
            line-height: 1.3 !important;
            word-break: break-word !important;
        }
        </style>
        """, unsafe_allow_html=True)

def render_top_navbar(breadcrumb: str = "Workspace / Dashboard", active_dataset: Optional[str] = None):
    """Renders the clean, fully functional SaaS top navigation bar."""
    curr_theme = st.session_state.get("theme", "dark_futuristic")
    is_dark = (curr_theme == "dark_futuristic")
    user = st.session_state.get("user") or {}
    is_auth = bool(st.session_state.get("authenticated", False))

    col_left, col_search, col_right = st.columns([2.5, 4.0, 3.5])

    with col_left:
        dataset_label = f"<span style='color: {'#00F0FF' if is_dark else '#0284C7'}; font-weight: 600;'>{active_dataset}</span>" if active_dataset else "<span style='color: #94A3B8;'>Ready for ingestion</span>"
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; padding-top: 7px;">
            <span style="color: {'#64748B' if is_dark else '#475569'}; font-size: 0.84rem; font-weight: 500;">{breadcrumb}</span>
            <span style="color: {'#475569' if is_dark else '#94A3B8'};">•</span>
            <span style="font-size: 0.84rem;">{dataset_label}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_search:
        # Sleek command query bar
        query = st.text_input(
            "AI Command Query",
            placeholder="🔍 Ask anything about your data... (Ctrl + K)",
            label_visibility="collapsed",
            key="global_top_ai_search_input"
        )
        if query:
            st.session_state["active_nav"] = "AI Chat"
            st.session_state["pending_chat_prompt"] = query

    with col_right:
        c_theme, c_notif, c_prof = st.columns([1.0, 1.1, 1.9], gap="small")

        # 1. THEME TOGGLE (Dark / Light)
        with c_theme:
            theme_icon = "🌙" if is_dark else "☀️"
            theme_btn_label = f"{theme_icon} {'Dark' if is_dark else 'Light'}"
            if st.button(
                theme_btn_label,
                key="top_theme_toggle_btn",
                use_container_width=True,
                help=f"Currently {theme_btn_label}. Click to toggle {'Light' if is_dark else 'Dark'} Mode"
            ):
                st.session_state["theme"] = "light_classic" if is_dark else "dark_futuristic"
                st.rerun()

        # 2. NOTIFICATIONS (Live Alerts, Badge & Popover)
        with c_notif:
            from core.models.notification_model import NotificationModel
            u_id = user.get("id") if is_auth else None
            unread_cnt = NotificationModel.get_unread_count(user_id=u_id)
            notif_btn_label = f"🔔 {unread_cnt}" if unread_cnt > 0 else "🔔"

            with st.popover(notif_btn_label, use_container_width=True):
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 0.92rem; color: {'#F8FAFC' if is_dark else '#0F172A'};">
                        🔔 System & Alerts
                    </span>
                    <span style="background: {'rgba(0, 240, 255, 0.18)' if is_dark else 'rgba(2, 132, 199, 0.15)'}; 
                                 color: {'#00F0FF' if is_dark else '#0284C7'}; 
                                 font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 10px;">
                        {unread_cnt} New
                    </span>
                </div>
                """, unsafe_allow_html=True)

                c_a1, c_a2 = st.columns(2)
                with c_a1:
                    if st.button("✓ Mark Read", key="top_notif_mark_read_btn", use_container_width=True):
                        NotificationModel.mark_all_as_read(user_id=u_id)
                        st.rerun()
                with c_a2:
                    if st.button("🗑️ Clear All", key="top_notif_clear_all_btn", use_container_width=True):
                        NotificationModel.clear_all(user_id=u_id)
                        st.rerun()

                st.markdown(f"<hr style='margin: 8px 0; border-color: {'rgba(255, 255, 255, 0.08)' if is_dark else '#E2E8F0'};'>", unsafe_allow_html=True)

                notifs = NotificationModel.get_notifications(user_id=u_id, limit=6)
                if not notifs:
                    st.caption("✨ All caught up! No active notifications.")
                else:
                    for n in notifs:
                        is_unread = (n.get("is_read", 0) == 0)
                        dot = "● " if is_unread else ""
                        dot_color = "#00F0FF" if is_dark else "#0284C7"
                        unread_bg = ("rgba(0, 240, 255, 0.06)" if is_dark else "rgba(2, 132, 199, 0.06)") if is_unread else "transparent"
                        border_c = ("rgba(0, 240, 255, 0.2)" if is_dark else "rgba(2, 132, 199, 0.3)") if is_unread else ("rgba(255, 255, 255, 0.06)" if is_dark else "#E2E8F0")

                        st.markdown(f"""
                        <div style="background: {unread_bg}; border: 1px solid {border_c}; border-radius: 8px; padding: 8px 10px; margin-bottom: 6px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                                <span style="font-size: 0.82rem; font-weight: 600; color: {'#FFFFFF' if is_dark else '#0F172A'};">
                                        {n.get('icon', '🔔')} <span style="color: {dot_color};">{dot}</span>{n.get('title', '')}
                                </span>
                                <span style="font-size: 0.68rem; color: #64748B;">{n.get('created_at', '')[:10]}</span>
                            </div>
                            <div style="font-size: 0.77rem; color: {'#94A3B8' if is_dark else '#475569'}; line-height: 1.35;">
                                {n.get('message', '')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        route = n.get("link_route")
                        if route:
                            if st.button(f"Go to {route} →", key=f"notif_route_{n['id']}", use_container_width=True):
                                NotificationModel.mark_as_read(n["id"])
                                st.session_state["active_nav"] = route
                                st.rerun()

        # 3. PROFILE SECTION (User Card & Popover Actions)
        with c_prof:
            if is_auth:
                full_name = user.get("full_name") or user.get("username") or "Chintu Kumar"
                role = user.get("role") or "Lead Data Analyst"
                email = user.get("email") or "analyst@quantum.ai"
                initials = "".join([part[0].upper() for part in full_name.split()[:2]]) or "CK"
                first_name = full_name.split()[0]
                prof_btn_label = f"👤 {first_name[:10]} ▾"

                with st.popover(prof_btn_label, use_container_width=True):
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 12px; padding: 4px 2px 10px 2px; border-bottom: 1px solid {'rgba(255, 255, 255, 0.08)' if is_dark else '#E2E8F0'}; margin-bottom: 10px;">
                        <div style="width: 40px; height: 40px; border-radius: 50%; 
                                    background: linear-gradient(135deg, #00F0FF, #8B5CF6); 
                                    display: flex; align-items: center; justify-content: center; 
                                    font-size: 0.95rem; font-weight: 800; color: #050811; box-shadow: 0 0 10px rgba(0, 240, 255, 0.3);">
                            {initials}
                        </div>
                        <div>
                            <div style="font-weight: 700; font-size: 0.94rem; color: {'#FFFFFF' if is_dark else '#0F172A'};">
                                {full_name}
                            </div>
                            <div style="font-size: 0.75rem; color: #94A3B8;">{email}</div>
                            <div style="display: inline-block; background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 4px; padding: 1px 6px; font-size: 0.68rem; font-weight: 600; margin-top: 3px;">
                                ● {role}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="background: {'rgba(255, 255, 255, 0.04)' if is_dark else '#F8FAFC'}; border: 1px solid {'rgba(255, 255, 255, 0.06)' if is_dark else '#E2E8F0'}; border-radius: 6px; padding: 6px 10px; font-size: 0.76rem; color: {'#CBD5E1' if is_dark else '#334155'}; margin-bottom: 10px;">
                        🔐 <b>Session:</b> Verified Authenticated<br>
                        🛡️ <b>Clearance:</b> Enterprise Level 3<br>
                        ⚡ <b>Tier:</b> Enterprise Pro Analyst
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("⚙️ Platform Settings", key="top_prof_settings_btn", use_container_width=True):
                        st.session_state["active_nav"] = "Settings"
                        st.rerun()

                    if st.button("🛡️ Security & Login Logs", key="top_prof_security_btn", use_container_width=True):
                        st.session_state["active_nav"] = "Auth"
                        st.rerun()

                    st.markdown(f"<hr style='margin: 8px 0; border-color: {'rgba(255, 255, 255, 0.08)' if is_dark else '#E2E8F0'};'>", unsafe_allow_html=True)

                    if st.button("🚪 Sign Out", key="top_prof_signout_btn", use_container_width=True, type="primary"):
                        st.session_state["authenticated"] = False
                        st.session_state["user"] = None
                        st.session_state["df"] = None
                        st.session_state["raw_df"] = None
                        st.session_state["current_project_id"] = None
                        st.session_state["current_project_name"] = None
                        st.session_state["active_nav"] = "Auth"
                        st.rerun()
            else:
                if st.button("🔑 Sign In", key="top_prof_signin_btn", type="primary", use_container_width=True):
                    st.session_state["active_nav"] = "Auth"
                    st.rerun()

    st.markdown(f"<hr style='border-color: {'rgba(255, 255, 255, 0.06)' if is_dark else '#E2E8F0'}; margin: 8px 0 18px 0;'>", unsafe_allow_html=True)

def render_saas_hero():
    """Renders the clean welcome/AI analyst hero section (replaces old code box)."""
    st.markdown("""
    <div class="hero-container">
        <div>
            <div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(0, 240, 255, 0.08); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 6px; padding: 4px 12px; margin-bottom: 14px;">
                <span style="color: #00F0FF; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em;">TURN DATA INTO INSIGHTS WITH AI</span>
            </div>
            <h1 style="font-size: 2.65rem; font-weight: 800; line-height: 1.15; margin-bottom: 12px; color: #FFFFFF;">
                Your Complete<br>
                <span class="gradient-text">AI Data Analyst</span>
            </h1>
            <p style="font-size: 1.02rem; color: #94A3B8; max-width: 580px; margin-bottom: 0; line-height: 1.6;">
                Upload any dataset and let AI handle the entire analytics workflow — from data cleaning and EDA to SQL queries, automated insights, and interactive dashboards.
            </p>
        </div>
        <div style="background: rgba(10, 16, 30, 0.85); border: 1px solid rgba(255, 255, 255, 0.09); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 12px; line-height: 1.35;">
                “Upload. Analyze. Visualize.<br>All in one place.”
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.85rem; color: #CBD5E1;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #00F0FF; font-weight: 700;">✓</span> No coding required
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #00F0FF; font-weight: 700;">✓</span> Multiple file formats
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #00F0FF; font-weight: 700;">✓</span> AI-powered analysis
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #10B981; font-weight: 700;">✓</span> Enterprise-grade security
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_workflow_stepper(df_loaded: bool = False, current_stage: str = "INGEST"):
    """Renders the horizontal AI Analytics Workflow with dynamic statuses."""
    steps = [
        {"num": "01", "name": "INGEST", "icon": "📥", "desc": "Ingestion", "route": "Upload Dataset"},
        {"num": "02", "name": "PROFILE", "icon": "📋", "desc": "Schema Profiling", "route": "Data Profile"},
        {"num": "03", "name": "CLEAN", "icon": "🧹", "desc": "Audit & Cleanse", "route": "Data Cleaning"},
        {"num": "04", "name": "EDA", "icon": "🔬", "desc": "Distributions & EDA", "route": "EDA"},
        {"num": "05", "name": "FEATURE ENG", "icon": "⚡", "desc": "Transformations", "route": "Feature Engineering"},
        {"num": "06", "name": "SQL", "icon": "💾", "desc": "Query Engine", "route": "SQL Analytics"},
        {"num": "07", "name": "AI INSIGHTS", "icon": "✨", "desc": "Multi-Horizon AI", "route": "AI Insights"},
        {"num": "08", "name": "DASHBOARD", "icon": "📊", "desc": "Interactive Matrix", "route": "Dashboard"}
    ]

    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin: 12px 0 10px 0;">
        <span style="font-size: 0.78rem; font-weight: 700; color: #00F0FF; letter-spacing: 0.08em;">// AI ANALYTICS WORKFLOW</span>
        <span style="font-size: 0.78rem; color: #64748B;">Autonomous 8-stage intelligence pipeline</span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(8)
    for idx, s in enumerate(steps):
        with cols[idx]:
            if not df_loaded:
                status_badge = "<span style='color: #64748B; font-size: 0.7rem;'>○ Ready</span>"
                border_color = "rgba(255, 255, 255, 0.07)"
                bg = "rgba(14, 20, 36, 0.7)"
            else:
                if idx <= 2:
                    status_badge = "<span style='color: #10B981; font-size: 0.7rem; font-weight: 600;'>✓ Done</span>"
                    border_color = "rgba(16, 185, 129, 0.35)"
                    bg = "rgba(16, 185, 129, 0.06)"
                elif idx == 3:
                    status_badge = "<span style='color: #00F0FF; font-size: 0.7rem; font-weight: 600;'>⟳ Active</span>"
                    border_color = "rgba(0, 240, 255, 0.5)"
                    bg = "rgba(0, 240, 255, 0.09)"
                else:
                    status_badge = "<span style='color: #94A3B8; font-size: 0.7rem;'>○ Ready</span>"
                    border_color = "rgba(255, 255, 255, 0.08)"
                    bg = "rgba(14, 20, 36, 0.7)"

            st.markdown(f"""
            <div style="background: {bg}; border: 1px solid {border_color}; border-radius: 9px; padding: 10px 6px; text-align: center; height: 100%;">
                <div style="font-size: 0.68rem; font-weight: 700; color: #64748B;">{s['num']}</div>
                <div style="font-size: 1.15rem; margin: 2px 0;">{s['icon']}</div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #F8FAFC; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{s['name']}</div>
                <div style="margin-top: 4px;">{status_badge}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Go", key=f"wf_btn_{s['num']}", use_container_width=True, help=f"Navigate to {s['desc']}"):
                st.session_state["active_nav"] = s["route"]
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

def render_dataset_header_card(df, project_name: str, quality_score: int = 94):
    """Renders a next-level cyber glassmorphism telemetry cockpit card for the active dataset."""
    n_rows, n_cols = df.shape
    est_size_mb = (df.memory_usage().sum()) / (1024 * 1024)
    size_str = f"{est_size_mb:.1f} MB" if est_size_mb >= 0.1 else f"{int(est_size_mb * 1024)} KB"

    # Dynamic Integrity Score styling & Grade
    if quality_score >= 90:
        score_color = "#10B981"
        score_bg = "rgba(16, 185, 129, 0.14)"
        score_border = "rgba(16, 185, 129, 0.4)"
        grade_label = "GRADE A • PRISTINE"
    elif quality_score >= 75:
        score_color = "#F59E0B"
        score_bg = "rgba(245, 158, 11, 0.14)"
        score_border = "rgba(245, 158, 11, 0.4)"
        grade_label = "GRADE B • STABLE"
    else:
        score_color = "#EF4444"
        score_bg = "rgba(239, 68, 68, 0.14)"
        score_border = "rgba(239, 68, 68, 0.4)"
        grade_label = "GRADE C • ATTENTION REQUIRED"

    # Scale and Engine telemetry badge
    if n_rows >= 250000:
        scale_badge = '<span style="background: rgba(124, 58, 237, 0.22); color: #A78BFA; border: 1px solid rgba(124, 58, 237, 0.45); padding: 3px 9px; border-radius: 6px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;">⚡ 1M+ DUCKDB OLAP</span>'
    else:
        scale_badge = '<span style="background: rgba(0, 240, 255, 0.14); color: #00F0FF; border: 1px solid rgba(0, 240, 255, 0.35); padding: 3px 9px; border-radius: 6px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;">⚡ VECTORIZED SIMD</span>'

    # Build bulletproof HTML with zero empty lines to prevent markdown code block leaks
    header_html = (
        f'<div style="background: radial-gradient(circle at 90% 10%, rgba(0, 240, 255, 0.09) 0%, rgba(15, 23, 42, 0.96) 65%); '
        f'border: 1px solid rgba(0, 240, 255, 0.28); border-radius: 16px; padding: 22px 28px; margin-bottom: 18px; '
        f'box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08); backdrop-filter: blur(16px);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">'
        f'<div>'
        f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 5px;">'
        f'<span style="width: 8px; height: 8px; border-radius: 50%; background: #00F0FF; display: inline-block; box-shadow: 0 0 10px #00F0FF;"></span>'
        f'<span style="font-size: 0.72rem; font-weight: 800; color: #00F0FF; letter-spacing: 0.1em; text-transform: uppercase;">ACTIVE TELEMETRY WORKSPACE</span>'
        f'{scale_badge}'
        f'</div>'
        f'<h2 style="margin: 2px 0 6px 0; font-size: 1.65rem; color: #FFFFFF; font-weight: 800; letter-spacing: -0.02em;">{project_name}</h2>'
        f'<div style="display: flex; gap: 12px; align-items: center; font-size: 0.84rem; color: #94A3B8; font-family: monospace;">'
        f'<span><strong style="color: #F8FAFC;">{n_rows:,}</strong> rows</span>'
        f'<span style="color: #475569;">•</span>'
        f'<span><strong style="color: #F8FAFC;">{n_cols}</strong> columns</span>'
        f'<span style="color: #475569;">•</span>'
        f'<span><strong style="color: #F8FAFC;">{size_str}</strong> footprint</span>'
        f'<span style="color: #475569;">•</span>'
        f'<span style="color: #00F0FF;">Parquet Snappy Zero-Copy</span>'
        f'</div>'
        f'</div>'
        f'<div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">'
        f'<div style="background: {score_bg}; border: 1px solid {score_border}; border-radius: 12px; padding: 10px 18px; text-align: center; min-width: 140px; box-shadow: 0 4px 14px {score_border};">'
        f'<div style="font-size: 0.68rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase;">DATA INTEGRITY</div>'
        f'<div style="font-size: 1.85rem; font-weight: 900; color: {score_color}; line-height: 1.1; font-family: monospace;">{quality_score}%</div>'
        f'<div style="font-size: 0.66rem; font-weight: 700; color: {score_color}; letter-spacing: 0.04em; margin-top: 2px;">{grade_label}</div>'
        f'</div>'
        f'<div style="display: flex; flex-direction: column; gap: 6px;">'
        f'<span style="background: rgba(16, 185, 129, 0.14); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.35); padding: 4px 12px; border-radius: 6px; font-size: 0.74rem; font-weight: 600; display: inline-flex; align-items: center; gap: 6px;">✓ Ingestion & Schema Locked</span>'
        f'<span style="background: rgba(16, 185, 129, 0.14); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.35); padding: 4px 12px; border-radius: 6px; font-size: 0.74rem; font-weight: 600; display: inline-flex; align-items: center; gap: 6px;">✓ 6D Quality Audit Passed</span>'
        f'<span style="background: rgba(0, 240, 255, 0.14); color: #00F0FF; border: 1px solid rgba(0, 240, 255, 0.35); padding: 4px 12px; border-radius: 6px; font-size: 0.74rem; font-weight: 600; display: inline-flex; align-items: center; gap: 6px;">⟳ Vectorized Analytics Armed</span>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.html(header_html)

    # Next-Level Glass Quick Command Bar
    c_b1, c_b2, c_b3, c_b4, c_b5 = st.columns(5)
    with c_b1:
        if st.button("🧹 AI Data Cleaning", type="primary", use_container_width=True, help="Scan & automatically fix missing values, type conflicts, and duplicate rows"):
            st.session_state["active_nav"] = "Data Cleaning"
            st.rerun()
    with c_b2:
        if st.button("🔬 Deep EDA Matrix", use_container_width=True, help="Explore distributions, correlations, multivariate anomalies, and chronological trends"):
            st.session_state["active_nav"] = "EDA"
            st.rerun()
    with c_b3:
        if st.button("⚡ SQL Analytics", use_container_width=True, help="Execute natural language or direct DuckDB SQL queries with zero Pandas overhead"):
            st.session_state["active_nav"] = "SQL Analytics"
            st.rerun()
    with c_b4:
        if st.button("✨ Ask AI Analyst", use_container_width=True, help="Synthesize executive narratives and ask deep questions about this dataset"):
            st.session_state["active_nav"] = "AI Chat"
            st.rerun()
    with c_b5:
        if st.button("📂 Switch / Upload", use_container_width=True, help="Switch projects or upload a new telemetry dataset"):
            st.session_state["active_nav"] = "Upload Dataset"
            st.rerun()

def render_autonomous_intelligence_banner(df, quality_dict: dict = None):
    """Renders a next-level autonomous AI intelligence briefing and exploration launchpad."""
    n_rows, n_cols = df.shape
    missing_cells = int(df.isnull().sum().sum())
    total_cells = n_rows * n_cols
    completeness_pct = round((1 - (missing_cells / total_cells if total_cells else 0)) * 100, 1)
    
    missing_cols_dict = {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0}
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    # Dynamic AI synthesis narrative
    narrative_points = []
    narrative_points.append(f"Scanned <b>{n_rows:,}</b> records across <b>{n_cols}</b> dimensions (<b>{len(num_cols)}</b> numerical, <b>{len(cat_cols)}</b> categorical).")
    
    if missing_cells > 0:
        top_missing = sorted(missing_cols_dict.items(), key=lambda x: x[1], reverse=True)[:2]
        missing_str = ", ".join([f"<code>{col}</code> ({round((cnt/n_rows)*100, 1)}%)" for col, cnt in top_missing])
        narrative_points.append(f"Completeness is at <b>{completeness_pct}%</b>. Missingness concentrated in {missing_str}.")
        action_advice = "Recommend executing AI Data Cleaning to impute missing attributes and standardize categories."
    else:
        narrative_points.append("Dataset is <b>100% complete</b> with zero missing values detected.")
        action_advice = "Dataset is in pristine health. Ready for multivariate correlation analysis and ML feature extraction."

    banner_html = (
        f'<div style="background: linear-gradient(135deg, rgba(124, 58, 237, 0.12) 0%, rgba(15, 23, 42, 0.95) 100%); '
        f'border: 1px solid rgba(139, 92, 246, 0.35); border-left: 4px solid #A78BFA; border-radius: 14px; padding: 16px 20px; margin-bottom: 14px; '
        f'box-shadow: 0 8px 24px -8px rgba(124, 58, 237, 0.2);">'
        f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">'
        f'<div style="display: flex; align-items: center; gap: 8px;">'
        f'<span style="font-size: 1rem;">✨</span>'
        f'<span style="font-size: 0.78rem; font-weight: 800; color: #C4B5FD; letter-spacing: 0.08em; text-transform: uppercase;">AUTONOMOUS AI TELEMETRY BRIEFING</span>'
        f'</div>'
        f'<span style="font-size: 0.72rem; color: #94A3B8; font-family: monospace;">LATENCY: 0.04ms • ZERO RAW DATA EXPOSURE</span>'
        f'</div>'
        f'<div style="font-size: 0.86rem; color: #E2E8F0; line-height: 1.5;">'
        f'{" ".join(narrative_points)} <span style="color: #67E8F9;">{action_advice}</span>'
        f'</div>'
        f'</div>'
    )
    st.html(banner_html)

    # Next-Level Quick AI Prompt Chips
    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
    with chip_col1:
        if st.button("💡 Key Variance Drivers", key="ai_chip_drivers", use_container_width=True, help="Analyze primary drivers of variance"):
            st.session_state["active_nav"] = "EDA"
            st.session_state["eda_active_tab"] = "Drivers"
            st.rerun()
    with chip_col2:
        if st.button("🔍 Outlier Radar Scan", key="ai_chip_outliers", use_container_width=True, help="Scan multivariate Isolation Forest outliers"):
            st.session_state["active_nav"] = "EDA"
            st.session_state["eda_active_tab"] = "Anomalies"
            st.rerun()
    with chip_col3:
        if st.button("📊 Correlation Matrix", key="ai_chip_corrs", use_container_width=True, help="Inspect collinearity and relationship clusters"):
            st.session_state["active_nav"] = "EDA"
            st.session_state["eda_active_tab"] = "Correlations"
            st.rerun()
    with chip_col4:
        if st.button("📑 Export Analytics PDF", key="ai_chip_report", use_container_width=True, help="Open formal report generator"):
            st.session_state["active_nav"] = "Reports"
            st.rerun()

def render_quick_actions():
    """Renders 4 clean SaaS cards for Quick Actions."""
    st.markdown("""
    <div style="font-size: 0.78rem; font-weight: 700; color: #00F0FF; letter-spacing: 0.08em; margin: 20px 0 10px 0;">
        // QUICK ACTIONS
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown("""
        <div class="saas-card">
            <div style="font-size: 1.4rem; margin-bottom: 8px;">🧹</div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.96rem; margin-bottom: 4px;">AI Data Cleaning</div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 12px;">12-category automated hygiene & verification.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Clean Dataset →", key="qa_clean", use_container_width=True):
            st.session_state["active_nav"] = "Data Cleaning"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="saas-card">
            <div style="font-size: 1.4rem; margin-bottom: 8px;">📊</div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.96rem; margin-bottom: 4px;">Analyze Dataset</div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 12px;">Distributions, skewness, moments & correlations.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open EDA →", key="qa_eda", use_container_width=True):
            st.session_state["active_nav"] = "EDA"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="saas-card">
            <div style="font-size: 1.4rem; margin-bottom: 8px;">⚡</div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.96rem; margin-bottom: 4px;">Run SQL</div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 12px;">Natural language queries executed via DuckDB.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Execute SQL →", key="qa_sql", use_container_width=True):
            st.session_state["active_nav"] = "SQL Analytics"
            st.rerun()

    with c4:
        st.markdown("""
        <div class="saas-card">
            <div style="font-size: 1.4rem; margin-bottom: 8px;">✨</div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.96rem; margin-bottom: 4px;">Generate Insights</div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 12px;">Multi-horizon diagnostic and predictive synthesis.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Insights →", key="qa_insights", use_container_width=True):
            st.session_state["active_nav"] = "AI Insights"
            st.rerun()

    with c5:
        st.markdown("""
        <div class="saas-card">
            <div style="font-size: 1.4rem; margin-bottom: 8px;">📈</div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.96rem; margin-bottom: 4px;">Build Dashboard</div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 12px;">Custom drag & drop visual metric canvases.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Builder →", key="qa_builder", use_container_width=True):
            st.session_state["active_nav"] = "Dashboard Builder"
            st.rerun()

def render_recent_projects_cards():
    """Renders the recent projects section with attractive SaaS cards."""
    st.markdown("""
    <div style="font-size: 0.78rem; font-weight: 700; color: #00F0FF; letter-spacing: 0.08em; margin: 24px 0 10px 0;">
        // RECENT PROJECTS
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="saas-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 1rem;">Sales Analysis</div>
                <span style="background: rgba(0, 240, 255, 0.12); color: #00F0FF; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; font-weight: 600;">ACTIVE</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.84rem; margin-bottom: 12px;">
                <b>9,800</b> rows • <b>21</b> columns<br>
                <span style="color: #64748B;">Last analyzed: Today</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Project", key="rp_open_sales", use_container_width=True):
            from core.sample_data import generate_sales_dataset
            from core.models import ProjectModel
            df = generate_sales_dataset(800)
            p = ProjectModel.create_project("Sales Analysis", df, goal="Revenue Maximization & Growth")
            st.session_state["current_project_id"] = p.project_id
            st.session_state["current_project_name"] = "Sales Analysis"
            st.session_state["df"] = df
            st.session_state["raw_df"] = df.copy()
            st.session_state["active_nav"] = "Dashboard"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="saas-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 1rem;">Customer Analytics</div>
                <span style="background: rgba(168, 85, 247, 0.12); color: #C084FC; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; font-weight: 600;">CHURN MODEL</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.84rem; margin-bottom: 12px;">
                <b>12,430</b> rows • <b>18</b> columns<br>
                <span style="color: #64748B;">Last analyzed: 2 days ago</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Project", key="rp_open_churn", use_container_width=True):
            from core.sample_data import generate_customer_churn_dataset
            from core.models import ProjectModel
            df = generate_customer_churn_dataset(400)
            p = ProjectModel.create_project("Customer Analytics", df, goal="Customer Churn Prevention & Retention")
            st.session_state["current_project_id"] = p.project_id
            st.session_state["current_project_name"] = "Customer Analytics"
            st.session_state["df"] = df
            st.session_state["raw_df"] = df.copy()
            st.session_state["active_nav"] = "Dashboard"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="saas-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 1rem;">Finance Analytics</div>
                <span style="background: rgba(16, 185, 129, 0.12); color: #10B981; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; font-weight: 600;">PORTFOLIO</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.84rem; margin-bottom: 12px;">
                <b>7,820</b> rows • <b>24</b> columns<br>
                <span style="color: #64748B;">Last analyzed: 3 days ago</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Project", key="rp_open_fin", use_container_width=True):
            from core.sample_data import generate_finance_dataset
            from core.models import ProjectModel
            df = generate_finance_dataset(600)
            p = ProjectModel.create_project("Finance Analytics", df, goal="Cost Reduction & Margin Optimization")
            st.session_state["current_project_id"] = p.project_id
            st.session_state["current_project_name"] = "Finance Analytics"
            st.session_state["df"] = df
            st.session_state["raw_df"] = df.copy()
            st.session_state["active_nav"] = "Dashboard"
            st.rerun()

def render_ai_analyst_card():
    """Renders the prominent MEET YOUR AI DATA ANALYST section."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(16, 24, 46, 0.9) 0%, rgba(12, 18, 36, 0.9) 100%); 
                border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 14px; padding: 24px; margin: 24px 0 16px 0;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span style="font-size: 1.25rem;">✨</span>
            <span style="font-size: 0.8rem; font-weight: 700; color: #A855F7; letter-spacing: 0.08em;">MEET YOUR AI DATA ANALYST</span>
        </div>
        <h3 style="font-size: 1.35rem; color: #FFFFFF; margin: 0 0 8px 0; font-weight: 800;">
            Ask questions in plain English.
        </h3>
        <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.55; max-width: 720px; margin-bottom: 16px;">
            Quantum Analyst automatically understands your dataset schema, writes SQL queries, performs statistical analysis, creates visualizations, and explains the results.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<span style='font-size: 0.8rem; color: #64748B; font-weight: 600;'>TRY ASKING:</span>", unsafe_allow_html=True)
    
    p1, p2, p3, p4, p5 = st.columns(5)
    sample_prompts = [
        ("Top 10 Products", "What are my top 10 products?"),
        ("Why Sales Declined", "Why did sales decline?"),
        ("Valuable Customers", "Which customers are most valuable?"),
        ("Monthly Trends", "Show monthly revenue trends."),
        ("Profit Drivers", "What factors affect profit?")
    ]
    
    for i, (col, (label, prompt_text)) in enumerate(zip([p1, p2, p3, p4, p5], sample_prompts)):
        with col:
            if st.button(f"💬 {label}", key=f"ai_prompt_{i}", use_container_width=True):
                st.session_state["active_nav"] = "AI Chat"
                st.session_state["pending_chat_prompt"] = prompt_text
                st.rerun()

    c_input, c_btn = st.columns([5, 1])
    with c_input:
        user_prompt = st.text_input("Ask question", placeholder="e.g. Which region generated the highest profit margin last quarter?", label_visibility="collapsed", key="ai_card_custom_query")
    with c_btn:
        if st.button("Start AI Analysis", type="primary", use_container_width=True):
            st.session_state["active_nav"] = "AI Chat"
            if user_prompt:
                st.session_state["pending_chat_prompt"] = user_prompt
            st.rerun()

def render_kpi_card(title: str, value: str, subtext: str = "", icon: str = "📊", status: str = "neutral", badge: str = "METRIC"):
    """Clean enterprise KPI card."""
    pill_color = "#10B981" if status == "positive" else ("#F43F5E" if status == "negative" else "#00F0FF")
    pill_bg = "rgba(16, 185, 129, 0.15)" if status == "positive" else ("rgba(244, 63, 94, 0.15)" if status == "negative" else "rgba(0, 240, 255, 0.12)")

    st.markdown(f"""
    <div class="kpi-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">{title}</span>
            <span style="font-size: 1.15rem;">{icon}</span>
        </div>
        <div class="kpi-val">{value}</div>
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.78rem; color: #94A3B8;">
            <span style="background: {pill_bg}; color: {pill_color}; font-size: 0.7rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">{badge}</span>
            <span>{subtext}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_requirement_spec_box(title: str, rationale: str, business_question: str = "", data_lineage: str = ""):
    st.markdown(f"""
    <div style="background: rgba(14, 20, 36, 0.9); border: 1px solid rgba(139, 92, 246, 0.25); border-left: 3px solid #00F0FF; border-radius: 8px; padding: 14px 18px; margin: 12px 0;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; margin-bottom: 6px;">📋 REQUIREMENT SPECIFICATION: {title.upper()}</div>
        <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.5;">
            <b>► Analytical Objective:</b> {rationale}<br>
            {f'<b>► Business Question:</b> {business_question}<br>' if business_question else ''}
            {f'<b>► Data Lineage:</b> {data_lineage}' if data_lineage else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_landing_hero(theme: str = "dark_futuristic", *args, **kwargs):
    render_saas_hero()

