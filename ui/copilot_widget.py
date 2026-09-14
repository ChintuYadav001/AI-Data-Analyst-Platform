"""
ui.copilot_widget
=================
High-Quality, Project-Aware Quantum AI Data Analyst & Voice Assistant.
Renders an ultra-modern floating launcher button [ ✨ AI ] at the bottom-right of the screen.
When opened, presents an institutional-grade AI Data Assistant equipped with:
- Natural language chat & Web Speech voice assistance (STT & TTS)
- Context-awareness of the active page, dataset, target, and champion model
- 18 controlled calculation tools (DuckDB, Profiler, Quality, Cleaning, EDA, ML, Forecasting)
- Simple Language Mode with intuitive analogies
- Structured Answer -> Evidence -> Explanation -> Recommendation output
- Dynamic follow-up action buttons and context-aware suggestion chips
"""

import json
from typing import Optional, Dict, Any, List
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from core.ai.quantum_ai_engine import QuantumAIEngine
from core.ml.experiment_tracker import get_active_champion


def init_copilot_state():
    """Initializes session state variables for Quantum AI Assistant."""
    if "copilot_open" not in st.session_state:
        st.session_state["copilot_open"] = False
    if "copilot_minimized" not in st.session_state:
        st.session_state["copilot_minimized"] = False
    if "copilot_voice_active" not in st.session_state:
        st.session_state["copilot_voice_active"] = False
    if "copilot_voice_to_play" not in st.session_state:
        st.session_state["copilot_voice_to_play"] = None
    if "copilot_stop_speech" not in st.session_state:
        st.session_state["copilot_stop_speech"] = False
    if "copilot_pending_query" not in st.session_state:
        st.session_state["copilot_pending_query"] = None
    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "### ✨ Hello! I am Quantum AI\n"
                    "**Your Intelligent Data Analyst & Predictive Analytics Assistant.**\n\n"
                    "I can help you analyze your data, audit quality, discover EDA patterns, "
                    "train machine learning models, forecast future numbers, or explain how Quantum Analyst works in simple words.\n\n"
                    "💡 *Click a suggested chip below, type a question, or click the **🎙️ Microphone** to talk hands-free!*"
                ),
                "voice_text": "Hello! I am Quantum AI, your intelligent data analyst. You can ask me to analyze your data, check data quality, explain models in simple words, or forecast future numbers.",
                "actions": ["What does this project do?", "Summarize my dataset", "What is PCA?"],
            }
        ]


def get_context_suggested_questions(current_page: str, has_data: bool) -> List[str]:
    """Generates dynamic suggestion chips tailored specifically to the user's active page."""
    if not has_data:
        return [
            "💡 What does this project do?",
            "📷 What is PCA?",
            "🌲 What is Random Forest?",
            "⚡ Why is DuckDB used?",
        ]

    page_clean = (current_page or "").lower()

    if "profile" in page_clean or "upload" in page_clean:
        return [
            "📋 Summarize my dataset",
            "🛡️ Are there data-quality problems?",
            "🧹 What should I clean?",
            "💡 Explain the workflow",
        ]
    elif "quality" in page_clean:
        return [
            "🛡️ Find the biggest problem in my data",
            "❓ Why is my quality score this value?",
            "🧹 What should I clean first?",
            "📋 What are the missing values?",
        ]
    elif "cleaning" in page_clean:
        return [
            "🧹 What cleaning actions do you recommend?",
            "🛡️ How will cleaning improve ML accuracy?",
            "❓ Why was this column modified?",
            "📋 Check duplicate records",
        ]
    elif "eda" in page_clean:
        return [
            "🔬 What are the main insights?",
            "🚨 Find unusual patterns and outliers",
            "📊 Which variables have strong correlation?",
            "📷 What is PCA?",
        ]
    elif "feature" in page_clean:
        return [
            "⚙️ What features should I engineer?",
            "📊 Which features are most important?",
            "📈 Explain power transforms (Yeo-Johnson)",
            "⚖️ Regression vs Classification",
        ]
    elif "machine" in page_clean or "ml" in page_clean:
        return [
            "🏆 Which model is best for my data?",
            "🌲 Random Forest vs XGBoost",
            "🎯 What are the most important features?",
            "⚖️ Difference between Regression & Classification",
        ]
    elif "forecast" in page_clean:
        return [
            "📈 Explain this forecast trajectory",
            "🔮 What is the expected trend and growth?",
            "⏱️ How does the forecasting engine work?",
            "📊 Show confidence intervals",
        ]
    elif "sql" in page_clean:
        return [
            "⚡ Show top 5 categories by sales",
            "📊 Average numbers grouped by region",
            "🔍 Explain how Natural Language to SQL works",
            "⚡ Why is DuckDB so fast?",
        ]
    elif "dashboard" in page_clean or "report" in page_clean:
        return [
            "📊 Summarize this dashboard",
            "🎯 What are the most important KPIs?",
            "📄 Generate an executive report",
            "💡 How does AI generate insights?",
        ]
    else:
        return [
            "📋 Summarize my dataset",
            "🛡️ Check data quality",
            "🏆 Which model is best?",
            "💡 What does this project do?",
        ]


def render_floating_ai_copilot():
    """
    Renders the persistent floating launcher button [ ✨ AI ] anchored at the bottom-right.
    Expands into the full Quantum AI Data Analyst Assistant window.
    """
    init_copilot_state()

    df = st.session_state.get("df")
    current_page = st.session_state.get("active_nav", "Dashboard")
    champ_info = get_active_champion()
    dataset_name = st.session_state.get("current_project_name", "Active Dataset")
    target_col = st.session_state.get("ml_target_col")

    ai_engine = QuantumAIEngine()

    # Handle pending query from chips, quick prompts, or voice
    if st.session_state.get("copilot_pending_query"):
        user_q = st.session_state["copilot_pending_query"]
        st.session_state["copilot_pending_query"] = None
        _process_question(user_q, ai_engine, df, current_page, champ_info, dataset_name, target_col)

    # Handle voice playback
    if st.session_state.get("copilot_voice_to_play"):
        voice_script = st.session_state["copilot_voice_to_play"]
        st.session_state["copilot_voice_to_play"] = None
        _render_speech_synthesis(voice_script)

    if st.session_state.get("copilot_stop_speech"):
        st.session_state["copilot_stop_speech"] = False
        _render_stop_speech()

    is_open = st.session_state.get("copilot_open", False)
    is_minimized = st.session_state.get("copilot_minimized", False)

    # 1. Bulletproof CSS for Floating Launcher & Window
    st.markdown("""
    <style>
    /* Floating Launcher Marker */
    div[data-testid="stElementContainer"]:has(.quantum-ai-anchor) {
        display: none !important;
    }

    div[data-testid="stElementContainer"]:has(.quantum-ai-anchor) + div[data-testid="stElementContainer"] {
        position: fixed !important;
        bottom: 24px !important;
        right: 28px !important;
        z-index: 9999999 !important;
        width: 62px !important;
        height: 62px !important;
    }

    div[data-testid="stElementContainer"]:has(.quantum-ai-anchor) + div[data-testid="stElementContainer"] button {
        background: linear-gradient(135deg, #00F0FF 0%, #0284C7 100%) !important;
        color: #0B0F19 !important;
        font-weight: 900 !important;
        font-size: 1.55rem !important;
        border-radius: 50% !important;
        width: 62px !important;
        height: 62px !important;
        min-width: 62px !important;
        min-height: 62px !important;
        padding: 0 !important;
        box-shadow: 0 8px 30px rgba(0, 240, 255, 0.75), 0 0 16px rgba(0, 240, 255, 0.45) !important;
        border: 2.5px solid rgba(255, 255, 255, 0.95) !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        animation: quantum-pulse 2.8s infinite !important;
    }

    @keyframes quantum-pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0.7), 0 8px 30px rgba(0, 240, 255, 0.75); }
        70% { box-shadow: 0 0 0 14px rgba(0, 240, 255, 0), 0 8px 30px rgba(0, 240, 255, 0.75); }
        100% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0), 0 8px 30px rgba(0, 240, 255, 0.75); }
    }

    div[data-testid="stElementContainer"]:has(.quantum-ai-anchor) + div[data-testid="stElementContainer"] button:hover {
        transform: scale(1.14) translateY(-3px) !important;
        box-shadow: 0 12px 45px rgba(0, 240, 255, 1), 0 0 25px rgba(0, 240, 255, 0.8) !important;
    }

    /* Pinned Slide-Up Quantum AI Window */
    div[data-testid="stElementContainer"]:has(.quantum-ai-window-pin) {
        display: none !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .quantum-ai-window-pin) {
        position: fixed !important;
        bottom: 96px !important;
        right: 28px !important;
        z-index: 9999998 !important;
        width: 530px !important;
        max-width: calc(100vw - 36px) !important;
        max-height: 84vh !important;
        overflow-y: auto !important;
        background: rgba(11, 15, 25, 0.98) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border: 2px solid #00F0FF !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.95), 0 0 35px rgba(0, 240, 255, 0.45) !important;
        padding: 16px !important;
    }

    /* Audio Wave Visualizer Animation */
    .audio-wave-anim {
        display: flex;
        align-items: center;
        gap: 3px;
        height: 16px;
    }
    .audio-wave-anim .aw-bar {
        width: 3px;
        background: #00F0FF;
        border-radius: 2px;
        animation: awPulse 1.2s infinite ease-in-out;
    }
    .audio-wave-anim .aw-bar1 { height: 6px; animation-delay: 0.1s; }
    .audio-wave-anim .aw-bar2 { height: 13px; animation-delay: 0.3s; }
    .audio-wave-anim .aw-bar3 { height: 16px; animation-delay: 0.5s; }
    .audio-wave-anim .aw-bar4 { height: 11px; animation-delay: 0.2s; }
    .audio-wave-anim .aw-bar5 { height: 7px; animation-delay: 0.4s; }

    @keyframes awPulse {
        0%, 100% { transform: scaleY(0.4); }
        50% { transform: scaleY(1.2); }
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. Render Floating Trigger Button (Round Action Button)
    st.markdown('<div class="quantum-ai-anchor"></div>', unsafe_allow_html=True)
    launcher_label = "✖" if is_open else "✨"
    if st.button(
        launcher_label,
        key="quantum_ai_floating_launcher_btn",
        type="primary" if not is_open else "secondary",
        help="Quantum AI - Intelligent Data Analyst Assistant (Click to open)",
    ):
        st.session_state["copilot_open"] = not is_open
        st.session_state["copilot_minimized"] = False
        st.rerun()

    # 3. If Open, render the Quantum AI Assistant Window
    if is_open:
        with st.container():
            st.markdown('<div class="quantum-ai-window-pin"></div>', unsafe_allow_html=True)

            is_voice_active = st.session_state.get("copilot_voice_active", False)

            # Window Header with Controls: Unmute/Voice, Minimize, New Chat, Clear, Close
            h_col_info, h_col_actions = st.columns([4.7, 5.3])
            with h_col_info:
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.35rem;">✨</span>
                    <div>
                        <div style="font-weight: 800; color: #FFFFFF; font-size: 1.02rem; letter-spacing: -0.01em;">Quantum Analyst AI</div>
                        <div style="font-size: 0.70rem; color: #00F0FF; font-weight: 700; letter-spacing: 0.05em;">YOUR DATA ANALYST COPILOT</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with h_col_actions:
                c_unmute, c_min, c_new, c_clear, c_close = st.columns([1.9, 0.75, 0.75, 0.75, 0.75], gap="small")
                with c_unmute:
                    if not is_voice_active:
                        if st.button("🔇 Unmute", key="copilot_voice_toggle_btn", help="Click to activate voice assistance (speaks responses out loud)"):
                            st.session_state["copilot_voice_active"] = True
                            for msg in reversed(st.session_state.get("copilot_messages", [])):
                                if msg.get("role") == "assistant" and msg.get("voice_text"):
                                    st.session_state["copilot_voice_to_play"] = msg["voice_text"]
                                    break
                            st.rerun()
                    else:
                        if st.button("🔊 Voice On", key="copilot_voice_toggle_btn", help="Mute voice assistance (return to silent chat)"):
                            st.session_state["copilot_voice_active"] = False
                            st.session_state["copilot_stop_speech"] = True
                            st.rerun()

                with c_min:
                    min_icon = "＋" if is_minimized else "−"
                    if st.button(min_icon, key="copilot_min_btn", help="Minimize / Expand Assistant"):
                        st.session_state["copilot_minimized"] = not is_minimized
                        st.rerun()
                with c_new:
                    if st.button("➕", key="copilot_new_chat_btn", help="Start New Conversation"):
                        st.session_state["copilot_messages"] = [
                            {
                                "role": "assistant",
                                "content": "✨ **New Chat Started.** Ask anything about your dataset insights, data quality, or machine learning algorithms!",
                                "voice_text": "New chat started. What would you like to explore or analyze?",
                                "actions": ["Summarize dataset", "Check data quality", "Which model is best?"],
                            }
                        ]
                        st.rerun()
                with c_clear:
                    if st.button("🧹", key="copilot_clear_chat_btn", help="Clear Conversation"):
                        st.session_state["copilot_messages"] = []
                        st.rerun()
                with c_close:
                    if st.button("✖", key="copilot_close_header_btn", help="Close Assistant"):
                        st.session_state["copilot_open"] = False
                        st.session_state["copilot_stop_speech"] = True
                        st.rerun()

            # If Minimized, don't show body
            if is_minimized:
                st.caption("Assistant minimized. Click '＋' or round '✨' button to expand.")
                return

            # Active Telemetry & Page Context Ribbon
            champ_name = champ_info.get("model_name", "None") if champ_info else "None"
            target_str = str(target_col) if target_col else "Auto"
            st.markdown(f"""
            <div style="background: rgba(0, 240, 255, 0.06); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 9px; padding: 6px 12px; margin: 6px 0; font-size: 0.76rem; color: #CBD5E1; display: flex; justify-content: space-between; align-items: center;">
                <div>📍 <b>Page:</b> <span style="color: #00F0FF;">{current_page}</span></div>
                <div>📁 <b>Data:</b> {dataset_name[:12]}</div>
                <div>🏆 <b>Model:</b> {champ_name[:12]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Enhanced Voice Visualizer & Status Bar
            if is_voice_active:
                st.markdown("""
                <div style="background: linear-gradient(90deg, rgba(0, 240, 255, 0.12) 0%, rgba(16, 185, 129, 0.12) 100%); border: 1px solid #00F0FF; border-radius: 9px; padding: 6px 12px; margin: 4px 0 8px 0; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981;"></span>
                        <span style="font-size: 0.76rem; font-weight: 800; color: #00F0FF; letter-spacing: 0.04em;">VOICE ASSISTANT ACTIVE</span>
                        <span style="font-size: 0.70rem; color: #94A3B8;">(Auto-Speaking Answers Aloud)</span>
                    </div>
                    <div class="audio-wave-anim">
                        <span class="aw-bar aw-bar1"></span>
                        <span class="aw-bar aw-bar2"></span>
                        <span class="aw-bar aw-bar3"></span>
                        <span class="aw-bar aw-bar4"></span>
                        <span class="aw-bar aw-bar5"></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 9px; padding: 6px 12px; margin: 4px 0 8px 0; display: flex; align-items: center; justify-content: space-between;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">
                        💬 <b>Chat Assistance Active</b> (Silent mode)
                    </div>
                    <div style="font-size: 0.70rem; color: #38BDF8; font-weight: 700;">
                        Click 🔇 Unmute for Voice Mode
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Suggested Questions (Context-Aware based on active page)
            suggestions = get_context_suggested_questions(current_page, has_data=(df is not None and not df.empty))
            st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #94A3B8; margin: 4px 0 2px 0;'>SUGGESTED QUESTIONS:</div>", unsafe_allow_html=True)

            q_c1, q_c2 = st.columns(2)
            with q_c1:
                if st.button(suggestions[0], key="sugg_chip_0", use_container_width=True):
                    st.session_state["copilot_pending_query"] = suggestions[0].split(" ", 1)[-1]
                    st.rerun()
                if st.button(suggestions[1], key="sugg_chip_1", use_container_width=True):
                    st.session_state["copilot_pending_query"] = suggestions[1].split(" ", 1)[-1]
                    st.rerun()
            with q_c2:
                if st.button(suggestions[2], key="sugg_chip_2", use_container_width=True):
                    st.session_state["copilot_pending_query"] = suggestions[2].split(" ", 1)[-1]
                    st.rerun()
                if st.button(suggestions[3], key="sugg_chip_3", use_container_width=True):
                    st.session_state["copilot_pending_query"] = suggestions[3].split(" ", 1)[-1]
                    st.rerun()

            st.markdown("<hr style='margin: 8px 0; border-color: #1E293B;'>", unsafe_allow_html=True)

            # Scrollable Message Feed
            msg_feed = st.container(height=310)
            with msg_feed:
                for idx, msg in enumerate(st.session_state["copilot_messages"]):
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                            <div style="background: #1E293B; border: 1px solid #334155; color: #E2E8F0; padding: 8px 14px; border-radius: 14px 14px 2px 14px; max-width: 85%; font-size: 0.88rem;">
                                <b>You:</b> {msg['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(0, 240, 255, 0.35); color: #F1F5F9; padding: 12px 16px; border-radius: 14px 14px 14px 2px; max-width: 95%; font-size: 0.88rem; line-height: 1.55;">
                                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                                    <span style="color: #00F0FF; font-weight: 800; font-size: 0.74rem; letter-spacing: 0.05em;">✨ QUANTUM AI</span>
                                    <span style="font-size: 0.68rem; color: #64748B;">{msg.get('tool_badge', 'Calculated Engine')}</span>
                                </div>
                                {msg['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Display tabular data if attached
                        if msg.get("data") is not None:
                            try:
                                st.dataframe(msg["data"], use_container_width=True)
                            except Exception:
                                pass

                        # Action Buttons and Voice playback bar
                        act_c1, act_c2 = st.columns([7, 3])
                        with act_c1:
                            actions = msg.get("actions", [])
                            if actions:
                                act_cols = st.columns(min(3, len(actions)))
                                for a_idx, act_label in enumerate(actions[:3]):
                                    with act_cols[a_idx]:
                                        if st.button(f"👉 {act_label}", key=f"act_btn_{idx}_{a_idx}", use_container_width=True):
                                            st.session_state["copilot_pending_query"] = act_label.replace("👉", "").strip()
                                            st.rerun()

                        with act_c2:
                            if msg.get("voice_text"):
                                v_b1, v_b2 = st.columns([1.2, 1])
                                with v_b1:
                                    if st.button("🔊 Listen", key=f"voice_play_{idx}", use_container_width=True):
                                        st.session_state["copilot_voice_to_play"] = msg["voice_text"]
                                        st.rerun()
                                with v_b2:
                                    if st.button("⏹️", key=f"voice_stop_{idx}", help="Stop Speech"):
                                        st.session_state["copilot_stop_speech"] = True
                                        st.rerun()

            st.markdown("<hr style='margin: 8px 0; border-color: #1E293B;'>", unsafe_allow_html=True)

            # Voice Input + Question Input Bar
            c_mic, c_input, c_btn = st.columns([1.1, 7.1, 1.8])
            with c_mic:
                _render_voice_input_button()
            with c_input:
                user_text = st.text_input(
                    "Ask anything about your data:",
                    placeholder="Ask anything or click 🎙️ to talk...",
                    label_visibility="collapsed",
                    key="quantum_ai_text_input_main",
                )
            with c_btn:
                if st.button("➤ Send", type="primary", use_container_width=True, key="quantum_ai_send_btn"):
                    if user_text and user_text.strip():
                        _process_question(user_text.strip(), ai_engine, df, current_page, champ_info, dataset_name, target_col)
                        st.rerun()

            # Dedicated full-page view switch
            if st.button("↗️ Open Dedicated Full-Page Workspace", key="copilot_full_view_link_btn", use_container_width=True):
                st.session_state["copilot_open"] = False
                st.session_state["active_nav"] = "AI Model Copilot"
                st.rerun()


def _process_question(
    question: str,
    engine: QuantumAIEngine,
    df: Optional[pd.DataFrame],
    current_page: str,
    champ_info: Optional[Dict[str, Any]],
    dataset_name: Optional[str],
    target_col: Optional[str],
):
    """Processes user question through QuantumAIEngine and updates conversation."""
    st.session_state["copilot_messages"].append({
        "role": "user",
        "content": question,
    })

    ans_dict = engine.process_query(
        query=question,
        df=df,
        current_page=current_page,
        conversation_history=st.session_state["copilot_messages"],
        champ_info=champ_info,
        dataset_name=dataset_name,
        target_col=target_col,
    )

    st.session_state["copilot_messages"].append({
        "role": "assistant",
        "content": ans_dict["markdown"],
        "voice_text": ans_dict.get("voice_text"),
        "tool_badge": f"Tool: {ans_dict.get('tool_used', 'Calculated Engine')}",
        "data": ans_dict.get("data"),
        "actions": ans_dict.get("actions", []),
    })

    # Auto-queue spoken response only when voice assistance is active (unmuted)
    if st.session_state.get("copilot_voice_active") and ans_dict.get("voice_text"):
        st.session_state["copilot_voice_to_play"] = ans_dict["voice_text"]


def _process_copilot_question(
    question: str,
    engine: Any = None,
    df: Optional[pd.DataFrame] = None,
    champ_info: Optional[Dict[str, Any]] = None,
    bench_res: Optional[Dict[str, Any]] = None,
    current_page: str = "AI Model Copilot",
    dataset_name: Optional[str] = None,
    target_col: Optional[str] = None,
):
    """Bridge for both QuantumAIEngine and legacy copilot calls."""
    if not isinstance(engine, QuantumAIEngine):
        real_engine = QuantumAIEngine()
    else:
        real_engine = engine

    _process_question(
        question=question,
        engine=real_engine,
        df=df if df is not None else st.session_state.get("df"),
        current_page=current_page,
        champ_info=champ_info or get_active_champion(),
        dataset_name=dataset_name or st.session_state.get("current_project_name", "Active Dataset"),
        target_col=target_col or st.session_state.get("ml_target_col"),
    )


def _render_speech_synthesis(text: str):
    """Injects Web Speech API Text-to-Speech into the browser."""
    clean_json_text = json.dumps(text)
    script_html = f"""
    <script>
    (function() {{
        try {{
            const win = window.parent || window;
            if ('speechSynthesis' in win) {{
                win.speechSynthesis.cancel();
                const msg = new win.SpeechSynthesisUtterance({clean_json_text});
                msg.rate = 1.0;
                msg.pitch = 1.0;
                msg.lang = 'en-US';

                const voices = win.speechSynthesis.getVoices();
                const preferredVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha')));
                if (preferredVoice) msg.voice = preferredVoice;

                win.speechSynthesis.speak(msg);
            }}
        }} catch(e) {{
            console.warn("Speech synthesis error: ", e);
        }}
    }})();
    </script>
    """
    components.html(script_html, height=0, width=0)


def _render_stop_speech():
    """Cancels any active speech playback."""
    script_html = """
    <script>
    (function() {
        try {
            const win = window.parent || window;
            if ('speechSynthesis' in win) {
                win.speechSynthesis.cancel();
            }
        } catch(e) {}
    })();
    </script>
    """
    components.html(script_html, height=0, width=0)


def _render_voice_input_button():
    """Renders interactive Web Speech Recognition microphone trigger in browser with active status states."""
    components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { margin: 0; padding: 0; background: transparent; overflow: hidden; display: flex; align-items: center; justify-content: center; height: 100%; }
                #micBtn {
                    background: #1E293B;
                    border: 1px solid #00F0FF;
                    color: #00F0FF;
                    border-radius: 8px;
                    width: 38px;
                    height: 38px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                    transition: all 0.2s ease;
                }
                #micBtn:hover {
                    background: #00F0FF;
                    color: #0B0F19;
                    box-shadow: 0 0 10px #00F0FF;
                }
                #micBtn.listening {
                    background: #EF4444;
                    color: #FFFFFF;
                    border-color: #EF4444;
                    animation: pulse 1s infinite;
                }
                @keyframes pulse {
                    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
                    70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
                }
            </style>
        </head>
        <body>
            <button id="micBtn" title="Speak Question (Voice Input)">🎙️</button>
            <script>
                const btn = document.getElementById('micBtn');
                const targetWindow = window.parent || window;
                const SpeechRecognition = targetWindow.SpeechRecognition || targetWindow.webkitSpeechRecognition || window.SpeechRecognition || window.webkitSpeechRecognition;

                if (!SpeechRecognition) {
                    btn.title = "Speech recognition not supported in this browser. You can type instead.";
                    btn.style.opacity = "0.5";
                } else {
                    const recognition = new SpeechRecognition();
                    recognition.lang = 'en-US';
                    recognition.interimResults = false;
                    recognition.maxAlternatives = 1;

                    btn.addEventListener('click', () => {
                        try {
                            btn.classList.add('listening');
                            btn.title = "Listening... Speak your question now";
                            recognition.start();
                        } catch(e) {
                            recognition.stop();
                            btn.classList.remove('listening');
                            btn.title = "Speak Question (Voice Input)";
                        }
                    });

                    recognition.onresult = (event) => {
                        btn.classList.remove('listening');
                        btn.title = "Speak Question (Voice Input)";
                        const transcript = event.results[0][0].transcript;
                        const parentInputs = targetWindow.document.querySelectorAll('input[type="text"]');
                        for (let inp of parentInputs) {
                            if (inp.placeholder && (inp.placeholder.includes('Ask') || inp.placeholder.includes('talk') || inp.placeholder.includes('data') || inp.placeholder.includes('question'))) {
                                inp.value = transcript;
                                inp.dispatchEvent(new Event('input', { bubbles: true }));
                                inp.dispatchEvent(new Event('change', { bubbles: true }));
                                break;
                            }
                        }
                    };

                    recognition.onerror = () => {
                        btn.classList.remove('listening');
                        btn.title = "Microphone access disabled or unavailable. You can type instead.";
                    };

                    recognition.onend = () => {
                        btn.classList.remove('listening');
                        btn.title = "Speak Question (Voice Input)";
                    };
                }
            </script>
        </body>
        </html>
        """,
        height=40,
        width=45,
    )
