"""
ui.views.copilot_view
=====================
Dedicated Full-Page View for Quantum AI Model Copilot & Voice Assistant.
Allows users to ask about any ML algorithm in simple language, explore analogies,
understand business use cases, and interact using Speech-to-Text & Text-to-Speech.
"""

import streamlit as st
import pandas as pd
from core.ml.model_copilot import ModelCopilotEngine, MODEL_ENCYCLOPEDIA
from core.ml.experiment_tracker import get_active_champion
from ui.copilot_widget import _process_copilot_question, _render_speech_synthesis, _render_voice_input_button, init_copilot_state


def render_copilot_full_view():
    init_copilot_state()

    df = st.session_state.get("df")
    champ_info = get_active_champion()
    bench_res = st.session_state.get("ml_benchmark_results")
    copilot_engine = ModelCopilotEngine()

    # Header
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <div style="font-size: 0.78rem; font-weight: 800; color: #00F0FF; letter-spacing: 0.08em; text-transform: uppercase;">
            // CONVERSATIONAL REASONING & VOICE ASSISTANT
        </div>
        <h2 style="margin: 2px 0 6px 0; color: #FFFFFF; font-size: 1.8rem; font-weight: 800;">
            🤖 Quantum AI Model Copilot & Voice Assistant
        </h2>
        <div style="font-size: 0.88rem; color: #94A3B8;">
            Demystifying machine learning in plain, intuitive language with real-world business use cases and hands-free voice interaction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active Context Telemetry Card
    ds_name = st.session_state.get("current_project_name", "No Dataset Loaded")
    champ_name = champ_info.get("model_name") if champ_info else (bench_res.get("champion_name") if (bench_res and isinstance(bench_res, dict)) else "None Trained Yet")
    target_v = st.session_state.get("ml_target_col", "None Configured")
    task_t = st.session_state.get("ml_task_type", "auto").replace("_", " ").title()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Active Dataset", ds_name)
    with c2:
        st.metric("Production Champion", champ_name)
    with c3:
        st.metric("Prediction Target", str(target_v))
    with c4:
        st.metric("Task Formulation", task_t)

    st.markdown("<hr style='border-color: #1E293B; margin: 16px 0;'>", unsafe_allow_html=True)

    # If audio voice synthesis is triggered, render invisible Web Speech audio tag
    if st.session_state.get("copilot_voice_to_play"):
        voice_script = st.session_state["copilot_voice_to_play"]
        st.session_state["copilot_voice_to_play"] = None
        _render_speech_synthesis(voice_script)

    # Handle pending query from quick chips
    if st.session_state.get("copilot_pending_query"):
        user_q = st.session_state["copilot_pending_query"]
        st.session_state["copilot_pending_query"] = None
        _process_copilot_question(user_q, copilot_engine, df, champ_info, bench_res)

    # Layout: 2 Columns (Left: Chat Feed & Voice Input, Right: Interactive Model Catalog & Analogies)
    col_chat, col_catalog = st.columns([6, 4])

    with col_chat:
        st.markdown("#### 💬 Interactive Dialogue & Voice Questions")

        # Quick Question Chips
        st.markdown("<div style='font-size: 0.76rem; font-weight: 700; color: #94A3B8; margin-bottom: 6px;'>POPULAR INQUIRIES (CLICK TO ASK):</div>", unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        with q1:
            if st.button("💡 Explain all models simply", key="fp_chip_1", width="stretch"):
                st.session_state["copilot_pending_query"] = "Explain all models in simple words"
                st.rerun()
            if st.button("🌲 Random Forest vs XGBoost", key="fp_chip_2", width="stretch"):
                st.session_state["copilot_pending_query"] = "Compare Random Forest vs XGBoost in simple language"
                st.rerun()
        with q2:
            if st.button("📷 Explain PCA in 1 minute", key="fp_chip_3", width="stretch"):
                st.session_state["copilot_pending_query"] = "Explain PCA simply with analogy"
                st.rerun()
            if st.button("🚨 What is Anomaly Detection?", key="fp_chip_4", width="stretch"):
                st.session_state["copilot_pending_query"] = "How does anomaly detection work in business?"
                st.rerun()

        # Chat Conversation Container
        chat_container = st.container(height=480)
        with chat_container:
            for idx, msg in enumerate(st.session_state["copilot_messages"]):
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                        <div style="background: #1E293B; border: 1px solid #334155; color: #E2E8F0; padding: 10px 16px; border-radius: 16px 16px 2px 16px; max-width: 85%; font-size: 0.9rem;">
                            <b>You:</b> {msg['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 12px;">
                        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(0, 240, 255, 0.35); color: #F1F5F9; padding: 14px 18px; border-radius: 16px 16px 16px 2px; max-width: 95%; font-size: 0.9rem; line-height: 1.6;">
                            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px;">
                                <span style="color: #00F0FF; font-weight: 800; font-size: 0.78rem;">🤖 QUANTUM MODEL COPILOT</span>
                            </div>
                            {msg['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if msg.get("voice_text"):
                        if st.button("🔊 Listen to Explanation", key=f"fp_voice_speak_{idx}"):
                            st.session_state["copilot_voice_to_play"] = msg["voice_text"]
                            st.rerun()

        # Voice Input + Text Input Bar
        st.markdown("<br>", unsafe_allow_html=True)
        cin_mic, cin_txt, cin_btn = st.columns([1, 7, 2])
        with cin_mic:
            _render_voice_input_button()
        with cin_txt:
            user_msg = st.text_input(
                "Ask a model question:",
                placeholder="Type or click 🎙️ to speak: e.g. What is the best model? Explain Ridge vs Lasso...",
                label_visibility="collapsed",
                key="fp_copilot_text_input",
            )
        with cin_btn:
            if st.button("⚡ Ask AI", type="primary", width="stretch", key="fp_copilot_send_btn"):
                if user_msg.strip():
                    _process_copilot_question(user_msg, copilot_engine, df, champ_info, bench_res)
                    st.rerun()

    # Right Column: Model Encyclopedia & Quick Lookup
    with col_catalog:
        st.markdown("#### 📖 Algorithm Encyclopedia & Analogies")
        st.caption("Select any model to instantly see its layman analogy, business uses, and pros & cons.")

        all_models = list(MODEL_ENCYCLOPEDIA.keys())
        selected_m = st.selectbox(
            "Browse Model Knowledge:",
            options=all_models,
            format_func=lambda k: f"{MODEL_ENCYCLOPEDIA[k]['name']} ({MODEL_ENCYCLOPEDIA[k]['family']})",
            index=all_models.index("xgboost_classifier") if "xgboost_classifier" in all_models else 0,
        )

        m_info = MODEL_ENCYCLOPEDIA[selected_m]
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1E293B; border-radius: 10px; padding: 16px;">
            <div style="font-weight: 800; color: #00F0FF; font-size: 1.1rem; margin-bottom: 4px;">{m_info['name']}</div>
            <div style="font-size: 0.75rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; margin-bottom: 10px;">{m_info['family']}</div>
            
            <div style="background: rgba(0, 240, 255, 0.05); border-left: 3px solid #00F0FF; padding: 8px 12px; border-radius: 4px; font-style: italic; font-size: 0.88rem; color: #E2E8F0; margin-bottom: 12px;">
                "{m_info['analogy']}"
            </div>

            <div style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 12px; line-height: 1.5;">
                {m_info['simple_explanation']}
            </div>

            <div style="font-size: 0.8rem; font-weight: 700; color: #38BDF8; margin-bottom: 4px;">Top Business Applications:</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 10px;">
                {'<br>'.join(m_info['business_uses'])}
            </div>

            <div style="display: flex; gap: 12px; font-size: 0.8rem;">
                <div><b style="color: #10B981;">Pros:</b> {m_info['pros'][0]}</div>
                <div><b style="color: #F59E0B;">Watch Out:</b> {m_info['cons'][0]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"🔊 Listen to {m_info['name']} Breakdown", key=f"speak_encyc_{selected_m}", width="stretch"):
            speech_script = (
                f"{m_info['name']}. In simple terms: {m_info['analogy']} "
                f"{m_info['simple_explanation']} "
                f"A common business use case is {m_info['business_uses'][0]}."
            )
            st.session_state["copilot_voice_to_play"] = speech_script
            st.rerun()
