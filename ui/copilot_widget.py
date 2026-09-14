"""
ui.copilot_widget
=================
Floating AI Model Copilot & Voice Assistant Widget for Quantum Analyst.
Renders an interactive floating button on the bottom-right of the screen that expands
into an AI Assistant specialized in explaining models in simple language with real-world
business uses, intuitive analogies, and native Web Speech Voice Input & Output.
"""

import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from core.ml.model_copilot import ModelCopilotEngine
from core.ml.experiment_tracker import get_active_champion


def init_copilot_state():
    """Initializes session state variables for the floating AI Copilot."""
    if "copilot_open" not in st.session_state:
        st.session_state["copilot_open"] = False
    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am your AI Model Copilot & Voice Assistant.**\n\n"
                    "I can explain every machine learning algorithm in simple, everyday words, "
                    "give you intuitive analogies, break down business use cases, or explain why "
                    "your champion model won on this dataset.\n\n"
                    "Click any prompt chip below, type a question, or click the **🎙️ Microphone** to talk!"
                ),
                "voice_text": (
                    "Hello! I am your AI Model Copilot. You can ask me to explain any model in simple language, "
                    "compare algorithms like Random Forest and XGBoost, or explain PCA and anomaly detection."
                ),
            }
        ]
    if "copilot_voice_to_play" not in st.session_state:
        st.session_state["copilot_voice_to_play"] = None
    if "copilot_pending_query" not in st.session_state:
        st.session_state["copilot_pending_query"] = None


def render_floating_ai_copilot():
    """Renders the persistent floating AI Copilot on the bottom-right corner."""
    init_copilot_state()

    df = st.session_state.get("df")
    champ_info = get_active_champion()
    bench_res = st.session_state.get("ml_benchmark_results")
    copilot_engine = ModelCopilotEngine()

    # Handle pending query from quick suggestion chip or voice
    if st.session_state.get("copilot_pending_query"):
        user_q = st.session_state["copilot_pending_query"]
        st.session_state["copilot_pending_query"] = None
        _process_copilot_question(user_q, copilot_engine, df, champ_info, bench_res)

    # 1. Custom CSS for Floating Anchor Button
    st.markdown("""
    <style>
    /* Floating Copilot Fixed Container */
    .copilot-floating-anchor {
        position: fixed;
        bottom: 24px;
        right: 28px;
        z-index: 999999;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }
    .copilot-pulse-badge {
        background: linear-gradient(135deg, #00F0FF, #3B82F6);
        color: #0B0F19;
        font-weight: 800;
        font-size: 0.85rem;
        padding: 10px 18px;
        border-radius: 30px;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.45);
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
        transition: all 0.25s ease-in-out;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .copilot-pulse-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 0 28px rgba(0, 240, 255, 0.7);
    }
    .copilot-drawer {
        background: rgba(11, 15, 25, 0.95);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid #00F0FF;
        box-shadow: 0 10px 35px rgba(0, 240, 255, 0.25);
        border-radius: 14px;
        padding: 18px;
        margin-top: 14px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # If audio voice synthesis is triggered, render invisible Web Speech audio tag
    if st.session_state.get("copilot_voice_to_play"):
        voice_script = st.session_state["copilot_voice_to_play"]
        st.session_state["copilot_voice_to_play"] = None
        _render_speech_synthesis(voice_script)

    # 2. Floating Bottom-Right Controls
    is_open = st.session_state.get("copilot_open", False)

    # Floating trigger bar
    f_c1, f_c2 = st.columns([8, 2])
    with f_c2:
        btn_label = "✖ Close AI Copilot" if is_open else "🤖 AI Model Copilot (Voice Ready)"
        if st.button(btn_label, type="primary" if not is_open else "secondary", key="copilot_toggle_btn", use_container_width=True):
            st.session_state["copilot_open"] = not is_open
            st.rerun()

    # 3. If Open, render the Copilot Window
    if is_open:
        st.markdown("""
        <div class="copilot-drawer">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E293B; padding-bottom: 10px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.3rem;">🤖</span>
                    <div>
                        <div style="font-weight: 800; color: #FFFFFF; font-size: 1.05rem;">Quantum AI Model Copilot</div>
                        <div style="font-size: 0.72rem; color: #00F0FF; font-weight: 700; letter-spacing: 0.05em;">● VOICE & REASONING ENGINE ACTIVE</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Context Ribbon
        ds_name = st.session_state.get("current_project_name", "No Active Dataset")
        champ_name = champ_info.get("model_name") if champ_info else (bench_res.get("champion_name") if bench_res else "None Trained")
        target_v = st.session_state.get("ml_target_col", "None")

        c_ctx1, c_ctx2, c_ctx3, c_ctx4 = st.columns([3, 3, 2, 2])
        with c_ctx1:
            st.caption(f"📁 **Dataset**: `{ds_name}`")
        with c_ctx2:
            st.caption(f"🏆 **Champion**: `{champ_name}`")
        with c_ctx3:
            st.caption(f"🎯 **Target**: `{target_v}`")
        with c_ctx4:
            if st.button("🧹 Clear Chat", key="copilot_clear_chat_btn"):
                st.session_state["copilot_messages"] = []
                st.rerun()

        # One-Click Suggestion Chips
        st.markdown("<div style='font-size: 0.76rem; font-weight: 700; color: #94A3B8; margin-top: 6px; margin-bottom: 4px;'>QUICK INQUIRY CHIPS:</div>", unsafe_allow_html=True)
        q_cols = st.columns(4)
        with q_cols[0]:
            if st.button("💡 Explain all models simply", key="chip_1", use_container_width=True):
                st.session_state["copilot_pending_query"] = "Explain all models in simple words"
                st.rerun()
        with q_cols[1]:
            if st.button("🌲 Random Forest vs XGBoost", key="chip_2", use_container_width=True):
                st.session_state["copilot_pending_query"] = "Compare Random Forest vs XGBoost in simple language"
                st.rerun()
        with q_cols[2]:
            if st.button("📷 Explain PCA in 1 minute", key="chip_3", use_container_width=True):
                st.session_state["copilot_pending_query"] = "Explain PCA simply with analogy"
                st.rerun()
        with q_cols[3]:
            if st.button("🚨 Anomaly detection in business", key="chip_4", use_container_width=True):
                st.session_state["copilot_pending_query"] = "How does anomaly detection work in business?"
                st.rerun()

        st.markdown("<hr style='border-color: #1E293B; margin: 12px 0;'>", unsafe_allow_html=True)

        # Message Feed
        for idx, msg in enumerate(st.session_state["copilot_messages"]):
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                    <div style="background: #1E293B; border: 1px solid #334155; color: #E2E8F0; padding: 10px 16px; border-radius: 16px 16px 2px 16px; max-width: 85%; font-size: 0.88rem;">
                        <b>You:</b> {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 12px;">
                    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(0, 240, 255, 0.3); color: #F1F5F9; padding: 14px 18px; border-radius: 16px 16px 16px 2px; max-width: 95%; font-size: 0.88rem; line-height: 1.6;">
                        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px;">
                            <span style="color: #00F0FF; font-weight: 800; font-size: 0.78rem;">🤖 QUANTUM MODEL COPILOT</span>
                        </div>
                        {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Voice Output Button for Assistant Response
                if msg.get("voice_text"):
                    v_col1, v_col2 = st.columns([1, 8])
                    with v_col1:
                        if st.button("🔊 Listen", key=f"voice_speak_{idx}"):
                            st.session_state["copilot_voice_to_play"] = msg["voice_text"]
                            st.rerun()

        st.markdown("<hr style='border-color: #1E293B; margin: 12px 0;'>", unsafe_allow_html=True)

        # Voice Input & Text Input Bar
        st.markdown("##### 💬 Ask a Model Question or Use Voice:")
        c_mic, c_input, c_btn = st.columns([1, 7, 2])

        with c_mic:
            _render_voice_input_button()

        with c_input:
            user_text = st.text_input(
                "Your question:",
                placeholder="e.g. Which model is best for predicting sales? Why did my champion win? Explain Ridge vs Lasso...",
                label_visibility="collapsed",
                key="copilot_text_input",
            )

        with c_btn:
            if st.button("⚡ Ask Copilot", type="primary", use_container_width=True, key="copilot_send_btn"):
                if user_text.strip():
                    _process_copilot_question(user_text, copilot_engine, df, champ_info, bench_res)
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _process_copilot_question(
    question: str,
    engine: ModelCopilotEngine,
    df: Optional[pd.DataFrame],
    champ_info: Optional[Dict[str, Any]],
    bench_res: Optional[Dict[str, Any]],
):
    """Processes question through ModelCopilotEngine and appends to conversation."""
    st.session_state["copilot_messages"].append({
        "role": "user",
        "content": question,
    })

    ans_dict = engine.answer_model_question(
        query=question,
        df=df,
        champ_info=champ_info,
        benchmark_results=bench_res,
    )

    st.session_state["copilot_messages"].append({
        "role": "assistant",
        "content": ans_dict["markdown"],
        "voice_text": ans_dict.get("voice_text"),
    })

    # Auto-play speech for newly answered question
    if ans_dict.get("voice_text"):
        st.session_state["copilot_voice_to_play"] = ans_dict["voice_text"]


def _render_speech_synthesis(text: str):
    """Injects Web Speech API Text-to-Speech into browser to speak text aloud."""
    clean_json_text = json.dumps(text)
    components.html(
        f"""
        <script>
        (function() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel(); // cancel any active speech
                const msg = new SpeechSynthesisUtterance({clean_json_text});
                msg.rate = 1.0;
                msg.pitch = 1.0;
                msg.lang = 'en-US';

                // Try to find natural sounding English voice
                const voices = window.speechSynthesis.getVoices();
                const preferredVoice = voices.find(v => v.lang === 'en-US' && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha')));
                if (preferredVoice) {{
                    msg.voice = preferredVoice;
                }}

                window.speechSynthesis.speak(msg);
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _render_voice_input_button():
    """Renders interactive Web Speech Recognition microphone trigger in browser."""
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
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

                if (!SpeechRecognition) {
                    btn.title = "Speech recognition not supported in this browser.";
                    btn.style.opacity = "0.5";
                } else {
                    const recognition = new SpeechRecognition();
                    recognition.lang = 'en-US';
                    recognition.interimResults = false;
                    recognition.maxAlternatives = 1;

                    btn.addEventListener('click', () => {
                        try {
                            btn.classList.add('listening');
                            recognition.start();
                        } catch(e) {
                            recognition.stop();
                            btn.classList.remove('listening');
                        }
                    });

                    recognition.onresult = (event) => {
                        btn.classList.remove('listening');
                        const transcript = event.results[0][0].transcript;
                        // Populate Streamlit text input via DOM access if accessible
                        const parentInputs = window.parent.document.querySelectorAll('input[type="text"]');
                        for (let inp of parentInputs) {
                            if (inp.placeholder && inp.placeholder.includes('Which model is best')) {
                                inp.value = transcript;
                                inp.dispatchEvent(new Event('input', { bubbles: true }));
                                inp.dispatchEvent(new Event('change', { bubbles: true }));
                                break;
                            }
                        }
                    };

                    recognition.onerror = () => {
                        btn.classList.remove('listening');
                    };

                    recognition.onend = () => {
                        btn.classList.remove('listening');
                    };
                }
            </script>
        </body>
        </html>
        """,
        height=40,
        width=45,
    )
