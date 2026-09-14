"""
ui.copilot_widget
=================
Floating AI Model Copilot & Voice Assistant Widget for Quantum Analyst.
Renders an ultra-visible floating pill on the bottom-right of the screen that expands
into an AI Assistant specialized in explaining models in simple language with real-world
business uses, intuitive analogies, and native Web Speech Voice Input & Output.
"""

import json
from typing import Optional, Dict, Any, List
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

    # If audio voice synthesis is triggered, render invisible Web Speech audio tag
    if st.session_state.get("copilot_voice_to_play"):
        voice_script = st.session_state["copilot_voice_to_play"]
        st.session_state["copilot_voice_to_play"] = None
        _render_speech_synthesis(voice_script)

    is_open = st.session_state.get("copilot_open", False)

    # 1. Inject Bulletproof CSS for Fixed Bottom-Right Positioning
    st.markdown("""
    <style>
    /* Fixed Floating Copilot Launcher Container */
    div[data-testid="stElementContainer"]:has(.copilot-marker-pin) {
        display: none !important;
    }

    div[data-testid="stElementContainer"]:has(.copilot-marker-pin) + div[data-testid="stElementContainer"] {
        position: fixed !important;
        bottom: 24px !important;
        right: 28px !important;
        z-index: 9999999 !important;
        width: auto !important;
        background: transparent !important;
        display: block !important;
    }

    div[data-testid="stElementContainer"]:has(.copilot-marker-pin) + div[data-testid="stElementContainer"] button {
        background: linear-gradient(135deg, #00F0FF 0%, #0284C7 100%) !important;
        color: #0B0F19 !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        border-radius: 50px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 25px rgba(0, 240, 255, 0.75) !important;
        border: 2px solid rgba(255, 255, 255, 0.7) !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        transition: all 0.25s ease-in-out !important;
    }

    div[data-testid="stElementContainer"]:has(.copilot-marker-pin) + div[data-testid="stElementContainer"] button:hover {
        transform: scale(1.06) translateY(-2px) !important;
        box-shadow: 0 6px 35px rgba(0, 240, 255, 0.95) !important;
    }

    /* Pinned Slide-Up Copilot Chat Window */
    div[data-testid="stElementContainer"]:has(.copilot-drawer-pin) {
        display: none !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .copilot-drawer-pin) {
        position: fixed !important;
        bottom: 78px !important;
        right: 28px !important;
        z-index: 9999998 !important;
        width: 520px !important;
        max-width: calc(100vw - 36px) !important;
        max-height: 82vh !important;
        overflow-y: auto !important;
        background: rgba(11, 15, 25, 0.98) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 2px solid #00F0FF !important;
        border-radius: 18px !important;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.95), 0 0 35px rgba(0, 240, 255, 0.45) !important;
        padding: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. Inject DOM Floating Pill via JS into Parent Document
    # This guarantees that even if CSS selectors are ignored by older browsers,
    # the floating button is 100% visible in the bottom-right corner!
    _render_dom_floating_pill(is_open)

    # 3. Streamlit Native Trigger Button Container
    with st.container():
        st.markdown('<div class="copilot-marker-pin"></div>', unsafe_allow_html=True)
        btn_label = "✖ Close AI Copilot" if is_open else "🤖 AI Model Copilot (Voice Ready)"
        if st.button(btn_label, key="copilot_floating_btn", type="primary" if not is_open else "secondary"):
            st.session_state["copilot_open"] = not is_open
            st.rerun()

    # 4. If Open, render the Copilot Window pinned to bottom right
    if is_open:
        with st.container():
            st.markdown('<div class="copilot-drawer-pin"></div>', unsafe_allow_html=True)

            # Window Header
            h_c1, h_c2 = st.columns([8, 2])
            with h_c1:
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.4rem;">🤖</span>
                    <div>
                        <div style="font-weight: 800; color: #FFFFFF; font-size: 1.05rem;">Quantum AI Model Copilot</div>
                        <div style="font-size: 0.72rem; color: #00F0FF; font-weight: 700; letter-spacing: 0.05em;">● VOICE & REASONING ENGINE ACTIVE</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with h_c2:
                if st.button("🧹 Clear", key="copilot_clear_chat_btn_2", use_container_width=True):
                    st.session_state["copilot_messages"] = []
                    st.rerun()

            # Active Context Ribbon
            ds_name = st.session_state.get("current_project_name", "No Active Dataset")
            champ_name = champ_info.get("model_name") if champ_info else (bench_res.get("champion_name") if bench_res else "None Trained")
            target_v = st.session_state.get("ml_target_col", "None")

            st.markdown(f"""
            <div style="background: rgba(0, 240, 255, 0.06); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 6px 12px; margin: 10px 0; font-size: 0.78rem; color: #CBD5E1; display: flex; justify-content: space-between;">
                <div>📁 <b>Data:</b> {ds_name}</div>
                <div>🏆 <b>Model:</b> {champ_name}</div>
                <div>🎯 <b>Target:</b> {target_v}</div>
            </div>
            """, unsafe_allow_html=True)

            # Quick Question Chips
            st.markdown("<div style='font-size: 0.74rem; font-weight: 700; color: #94A3B8; margin-bottom: 4px;'>POPULAR INQUIRIES:</div>", unsafe_allow_html=True)
            q1, q2 = st.columns(2)
            with q1:
                if st.button("💡 Explain all models simply", key="chip_1", use_container_width=True):
                    st.session_state["copilot_pending_query"] = "Explain all models in simple words"
                    st.rerun()
                if st.button("🌲 Random Forest vs XGBoost", key="chip_2", use_container_width=True):
                    st.session_state["copilot_pending_query"] = "Compare Random Forest vs XGBoost in simple language"
                    st.rerun()
            with q2:
                if st.button("📷 Explain PCA in 1 minute", key="chip_3", use_container_width=True):
                    st.session_state["copilot_pending_query"] = "Explain PCA simply with analogy"
                    st.rerun()
                if st.button("🚨 Anomaly detection in business", key="chip_4", use_container_width=True):
                    st.session_state["copilot_pending_query"] = "How does anomaly detection work in business?"
                    st.rerun()

            st.markdown("<hr style='border-color: #1E293B; margin: 10px 0;'>", unsafe_allow_html=True)

            # Message Feed
            msg_box = st.container(height=320)
            with msg_box:
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
                            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(0, 240, 255, 0.35); color: #F1F5F9; padding: 12px 16px; border-radius: 14px 14px 14px 2px; max-width: 95%; font-size: 0.88rem; line-height: 1.5;">
                                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                                    <span style="color: #00F0FF; font-weight: 800; font-size: 0.74rem;">🤖 QUANTUM MODEL COPILOT</span>
                                </div>
                                {msg['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if msg.get("voice_text"):
                            if st.button(f"🔊 Listen to Voice Explanation", key=f"voice_speak_{idx}"):
                                st.session_state["copilot_voice_to_play"] = msg["voice_text"]
                                st.rerun()

            st.markdown("<hr style='border-color: #1E293B; margin: 10px 0;'>", unsafe_allow_html=True)

            # Voice Input + Question Input
            c_mic, c_input, c_btn = st.columns([1.2, 6.8, 2])
            with c_mic:
                _render_voice_input_button()
            with c_input:
                user_text = st.text_input(
                    "Your question:",
                    placeholder="Type or click 🎙️ to talk: e.g. What is the best model? Explain Ridge vs Lasso...",
                    label_visibility="collapsed",
                    key="copilot_text_input_bottom",
                )
            with c_btn:
                if st.button("⚡ Ask AI", type="primary", use_container_width=True, key="copilot_send_btn_bottom"):
                    if user_text.strip():
                        _process_copilot_question(user_text, copilot_engine, df, champ_info, bench_res)
                        st.rerun()

            # Direct shortcut to Full-Page Copilot View
            st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
            if st.button("↗️ Open Dedicated Full-Page Copilot", key="drawer_open_fullpage_btn", use_container_width=True):
                st.session_state["copilot_open"] = False
                st.session_state["active_nav"] = "AI Model Copilot"
                st.rerun()


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


def _render_dom_floating_pill(is_open: bool):
    """Direct DOM anchor helper to guarantee bottom-right fixed positioning across all browsers."""
    script = """
    <script>
    (function() {
        try {
            const markers = document.querySelectorAll('.copilot-marker-pin');
            markers.forEach(m => {
                const elem = m.closest('[data-testid="stElementContainer"]');
                if (elem && elem.nextElementSibling) {
                    const btnBox = elem.nextElementSibling;
                    btnBox.style.setProperty('position', 'fixed', 'important');
                    btnBox.style.setProperty('bottom', '24px', 'important');
                    btnBox.style.setProperty('right', '28px', 'important');
                    btnBox.style.setProperty('z-index', '9999999', 'important');
                    btnBox.style.setProperty('width', 'auto', 'important');
                }
            });
        } catch(e) {}
    })();
    </script>
    """
    try:
        st.html(script, unsafe_allow_javascript=True)
    except Exception:
        pass


def _render_speech_synthesis(text: str):
    """Injects Web Speech API Text-to-Speech into browser to speak text aloud."""
    clean_json_text = json.dumps(text)
    components.html(
        f"""
        <script>
        (function() {{
            try {{
                const targetWindow = window.parent || window;
                if ('speechSynthesis' in targetWindow) {{
                    targetWindow.speechSynthesis.cancel();
                    const msg = new targetWindow.SpeechSynthesisUtterance({clean_json_text});
                    msg.rate = 1.0;
                    msg.pitch = 1.0;
                    msg.lang = 'en-US';

                    const voices = targetWindow.speechSynthesis.getVoices();
                    const preferredVoice = voices.find(v => v.lang === 'en-US' && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha')));
                    if (preferredVoice) {{
                        msg.voice = preferredVoice;
                    }}

                    targetWindow.speechSynthesis.speak(msg);
                }}
            }} catch(e) {{
                console.warn("Speech synthesis error: ", e);
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
                const targetWindow = window.parent || window;
                const SpeechRecognition = targetWindow.SpeechRecognition || targetWindow.webkitSpeechRecognition || window.SpeechRecognition || window.webkitSpeechRecognition;

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
                        const parentInputs = targetWindow.document.querySelectorAll('input[type="text"]');
                        for (let inp of parentInputs) {
                            if (inp.placeholder && (inp.placeholder.includes('talk') || inp.placeholder.includes('model') || inp.placeholder.includes('Ask') || inp.placeholder.includes('question') || inp.placeholder.includes('Which'))) {
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
