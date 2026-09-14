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

    if st.button("💾 Save Settings", type="primary"):
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["GEMINI_API_KEY"] = gemini_key
        # Save to .env
        env_file = BASE_DIR / ".env"
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(f"OPENAI_API_KEY={openai_key}\n")
            f.write(f"GEMINI_API_KEY={gemini_key}\n")
            f.write(f"DEFAULT_MODEL={model_choice}\n")
        st.success("Settings saved successfully!")

    st.markdown("---")
    st.markdown("### 🔒 Security & Privacy Guarantees")
    st.markdown("""
    - **No Arbitrary Code Execution:** The AI model cannot execute arbitrary Python or shell code. It calls controlled, sandboxed backend tools.
    - **DuckDB Analytical Guardrail:** All destructive SQL commands (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`) are strictly blocked before execution.
    - **Data Isolation:** Uploaded data is stored locally in `./data/`. Raw datasets are never overwritten.
    """)
