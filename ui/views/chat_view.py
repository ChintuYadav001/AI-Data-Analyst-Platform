import streamlit as st
import pandas as pd
from core.models import AIChatModel, SQLAnalyticsModel

def render_chat_view():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Please upload a dataset or load a sample dataset first.")
        return

    df = st.session_state["df"]
    st.title("💬 AI Data Chat Assistant")
    st.markdown("Ask questions in natural language. The AI uses controlled analytical tools without executing arbitrary code.")

    # Initialize SQL engine & Chat model
    if "sql_engine" not in st.session_state:
        st.session_state["sql_engine"] = SQLAnalyticsModel()
    
    engine = st.session_state["sql_engine"]
    engine.register_dataframe("dataset", df)
    chat_agent = AIChatModel(engine)

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": "Hello! I am your AI Data Analyst. Ask me about trends, top categories, anomalies, or to generate SQL queries and charts."
            }
        ]

    # Quick prompt buttons
    st.markdown("##### ⚡ Quick Prompts")
    q1, q2, q3, q4 = st.columns(4)
    quick_query = None
    with q1:
        if st.button("Top 5 by Revenue"):
            quick_query = "Show top 5 categories by sales"
    with q2:
        if st.button("Best Performing Region"):
            quick_query = "Which region has the highest profit?"
    with q3:
        if st.button("Find Outliers"):
            quick_query = "Find unusual values and outliers"
    with q4:
        if st.button("Generate Chart"):
            quick_query = "Create a chart for monthly revenue"

    # Display chat history
    for i, msg in enumerate(st.session_state["chat_messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sql" in msg:
                st.code(msg["sql"], language="sql")
            if "table" in msg and msg["table"] is not None:
                st.dataframe(msg["table"], width="stretch")
            if "chart" in msg and msg["chart"] is not None:
                st.plotly_chart(msg["chart"], width="stretch", key=f"chat_chart_hist_{i}")

    # Input handling
    user_prompt = st.chat_input("Ask a question about your dataset...") or quick_query

    if user_prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI analyzing and selecting appropriate tool..."):
                res = chat_agent.answer_query(user_prompt, df, table_name="dataset")
                
                tool_badge = f"`Tool Executed: {res.get('tool_used', 'none')}`"
                st.caption(tool_badge)
                st.markdown(res["response"])

                msg_entry = {
                    "role": "assistant",
                    "content": f"{tool_badge}\n\n{res['response']}"
                }

                if "sql" in res:
                    st.code(res["sql"], language="sql")
                    msg_entry["sql"] = res["sql"]

                if "data" in res and res["data"] is not None:
                    st.dataframe(res["data"], width="stretch")
                    msg_entry["table"] = res["data"]

                if "chart_figure" in res and res["chart_figure"] is not None:
                    st.plotly_chart(res["chart_figure"], width="stretch", key=f"chat_live_chart_{len(st.session_state['chat_messages'])}")
                    msg_entry["chart"] = res["chart_figure"]

                st.session_state["chat_messages"].append(msg_entry)
