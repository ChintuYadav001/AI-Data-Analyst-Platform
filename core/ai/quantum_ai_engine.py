"""
core.ai.quantum_ai_engine
=========================
Main Reasoning & Orchestration Engine for Quantum AI Assistant.
Unites:
- Project Knowledge Base (architecture, technologies, workflows)
- 18 Controlled Backend Calculation Tools (DuckDB, Pandas, Scikit-learn)
- Simple Language Mode (intuitive analogies, plain English)
- Conversation Memory & Multi-turn Context
- Active Page & Dataset Context Awareness
- Structured Answer-Evidence-Explanation-Recommendation formatting
- Natural, synchronized Text-to-Speech voice summaries
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

import core.ai.analytical_tools as tools
import core.ai.project_knowledge as project_knowledge
from core.ai.llm_client import LLMClient
from core.ml.model_copilot import MODEL_ENCYCLOPEDIA, ModelCopilotEngine
from core.ml.experiment_tracker import get_active_champion


class QuantumAIEngine:
    """Enterprise AI Data Analyst Assistant with multi-turn reasoning and controlled calculation tools."""

    def __init__(self):
        self.copilot_engine = ModelCopilotEngine()

    def process_query(
        self,
        query: str,
        df: Optional[pd.DataFrame] = None,
        current_page: str = "Dashboard",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        champ_info: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main query handler. Classifies intent, selects and executes controlled tools,
        and returns a structured response with Answer, Evidence, Explanation, Recommendation,
        and a speech-optimized voice summary.
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()
        conv_hist = conversation_history or []

        # 0. Check for Follow-Up Questions (Memory & Context)
        is_follow_up = any(q_lower.startswith(w) for w in ["why", "how", "explain more", "what about", "and then", "tell me more"])
        last_assistant_msg = None
        if is_follow_up and conv_hist:
            for msg in reversed(conv_hist):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg
                    break

        # Check for active page context first when query is contextual
        if current_page and current_page != "General":
            page_resp = self._handle_page_context(q_clean, current_page, df, champ_info)
            if page_resp:
                return page_resp

        # =====================================================================
        # INTENT 1: Project Knowledge & Architecture ("What does this project do?")
        # =====================================================================
        proj_item = project_knowledge.search_project_knowledge(q_lower)
        if proj_item:
            md_text, voice_text = project_knowledge.format_project_response(proj_item)
            return {
                "intent": "project_knowledge",
                "tool_used": "get_project_information()",
                "category": proj_item.get("category", project_knowledge.CATEGORY_PROJECT_INFO),
                "markdown": md_text,
                "voice_text": voice_text,
                "data": None,
                "chart_spec": None,
                "actions": ["Explain Workflow", "Show Tech Stack", "Explore My Data"],
            }

        # Check for general project queries
        if any(w in q_lower for w in ["what does this project do", "about quantum analyst", "who are you", "what are you"]):
            proj_res = tools.get_project_information(q_lower)
            return {
                "intent": "project_knowledge",
                "tool_used": "get_project_information()",
                "category": project_knowledge.CATEGORY_PROJECT_INFO,
                "markdown": (
                    f"### 💡 About Quantum Analyst\n\n"
                    f"{proj_res['answer']}\n\n"
                    f"**Core Principle:** *{project_knowledge.PROJECT_OVERVIEW['core_principle']}*\n\n"
                    f"**Complete Workflow:** Ingestion → Profiling → Quality Audit → Cleaning → EDA → "
                    f"Feature Engineering → SQL Analytics → Machine Learning → Forecasting → Reports."
                ),
                "voice_text": proj_res["voice_summary"],
                "data": None,
                "actions": ["Explain Architecture", "What is DuckDB?", "How AI Works"],
            }

        # =====================================================================
        # INTENT 2: Model & Concept Encyclopedia ("What is PCA?", "What is Random Forest?")
        # =====================================================================
        model_match = None
        for m_key, m_meta in MODEL_ENCYCLOPEDIA.items():
            if m_key.replace("_", " ") in q_lower or m_meta["name"].lower() in q_lower:
                model_match = (m_key, m_meta)
                break

        if model_match and not any(w in q_lower for w in ["champion", "my data", "this dataset", "why did"]):
            m_key, m_meta = model_match
            ans = self.copilot_engine.answer_model_question(query=q_clean, df=df, champ_info=champ_info)
            return {
                "intent": "concept_encyclopedia",
                "tool_used": "explain_algorithm_concept()",
                "category": project_knowledge.CATEGORY_GENERAL_KNOWLEDGE,
                "markdown": ans["markdown"],
                "voice_text": ans.get("voice_text"),
                "data": None,
                "actions": ["Compare with XGBoost", "When should I use this?", "Check Champion Model"],
            }

        # =====================================================================
        # INTENT 3: No Dataset Loaded Guard
        # =====================================================================
        if df is None or df.empty:
            return {
                "intent": "no_dataset",
                "tool_used": "none",
                "category": project_knowledge.CATEGORY_PROJECT_INFO,
                "markdown": (
                    "### 📂 No Dataset Active\n\n"
                    "I am ready to analyze your data! Please upload a **CSV, Excel, or Parquet file** in "
                    "**📂 Upload Dataset** or load a sample dataset from the workspace.\n\n"
                    "In the meantime, you can ask me anything about **Quantum Analyst, machine learning concepts, or algorithms!**"
                ),
                "voice_text": "Please upload a CSV or Excel dataset to begin analysis. You can also ask me about machine learning concepts or project architecture.",
                "data": None,
                "actions": ["What is EDA?", "What is PCA?", "Explain Quantum Analyst"],
            }

        # =====================================================================
        # INTENT 4: Data Quality & Cleaning ("Find the biggest problem in my data", "What to clean?")
        # =====================================================================
        if any(w in q_lower for w in ["quality", "problem", "defect", "missing", "dirty", "clean", "error", "duplicate", "null"]):
            if any(w in q_lower for w in ["clean", "fix", "heal", "action"]):
                res = tools.get_cleaning_summary(df)
            else:
                res = tools.get_data_quality(df)

            return self._build_tool_response(
                intent="data_quality",
                tool_name=res["tool"],
                res=res,
                actions=["🧹 Open Data Cleaning", "Run Full Profiling", "Proceed to EDA"],
            )

        # =====================================================================
        # INTENT 5: Dataset Summary & Profiling ("Summarize my dataset", "How many rows?")
        # =====================================================================
        if any(w in q_lower for w in ["summarize", "overview", "how many rows", "columns", "profile", "shape", "size", "what is this dataset"]):
            res = tools.profile_dataset(df)
            return self._build_tool_response(
                intent="dataset_profile",
                tool_name="profile_dataset",
                res=res,
                actions=["🛡️ Check Data Quality", "🔬 Run EDA", "🤖 ML Readiness"],
            )

        # =====================================================================
        # INTENT 6: Exploratory Data Analysis & Anomalies ("What are the main insights?", "Outliers")
        # =====================================================================
        if any(w in q_lower for w in ["insight", "pattern", "eda", "unusual", "outlier", "anomaly", "correlation", "distribution", "relationship"]):
            res = tools.run_eda(df, target_col=target_col)
            return self._build_tool_response(
                intent="eda",
                tool_name="run_eda",
                res=res,
                actions=["🔬 Open EDA Engine", "Detect Outliers", "🤖 Train Models"],
            )

        # =====================================================================
        # INTENT 7: Machine Learning & Best Model ("Which model is best?", "Why did this model win?", "Train models")
        # =====================================================================
        if any(w in q_lower for w in ["model", "champion", "best model", "train", "accuracy", "f1", "predictive", "algorithm", "target", "predict", "task", "benchmark"]):
            # Target discovery check
            if any(w in q_lower for w in ["which column should be the target", "target column", "recommend target", "which target"]):
                res = tools.detect_target(df)
                return self._build_tool_response("detect_target", "detect_target", res, ["Train Candidate Models", "Audit ML Readiness"])

            # Explicit training request
            if any(w in q_lower for w in ["train", "fit model", "run benchmark", "build model", "automl"]):
                target = target_col
                if not target:
                    t_res = tools.detect_target(df)
                    target = t_res.get("raw", {}).get("top_recommendation") or (df.columns[-1] if not df.empty else None)
                if target:
                    res = tools.train_model(df, target_col=target, compute_profile="Fast")
                    return self._build_tool_response("train_model", "train_model", res, ["Inspect Diagnostics", "Explain Features", "Simulate Prediction"])

            champ = champ_info or get_active_champion()
            if champ and any(w in q_lower for w in ["champion", "why", "win", "best", "winner", "current"]):
                eval_res = tools.evaluate_model(champ.get("experiment_id"), df=df, target_col=target_col)
                expl_res = tools.explain_model(champ.get("experiment_id"), df=df, target_col=target_col)

                metrics = champ.get("metrics", {})
                metric_highlight = ", ".join([f"**{k}**: {v:.3f}" if isinstance(v, float) else f"**{k}**: {v}" for k, v in list(metrics.items())[:3]])

                answer = f"The production champion is **{champ.get('model_name', 'Trained Model')}**, which achieved top validation performance ({metric_highlight})."
                evidence = {
                    "Champion Model": champ.get("model_name"),
                    "Target Variable": champ.get("target_col", target_col or "Target"),
                    "Task Type": champ.get("task_type", "Regression").replace("_", " ").title(),
                    "Validation Score": metric_highlight,
                }
                expl_data = expl_res.get("data")
                top_features_list = []
                if isinstance(expl_data, pd.DataFrame) and not expl_data.empty and "feature" in expl_data.columns:
                    top_features_list = [str(f) for f in expl_data["feature"].head(3).tolist()]
                top_feats_str = ", ".join(top_features_list) if top_features_list else "balanced attributes"

                explanation = (
                    f"It outperformed other candidates because its architecture captured feature interactions without overfitting. "
                    f"Top predictive drivers include: {top_feats_str}."
                )
                recommendation = "Simulate counterfactual predictions in 🤖 ML Cockpit → Scenario Analysis."

                return {
                    "intent": "champion_model",
                    "tool_used": "evaluate_model() + explain_model()",
                    "category": project_knowledge.CATEGORY_ANALYTICAL_RESULT,
                    "markdown": self._format_structured_output(answer, evidence, explanation, recommendation),
                    "voice_text": f"Your champion model is {champ.get('model_name')}. It won with the strongest validation score and lowest error rate on this dataset.",
                    "data": expl_data if isinstance(expl_data, pd.DataFrame) else None,
                    "actions": ["Explain Features", "Simulate Prediction", "View Leaderboard"],
                }

            # If no champion model is trained yet and user asks about best model
            if not champ and any(w in q_lower for w in ["best", "which model", "winner", "champion", "recommend"]):
                ml_prob = tools.detect_ml_problem(df, target_col=target_col)
                task_name = ml_prob.get("raw", {}).get("task_name", "Classification")
                rec_algo = "Random Forest or XGBoost" if "class" in task_name.lower() else "Random Forest Regressor or Ridge"
                target_cand = target_col or (df.columns[-1] if not df.empty else "Target")
                return {
                    "intent": "recommend_model",
                    "tool_used": "detect_ml_problem()",
                    "category": project_knowledge.CATEGORY_ANALYTICAL_RESULT,
                    "markdown": (
                        f"### 🤖 Model Recommendation\n\n"
                        f"Based on your active dataset ({len(df):,} rows × {len(df.columns)} columns) predicting `{target_cand}`, "
                        f"the recommended task formulation is **{task_name}**.\n\n"
                        f"**Top Recommended Algorithms:**\n"
                        f"- **{rec_algo}**: Best for non-linear interactions with minimal hyperparameter sensitivity.\n"
                        f"- **LightGBM / Gradient Boosting**: Fast, handles non-linearities and missing values natively.\n\n"
                        f"No models have been trained in this session yet. Click **Train Candidate Models** below to benchmark 10+ models and select your champion!"
                    ),
                    "voice_text": f"For your {task_name} dataset, I recommend benchmarking Random Forest and Gradient Boosting. Click train candidate models to find your champion.",
                    "actions": ["Train Candidate Models", "Detect Best Target", "What is Random Forest?"],
                }

            # ML problem detection & target check fallback
            if any(w in q_lower for w in ["target", "problem", "which task"]):
                res = tools.detect_target(df)
                return self._build_tool_response("detect_target", "detect_target", res, ["Train Candidate Models", "Audit ML Readiness"])

            # Default to ML problem analysis
            res = tools.detect_ml_problem(df, target_col=target_col)
            return self._build_tool_response("ml_problem", "detect_ml_problem", res, ["🤖 Open ML Cockpit", "Benchmark 10+ Models"])

        # =====================================================================
        # INTENT 8: Time-Series Forecasting ("Forecast sales", "Future trajectory")
        # =====================================================================
        if any(w in q_lower for w in ["forecast", "future", "trajectory", "next month", "next year", "projection", "will sales"]):
            # Auto-detect date and numerical target
            date_cols = df.select_dtypes(include=["datetime", "object"]).columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            date_col = next((c for c in date_cols if any(k in c.lower() for k in ["date", "time", "day", "month", "year"])), date_cols[0] if date_cols else None)
            target = target_col or (num_cols[0] if num_cols else None)

            if date_col and target:
                res = tools.forecast(df, date_col=date_col, target_col=target, horizon=14)
                return self._build_tool_response("forecasting", "forecast", res, ["📈 Open Forecasting", "Change Horizon", "Download Report"])
            else:
                return {
                    "intent": "forecasting",
                    "tool_used": "none",
                    "category": project_knowledge.CATEGORY_PROJECT_INFO,
                    "markdown": (
                        "### 📈 Forecasting Configuration\n\n"
                        "To forecast future trajectory, I need at least one **Date/Time column** and one **Numerical Target column**.\n\n"
                        f"Detected columns: Dates (`{date_cols[:3]}`), Numbers (`{num_cols[:3]}`). Please specify which column to forecast."
                    ),
                    "voice_text": "To forecast future numbers, please specify which date column and numerical target you would like to predict.",
                    "actions": ["Select Date Column", "Open ML Cockpit"],
                }

        # =====================================================================
        # INTENT 9: Natural Language SQL Query
        # =====================================================================
        sql_query = self._infer_sql_query(q_lower, df)
        if sql_query:
            res = tools.run_sql(sql_query, df)
            return self._build_tool_response(
                intent="sql_query",
                tool_name="run_sql",
                res=res,
                actions=["Show Data Table", "Run Another Query", "Export CSV"],
            )

        # =====================================================================
        # INTENT 10: Dashboard & KPIs ("Summarize this dashboard", "Important KPIs")
        # =====================================================================
        if any(w in q_lower for w in ["dashboard", "kpi", "metrics", "executive summary"]):
            res = tools.get_dashboard_data(df)
            return self._build_tool_response("dashboard_kpis", "get_dashboard_data", res, ["📈 Open Dashboard Builder", "Generate PDF Report"])

        # =====================================================================
        # INTENT 11: Page-Context Awareness
        # =====================================================================
        page_resp = self._handle_page_context(q_clean, current_page, df, champ_info)
        if page_resp:
            return page_resp

        # =====================================================================
        # DEFAULT / GENERAL ANALYTICAL RESPONSE (Grounded Calculation)
        # =====================================================================
        # If external LLM is configured, pass token-compact analytical summary
        prof_res = tools.profile_dataset(df)
        llm_resp = self._try_llm_explanation(q_clean, prof_res, current_page)
        if llm_resp:
            return llm_resp

        # Fallback to verified profile summary
        return {
            "intent": "general_guidance",
            "tool_used": "profile_dataset()",
            "category": project_knowledge.CATEGORY_ANALYTICAL_RESULT,
            "markdown": (
                f"### 🤖 Quantum AI Analyst\n\n"
                f"I evaluated your inquiry regarding **'{q_clean}'** in the context of **{current_page}**.\n\n"
                f"**Active Data Summary:**\n"
                f"- **Total Observations:** {len(df):,} records across {len(df.columns)} features.\n"
                f"- **Data Completeness:** {100 - prof_res['raw_metrics'].get('missing_pct', 0):.1f}%\n\n"
                f"**Suggested Next Steps:**\n"
                f"• Ask me specific questions like *'Find the biggest problem in my data'*, *'Which model is best?'*, or *'Show top 5 categories by sales'*.\n"
                f"• Or click one of the suggested prompt chips below!"
            ),
            "voice_text": f"I analyzed your dataset of {len(df):,} records. You can ask me to inspect data quality, find key insights, or forecast future numbers.",
            "data": df.head(5),
            "actions": ["🛡️ Check Quality", "🔬 Run EDA", "🤖 Best Model", "📈 Forecast"],
        }

    # =====================================================================
    # Helper: Formats Answer -> Evidence -> Explanation -> Recommendation
    # =====================================================================
    def _format_structured_output(
        self,
        answer: str,
        evidence: Dict[str, Any],
        explanation: str,
        recommendation: str,
    ) -> str:
        lines = [
            f"### 🎯 Answer",
            answer,
            "",
            "### 📊 Evidence",
        ]
        for k, v in evidence.items():
            lines.append(f"- **{k}:** {v}")

        lines.extend([
            "",
            "### 💡 What It Means (Simple Explanation)",
            explanation,
            "",
            "### 🚀 Recommendation",
            recommendation,
        ])
        return "\n".join(lines)

    def _build_tool_response(
        self,
        intent: str,
        tool_name: str,
        res: Dict[str, Any],
        actions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        md = self._format_structured_output(
            answer=res.get("answer", "Calculated successfully."),
            evidence=res.get("evidence", {}),
            explanation=res.get("explanation", ""),
            recommendation=res.get("recommendation", ""),
        )
        return {
            "intent": intent,
            "tool_used": f"{tool_name}()",
            "category": project_knowledge.CATEGORY_ANALYTICAL_RESULT,
            "markdown": md,
            "voice_text": res.get("voice_summary", res.get("answer", "")),
            "data": res.get("data"),
            "chart_spec": res.get("recommended_chart"),
            "actions": actions or ["Explain More", "Show Data", "Next Analysis"],
        }

    # =====================================================================
    # Helper: Page-Context Specific Dispatches
    # =====================================================================
    def _handle_page_context(
        self,
        query: str,
        current_page: str,
        df: pd.DataFrame,
        champ_info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        q_l = query.lower()
        if "what does this mean" in q_l or "explain this" in q_l or "help me understand" in q_l:
            if current_page == "Data Quality":
                res = tools.get_data_quality(df)
                return self._build_tool_response("page_quality", "get_data_quality", res)
            elif current_page == "Data Cleaning":
                res = tools.get_cleaning_summary(df)
                return self._build_tool_response("page_cleaning", "get_cleaning_summary", res)
            elif current_page == "EDA":
                res = tools.run_eda(df)
                return self._build_tool_response("page_eda", "run_eda", res)
            elif current_page == "Machine Learning":
                res = tools.evaluate_model()
                return self._build_tool_response("page_ml", "evaluate_model", res)
            elif current_page == "Future Forecasting":
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                date_cols = df.select_dtypes(include=["datetime", "object"]).columns.tolist()
                if num_cols and date_cols:
                    res = tools.forecast(df, date_col=date_cols[0], target_col=num_cols[0])
                    return self._build_tool_response("page_forecast", "forecast", res)
        return None

    # =====================================================================
    # Helper: Safe NL-to-SQL Synthesis
    # =====================================================================
    def _infer_sql_query(self, query_text: str, df: pd.DataFrame, table_name: str = "dataset") -> Optional[str]:
        cols = {str(c).lower(): str(c) for c in df.columns}
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "id" not in str(c).lower()]
        cat_cols = [c for c in df.columns if df[c].nunique() <= 50 and not pd.api.types.is_numeric_dtype(df[c])]

        # Match "top N <category> by <metric>"
        top_match = re.search(r"top\s+(\d+)\s+([a-zA-Z_]+)(?:\s+by\s+([a-zA-Z_]+))?", query_text)
        if top_match:
            limit_n = top_match.group(1)
            target_ent = top_match.group(2).lower()
            metric_ent = top_match.group(3).lower() if top_match.group(3) else None

            group_col = next((cols[k] for k in cols if target_ent in k or k in target_ent), cat_cols[0] if cat_cols else None)
            metric_col = next((cols[k] for k in cols if metric_ent in k or k in metric_ent), num_cols[0] if num_cols else None) if metric_ent else (num_cols[0] if num_cols else None)

            if group_col and metric_col:
                return f'SELECT "{group_col}", ROUND(SUM("{metric_col}"), 2) AS total_{metric_col} FROM "{table_name}" GROUP BY "{group_col}" ORDER BY total_{metric_col} DESC LIMIT {limit_n}'

        # Match "highest / best ... by ..."
        if any(w in query_text for w in ["highest", "best", "most", "lowest", "worst"]):
            order = "DESC" if any(w in query_text for w in ["highest", "best", "most"]) else "ASC"
            cat_found = next((cat for cat in cat_cols if cat.lower() in query_text), cat_cols[0] if cat_cols else None)
            num_found = next((num for num in num_cols if num.lower() in query_text), num_cols[0] if num_cols else None)

            if cat_found and num_found:
                return f'SELECT "{cat_found}", ROUND(AVG("{num_found}"), 2) AS avg_{num_found} FROM "{table_name}" GROUP BY "{cat_found}" ORDER BY avg_{num_found} {order} LIMIT 5'

        return None

    # =====================================================================
    # Helper: Optional LLM Synthesis
    # =====================================================================
    def _try_llm_explanation(self, user_query: str, profile_res: Dict[str, Any], current_page: str) -> Optional[Dict[str, Any]]:
        """Invokes configured LLM with token-compact context only when provider is active."""
        provider = LLMClient.get_configured_provider()
        if provider == "offline":
            return None

        prompt = (
            f"User Question: '{user_query}'\n"
            f"Active Page: {current_page}\n"
            f"Dataset Context: {profile_res['raw_metrics']}\n\n"
            "Format your response as:\n"
            "### 🎯 Answer\n<direct concise answer>\n\n"
            "### 📊 Evidence\n<verified numbers from context>\n\n"
            "### 💡 What It Means (Simple Explanation)\n<intuitive analogy and plain words>\n\n"
            "### 🚀 Recommendation\n<next logical step in Quantum Analyst>\n"
        )
        llm_call = LLMClient.call_llm(prompt, max_tokens=600)
        if llm_call.get("success") and llm_call.get("text"):
            text = llm_call["text"]
            voice_summary = text.split("###")[1].replace("🎯 Answer", "").strip() if "###" in text else text[:150]
            return {
                "intent": "llm_grounded",
                "tool_used": f"llm_reasoning({llm_call.get('model')})",
                "category": project_knowledge.CATEGORY_ANALYTICAL_RESULT,
                "markdown": text,
                "voice_text": voice_summary,
                "data": None,
                "actions": ["Explain More", "Check Quality", "Explore Data"],
            }
        return None
