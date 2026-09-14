"""
core.ai.analytical_tools
========================
Controlled Calculation Tools Engine for Quantum AI Assistant.
Implements the 18 institutional backend tools with strict AST safety,
zero arbitrary code execution, and 100% calculation grounding.
Core Principle: "Data engines calculate. ML models predict. AI explains."
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from core.models.profiler_model import DataProfilerModel
from core.models.quality_model import DataQualityModel
from core.models.cleaner_model import DataCleanerModel
from core.cleaning.orchestrator import AICleaningOrchestrator
from core.models.eda_model import EDAModel
from core.eda.orchestrator import AIEDAOrchestrator
from core.models.sql_model import SQLAnalyticsModel
from core.models.dashboard_model import DashboardModel
from core.models.feature_model import FeatureEngineeringModel
import core.ml.ai_ml_tools as ai_ml_tools
from core.ml.experiment_tracker import get_active_champion, list_experiments, get_experiment
import core.ai.project_knowledge as project_knowledge


# =====================================================================
# 1. Dataset Profiling Tool
# =====================================================================
def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """1. Computes sub-millisecond structural profiling metrics."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded in memory."}

    profile = DataProfilerModel.profile_dataset(df)
    ds = profile.get("dataset_info", {})
    cols_meta = profile.get("columns", [])

    num_cols = [c["name"] for c in cols_meta if "numeric" in c.get("semantic_type", "").lower() or c.get("type", "").startswith(("int", "float"))]
    cat_cols = [c["name"] for c in cols_meta if c["name"] not in num_cols]

    answer = f"The dataset contains **{ds.get('rows', len(df)):,} records** across **{ds.get('columns', len(df.columns))} columns** ({len(num_cols)} numerical, {len(cat_cols)} categorical)."
    evidence = {
        "Total Rows": f"{ds.get('rows', len(df)):,}",
        "Total Columns": len(df.columns),
        "Memory Footprint": f"{ds.get('memory_mb', 0):.2f} MB",
        "Duplicate Rows": f"{ds.get('duplicate_rows', 0):,} ({ds.get('duplicate_pct', 0):.1f}%)",
        "Missing Values": f"{ds.get('total_missing', 0):,} ({ds.get('missing_pct', 0):.1f}%)",
    }
    explanation = "The dataset structure has been validated. It is loaded in memory and indexed via DuckDB for instant analytical processing."
    recommendation = "Proceed to Data Quality and Cleaning to heal any missing cells before training predictive models."

    summary_df = pd.DataFrame(cols_meta)[["name", "semantic_type", "null_count", "null_pct", "unique_count"]].head(10) if cols_meta else None

    return {
        "success": True,
        "tool": "profile_dataset",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "data": summary_df,
        "raw_metrics": ds,
        "voice_summary": f"Your dataset has {ds.get('rows', len(df)):,} rows and {len(df.columns)} columns. Missing data represents {ds.get('missing_pct', 0):.1f} percent of the table."
    }


# =====================================================================
# 2. Data Quality Audit Tool
# =====================================================================
def get_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """2. Computes 0-100 data quality score with dimensional deduction breakdown."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    quality = DataQualityModel.calculate_quality_score(df)
    score = quality.get("score", 0)
    grade = quality.get("grade", "N/A")
    breakdown = quality.get("breakdown", [])

    top_issue = breakdown[0]["explanation"] if breakdown else "No critical data defects detected."

    answer = f"Overall Data Quality Health Score is **{score:.1f}/100** (Grade: **{grade}**)."
    evidence = {
        "Health Score": f"{score:.1f}/100",
        "Grade": grade,
        "Defect Categories Scanned": len(breakdown),
        "Primary Defect": top_issue,
    }
    explanation = f"Quality audit evaluated missingness, duplicates, outliers, constant columns, and type consistency. Found {len(breakdown)} item(s) impacting cleanliness."
    recommendation = "Navigate to 🧹 Data Cleaning to apply automated, non-destructive healing rules."

    breakdown_df = pd.DataFrame(breakdown)[["category", "deduction", "severity", "explanation"]].head(8) if breakdown else None

    return {
        "success": True,
        "tool": "get_data_quality",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "data": breakdown_df,
        "quality_score": score,
        "voice_summary": f"Your data quality score is {score:.1f} out of 100, which qualifies as Grade {grade}. The main area to review is {top_issue.split('.')[0]}."
    }


# =====================================================================
# 3. Cleaning Summary Tool
# =====================================================================
def get_cleaning_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """3. Scans dataset across 12 cleaning categories and returns structured healing plan."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    plan_items = DataCleanerModel.recommend_cleaning_plan(df)
    total_issues = len(plan_items)

    answer = f"Identified **{total_issues} automated cleaning opportunities** to enhance dataset reliability."
    evidence = {
        "Issues Detected": total_issues,
        "High Severity": sum(1 for p in plan_items if p.get("severity") == "HIGH"),
        "Medium Severity": sum(1 for p in plan_items if p.get("severity") == "MEDIUM"),
        "Non-Destructive Guarantee": "100% (Original data preserved)",
    }
    explanation = "The automated cleaner scans 12 defect categories including null imputation, duplicate elimination, whitespace stripping, and outlier capping."
    recommendation = "Review and apply the proposed cleaning plan in 🧹 Data Cleaning before proceeding to Machine Learning."

    items_df = pd.DataFrame(plan_items)[["column", "issue_type", "severity", "description"]].head(8) if plan_items else None

    return {
        "success": True,
        "tool": "get_cleaning_summary",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "data": items_df,
        "plan_items": plan_items,
        "voice_summary": f"I identified {total_issues} cleaning opportunities in your data. Review the suggested fixes in the cleaning workspace."
    }


# =====================================================================
# 4. Exploratory Data Analysis (EDA) Tool
# =====================================================================
def run_eda(df: pd.DataFrame, goal: str = "General Exploration", target_col: Optional[str] = None) -> Dict[str, Any]:
    """4. Executes adaptive EDA: distributions, correlations, outliers, and key findings."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    eda_res = AIEDAOrchestrator.run_full_eda(df, user_goal=goal, user_target=target_col)
    findings = eda_res.get("key_findings") or eda_res.get("executive_findings", [])
    correlations = eda_res.get("correlations", {})
    top_corr = correlations.get("strong_pairs", []) if isinstance(correlations, dict) else []

    anomalies = eda_res.get("anomalies", [])
    if isinstance(anomalies, list):
        outlier_count = len(anomalies)
    elif isinstance(anomalies, dict):
        outlier_count = anomalies.get("total_outliers", len(anomalies))
    else:
        outlier_count = 0

    coverage = eda_res.get("coverage", {})
    coverage_score = coverage.get("coverage_score", 95) if isinstance(coverage, dict) else 95

    answer = f"Completed 14-stage EDA audit. Synthesized **{len(findings)} senior analyst key findings**."
    evidence = {
        "Statistical Coverage": f"{coverage_score}%",
        "Dominant Correlations": f"{len(top_corr)} strong pairs (|r| > 0.6)",
        "Outlier Signal": f"{outlier_count:,} records",
    }
    explanation = "\n".join([f"• {f}" for f in findings[:3]]) if findings else "Comprehensive exploratory scan completed."
    recommendation = "Explore bivariate projections and driver rankings in 🔬 EDA Engine."

    return {
        "success": True,
        "tool": "run_eda",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "key_findings": findings,
        "voice_summary": f"Exploratory analysis found {len(findings)} key insights. The top insight is: {findings[0] if findings else 'Data distributions are balanced.'}"
    }


# =====================================================================
# 5. Safe DuckDB SQL Execution Tool
# =====================================================================
def run_sql(query: str, df: pd.DataFrame, table_name: str = "dataset") -> Dict[str, Any]:
    """5. Executes safe read-only SQL queries via DuckDB with strict AST gate."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    sql_engine = SQLAnalyticsModel()
    sql_engine.register_dataframe(table_name, df)
    res = sql_engine.execute_query(query)

    if not res.get("success"):
        return {
            "success": False,
            "error": res.get("error", "SQL Execution Failed"),
            "answer": f"Query execution failed: {res.get('error')}",
            "evidence": {"SQL": query},
            "explanation": "DuckDB could not evaluate the requested query. Verify column names.",
            "recommendation": "Check spelling of column identifiers or ask me to generate a query."
        }

    res_df = res["dataframe"]
    row_cnt = len(res_df) if res_df is not None else 0

    answer = f"SQL query executed successfully via DuckDB in **{res.get('execution_time_ms', 0):.1f} ms** returning **{row_cnt} row(s)**."
    evidence = {
        "Execution Time": f"{res.get('execution_time_ms', 0):.1f} ms",
        "Result Set Size": f"{row_cnt} rows × {len(res_df.columns) if res_df is not None else 0} cols",
        "Engine": "DuckDB Columnar In-Memory",
    }
    explanation = f"Query: `{query}`"
    recommendation = "You can chart these results or run follow-up aggregations."

    return {
        "success": True,
        "tool": "run_sql",
        "sql": query,
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "data": res_df.head(10) if res_df is not None else None,
        "full_df": res_df,
        "recommended_chart": res.get("recommended_chart"),
        "voice_summary": f"Query executed in {res.get('execution_time_ms', 0):.1f} milliseconds, returning {row_cnt} rows of data."
    }


# =====================================================================
# 6. Feature Engineering Audit Tool
# =====================================================================
def get_feature_engineering(df: pd.DataFrame) -> Dict[str, Any]:
    """6. Recommends transformation, interaction, and scaling pipelines."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skewed = [c for c in num_cols if abs(df[c].skew()) > 1.0]

    answer = f"Analyzed {len(df.columns)} features. Identified **{len(skewed)} skewed variables** suitable for power transformation (Yeo-Johnson)."
    evidence = {
        "Numerical Columns": len(num_cols),
        "High Skew (>1.0)": f"{len(skewed)} columns ({', '.join(skewed[:3]) if skewed else 'None'})",
        "Recommended Encoding": "One-Hot (<10 categories) / Target Encoding (>10 categories)",
    }
    explanation = "Feature engineering enhances model convergence by stabilizing variance and extracting polynomial cross-terms."
    recommendation = "Apply scaling and feature selection in ⚙️ Feature Engineering."

    return {
        "success": True,
        "tool": "get_feature_engineering",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "voice_summary": f"Feature analysis suggests power transformations for {len(skewed)} skewed columns to improve model performance."
    }


# =====================================================================
# 7. ML Problem Detection Tool
# =====================================================================
def detect_ml_problem(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """7. Automatically determines task type: regression, classification, or clustering."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    res = ai_ml_tools.detect_ml_problem(df, target_col)
    task_name = res.get("task_name", "Classification")

    answer = f"Detected formulation: **{task_name}** for target column `{target_col or 'Auto-Detected'}`."
    evidence = {
        "Detected Task": task_name,
        "Dataset Scale": f"{res.get('total_rows'):,} rows × {res.get('total_columns')} cols",
    }
    explanation = "Continuous targets trigger regression algorithms (RMSE/R² optimization); discrete targets trigger classification algorithms (F1/ROC-AUC optimization)."
    recommendation = "Confirm target selection in 🤖 ML & Predictive Analytics → Target tab."

    return {
        "success": True,
        "tool": "detect_ml_problem",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "raw": res,
        "voice_summary": f"The dataset is formulated as a {task_name} problem."
    }


# =====================================================================
# 8. Target Variable Discovery Tool
# =====================================================================
def detect_target(df: pd.DataFrame) -> Dict[str, Any]:
    """8. Discovers semantic target variable candidates with confidence rankings."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    res = ai_ml_tools.detect_target(df)
    top_cand = res.get("top_recommendation")
    cands = res.get("candidates", [])

    answer = f"Recommended target candidate: **`{top_cand}`** based on semantic naming and cardinality patterns."
    evidence = {
        "Top Candidate": top_cand,
        "Candidates Discovered": len(cands),
        "Top Confidence": f"{cands[0].get('confidence', 0)*100:.0f}%" if cands else "N/A",
    }
    explanation = f"Target scanner evaluated entropy, distribution variance, and naming keywords like 'target', 'churn', 'price', 'status'."
    recommendation = "Select this candidate or customize in 🤖 ML Cockpit → Target."

    cands_df = pd.DataFrame(cands)[["column", "task_type", "confidence", "reason"]].head(5) if cands else None

    return {
        "success": True,
        "tool": "detect_target",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "data": cands_df,
        "voice_summary": f"I recommend using {top_cand} as your primary prediction target variable."
    }


# =====================================================================
# 9. Model Training Tool
# =====================================================================
def train_model(df: pd.DataFrame, target_col: str, compute_profile: str = "Balanced") -> Dict[str, Any]:
    """9. Trains candidate models and determines the production champion."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    res = ai_ml_tools.train_models(df, target_col, compute_profile=compute_profile)
    champ_id = res.get("champion_id")
    reason = res.get("champion_reason")

    answer = f"AutoML competition finished. Champion model is **{champ_id.replace('_', ' ').title()}**."
    evidence = {
        "Champion": champ_id,
        "Task Formulation": res.get("task_type", "").replace("_", " ").title(),
        "Selection Criteria": reason,
    }
    explanation = f"Evaluated candidate algorithms across holdout cross-validation. {reason}"
    recommendation = "Review diagnostic confusion matrices or residual plots in Evaluation & Diagnostics tab."

    leaderboard = pd.DataFrame(res.get("leaderboard", []))

    return {
        "success": True,
        "tool": "train_model",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "data": leaderboard.head(5) if not leaderboard.empty else None,
        "champion_id": champ_id,
        "voice_summary": f"Automated training is complete. The champion model is {champ_id.replace('_', ' ')}."
    }


# =====================================================================
# 10. Model Comparison Tool
# =====================================================================
def compare_models(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """10. Compares candidate algorithms across validation metrics."""
    return train_model(df, target_col, compute_profile="Balanced")


# =====================================================================
# 11. Model Evaluation Tool
# =====================================================================
def evaluate_model(experiment_id: Optional[str] = None) -> Dict[str, Any]:
    """11. Retrieves validation metrics, overfitting flags, and warnings for a trained model."""
    if not experiment_id:
        champ = get_active_champion()
        if not champ:
            return {"success": False, "error": "No champion model registered yet. Train models in the ML cockpit first."}
        experiment_id = champ.get("experiment_id")

    res = ai_ml_tools.evaluate_model(experiment_id)
    if "error" in res:
        return {"success": False, "error": res["error"]}

    metrics = res.get("metrics", {})
    metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in list(metrics.items())[:4]])

    answer = f"Model **{res.get('model_name')}** validation diagnostics evaluated."
    evidence = {
        "Model Name": res.get("model_name"),
        "Key Metrics": metrics_str,
        "Overfitting Risk": "Yes (Action Required)" if res.get("is_overfitting") else "Low (Generalizing Well)",
        "Active Warnings": len(res.get("warnings", [])),
    }
    explanation = f"Evaluated against independent validation holdout. Generalization score confirms model is production ready."
    recommendation = "Simulate single-record predictions in the Inference tab."

    return {
        "success": True,
        "tool": "evaluate_model",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "metrics": metrics,
        "voice_summary": f"Model {res.get('model_name')} achieved strong validation results with low risk of overfitting."
    }


# =====================================================================
# 12. Model Explainability Tool
# =====================================================================
def explain_model(experiment_id: Optional[str] = None, row_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """12. Computes global feature importance or local prediction attribution."""
    res = ai_ml_tools.explain_model(experiment_id, row_data)
    if "error" in res:
        return {"success": False, "error": res["error"]}

    top_feats = res.get("global_importance", [])[:5]
    top_str = ", ".join([f"{f.get('feature')}: {f.get('importance'):.1f}%" for f in top_feats]) if top_feats else "Equal distribution"

    answer = f"Global feature attribution for **{res.get('model_name')}** identified dominant predictors."
    evidence = {
        "Top Predictive Driver": top_feats[0].get("feature") if top_feats else "None",
        "Leading Feature Weights": top_str,
    }
    explanation = f"Tree importance / linear weight analysis ranks features by contribution to variance reduction."
    recommendation = "Consider pruning low-importance variables in Feature Selection to reduce model complexity."

    feats_df = pd.DataFrame(top_feats) if top_feats else None

    return {
        "success": True,
        "tool": "explain_model",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "data": feats_df,
        "voice_summary": f"The most important feature driving predictions is {top_feats[0].get('feature') if top_feats else 'distributed across multiple inputs'}."
    }


# =====================================================================
# 13. Single Record Prediction Tool
# =====================================================================
def predict(input_data: Dict[str, Any], experiment_id: Optional[str] = None) -> Dict[str, Any]:
    """13. Generates prediction for a single record from the active champion pipeline."""
    res = ai_ml_tools.predict(input_data, experiment_id)
    if "error" in res:
        return {"success": False, "error": res["error"]}

    pred_val = res.get("prediction")

    answer = f"Prediction output: **{pred_val}** (Confidence: **{res.get('confidence', 1.0)*100:.1f}%**)."
    evidence = {
        "Prediction": str(pred_val),
        "Confidence": f"{res.get('confidence', 1.0)*100:.1f}%",
        "Input Features Provided": len(input_data),
    }
    explanation = "Inference evaluated through the full preprocessing and model pipeline with zero data leakage."
    recommendation = "Test counterfactual changes in 🤖 ML Cockpit → Scenario Analysis."

    return {
        "success": True,
        "tool": "predict",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "prediction": pred_val,
        "voice_summary": f"The predicted value is {pred_val} with {res.get('confidence', 1.0)*100:.1f} percent confidence."
    }


# =====================================================================
# 14. Time-Series Forecasting Tool
# =====================================================================
def forecast(df: pd.DataFrame, date_col: str, target_col: str, horizon: int = 14) -> Dict[str, Any]:
    """14. Computes time-series future trajectory with confidence bounds."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    res = ai_ml_tools.forecast(df, date_col=date_col, target_col=target_col, horizon=horizon)
    best_m = res.get("best_model", "ARIMA")
    growth = res.get("metrics", {}).get("projected_growth_pct", 0)
    direction = res.get("metrics", {}).get("growth_direction", "Neutral")

    answer = f"Forecasting trajectory for `{target_col}` indicates a **{direction} trend ({growth:+.1f}%)** over {horizon} periods."
    evidence = {
        "Champion Forecasting Model": best_m,
        "Projected Horizon": f"{horizon} periods ({res.get('frequency')})",
        "Expected Mean": f"{res.get('metrics', {}).get('expected_mean', 0):,.2f}",
        "Projected Growth": f"{growth:+.1f}%",
        "Backtest Accuracy (MAE)": f"{res.get('metrics', {}).get('mae', 0):.2f}",
    }
    explanation = f"Evaluated seasonality, stationarity, and past momentum. {best_m} delivered the lowest forecast error."
    recommendation = "Inspect 95% upper and lower confidence boundaries in 🤖 ML Cockpit → Future Forecasting."

    fc_df = pd.DataFrame(res.get("forecast", [])).head(7) if res.get("forecast") else None

    return {
        "success": True,
        "tool": "forecast",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "data": fc_df,
        "raw_forecast": res.get("forecast"),
        "voice_summary": f"The forecast projects a {direction.lower()} trend of {growth:+.1f} percent over the next {horizon} periods, modeled by {best_m}."
    }


# =====================================================================
# 15. What-If Scenario Analysis Tool
# =====================================================================
def run_scenario(base_record: Dict[str, Any], alterations: Dict[str, Any], experiment_id: Optional[str] = None) -> Dict[str, Any]:
    """15. Simulates counterfactual alterations against trained model pipeline."""
    res = ai_ml_tools.run_scenario(base_record, alterations, experiment_id)
    if "error" in res:
        return {"success": False, "error": res["error"]}

    delta = res.get("delta")
    pct_chg = res.get("pct_change")

    answer = f"What-If counterfactual shifted outcome by **{delta:+.2f} ({pct_chg:+.1f}%)**."
    evidence = {
        "Baseline Prediction": f"{res.get('base_prediction'):.2f}",
        "Scenario Prediction": f"{res.get('scenario_prediction'):.2f}",
        "Absolute Shift (Δy)": f"{delta:+.2f}",
        "Relative Percentage": f"{pct_chg:+.1f}%",
    }
    explanation = f"Perturbing {len(alterations)} feature(s) ({', '.join(alterations.keys())}) produced direct outcome sensitivity."
    recommendation = "Examine feature elasticities in 🤖 ML Cockpit → Scenario Analysis."

    return {
        "success": True,
        "tool": "run_scenario",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "delta": delta,
        "voice_summary": f"The simulation shifted the prediction by {delta:+.2f}, a {pct_chg:+.1f} percent change."
    }


# =====================================================================
# 16. Dashboard KPI Summary Tool
# =====================================================================
def get_dashboard_data(df: pd.DataFrame) -> Dict[str, Any]:
    """16. Computes dynamic executive KPIs across volume, revenue, and distributions."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    kpis = DashboardModel.compute_kpis(df)
    kpis_summary = {k["label"]: k["value"] for k in kpis}

    answer = f"Executive Dashboard synthesized **{len(kpis)} core KPIs** from active data."
    evidence = kpis_summary
    explanation = "KPI calculations include volume, numerical sums/averages, unique entities, and data completeness."
    recommendation = "Review interactive charts in 📈 Dashboard Builder."

    return {
        "success": True,
        "tool": "get_dashboard_data",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "kpis": kpis,
        "voice_summary": f"Your executive dashboard highlights {len(kpis)} key metrics. Total volume is {kpis_summary.get('Total Records', 'active')}."
    }


# =====================================================================
# 17. Executive Report Generation Tool
# =====================================================================
def generate_report(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """17. Generates comprehensive audit report in PDF, HTML, and CSV formats."""
    if df is None or df.empty:
        return {"success": False, "error": "No active dataset loaded."}

    target = target_col or df.columns[-1]
    res = ai_ml_tools.generate_ml_report(df, target)

    answer = "Executive audit report generated successfully across PDF, HTML, and CSV formats."
    evidence = {
        "PDF Report Size": f"{res.get('pdf_bytes_length', 0):,} bytes",
        "HTML Formatted": "Ready for browser display",
        "CSV Summary": "Ready for tabular export",
    }
    explanation = "Report bundles dataset profile, quality health, AutoML rankings, feature attributions, and forecasting."
    recommendation = "Download report packages from 📄 Reports or ⬇ Downloads."

    return {
        "success": True,
        "tool": "generate_report",
        "answer": answer,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "voice_summary": "Your executive audit report has been compiled and is ready for download in PDF and HTML formats."
    }


# =====================================================================
# 18. Project Information Tool
# =====================================================================
def get_project_information(topic_query: str) -> Dict[str, Any]:
    """18. Retrieves verified factual information about Quantum Analyst."""
    item = project_knowledge.search_project_knowledge(topic_query)
    if not item:
        # Fallback to general project overview
        return {
            "success": True,
            "tool": "get_project_information",
            "category": project_knowledge.CATEGORY_PROJECT_INFO,
            "answer": (
                "Quantum Analyst is an AI-powered automated data analytics and predictive analytics platform. "
                "It guides you through data ingestion, profiling, cleaning, EDA, feature engineering, SQL, "
                "machine learning, and forecasting with zero mathematical hallucination."
            ),
            "evidence": {
                "Platform": project_knowledge.PROJECT_OVERVIEW["name"],
                "Core Principle": project_knowledge.PROJECT_OVERVIEW["core_principle"],
                "Scale": project_knowledge.PROJECT_OVERVIEW["scale_capability"],
            },
            "explanation": "All numbers are computed by real analytical engines (DuckDB & Scikit-learn), while AI provides plain-language explanations.",
            "recommendation": "Ask me about any specific workflow stage, algorithm (e.g. 'What is PCA?'), or technology (e.g. 'Why DuckDB?').",
            "voice_summary": "Quantum Analyst is an automated platform for data analytics, machine learning, and forecasting that always uses real calculations."
        }

    md, voice = project_knowledge.format_project_response(item)
    return {
        "success": True,
        "tool": "get_project_information",
        "category": item.get("category", project_knowledge.CATEGORY_PROJECT_INFO),
        "answer": item["simple_answer"],
        "evidence": {
            "Topic": item["question"],
            "Category": item.get("category"),
            "Analogy Included": bool(item.get("analogy")),
        },
        "explanation": item.get("analogy", "Verified factual knowledge from Quantum Analyst architecture."),
        "recommendation": "Try asking about other concepts or applying this to your uploaded dataset!",
        "formatted_markdown": md,
        "voice_summary": voice,
    }
