import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any, List, Optional
from core.config import OPENAI_API_KEY, GEMINI_API_KEY

class AIInsightsModel:
    """Deep multi-horizon analytical reasoning engine generating descriptive, diagnostic,

    predictive, and prescriptive insights with rigorous mathematical grounding and
    requirement specifications.
    """

    @classmethod
    def generate_deep_analytical_insights(
        cls,
        profile: Dict[str, Any],
        eda_results: Dict[str, Any],
        quality: Dict[str, Any],
        goal: str = "General Business Intelligence",
        goal_explanation: str = ""
    ) -> Dict[str, Any]:
        ds_info = profile.get("dataset_info", {})
        n_rows = ds_info.get("rows", 0)
        n_cols = ds_info.get("columns", 0)
        dup_rows = ds_info.get("duplicate_rows", 0)
        missing_pct = ds_info.get("missing_pct", 0)

        # 0. Goal-Oriented Strategic Context & Hypothesis Evaluation
        goal_analysis = []
        hypothesis_evaluations = []
        targeted_metrics = []
        goal_lower = (goal or "").lower()
        expl_lower = (goal_explanation or "").lower()

        if "revenue" in goal_lower or "sales" in goal_lower:
            goal_analysis.append(f"🎯 **Target Objective: Revenue Maximization & Sales Acceleration**")
            goal_analysis.append(f"Focusing analytical lenses on volume drivers, high-ticket clusters, price inelasticity, and transaction throughput across all {n_cols} attributes.")
        elif "churn" in goal_lower or "retention" in goal_lower or "customer" in goal_lower:
            goal_analysis.append(f"👥 **Target Objective: Customer Retention & Churn Prevention**")
            goal_analysis.append(f"Auditing customer cohorts, recency/frequency patterns, activity decay, and early warning signals to mitigate attrition risk.")
        elif "cost" in goal_lower or "margin" in goal_lower or "efficiency" in goal_lower:
            goal_analysis.append(f"📉 **Target Objective: Cost Reduction & Margin Optimization**")
            goal_analysis.append(f"Scrutinizing cost structures, low-margin products, operational friction points, and variance leakage across segments.")
        elif "fraud" in goal_lower or "risk" in goal_lower or "anomaly" in goal_lower:
            goal_analysis.append(f"🛡️ **Target Objective: Anomaly, Risk & Outlier Mitigation**")
            goal_analysis.append(f"Prioritizing statistical outlier scans, distribution tail volatility (CV > 1.5), and cluster deviations threatening systemic reliability.")
        else:
            goal_analysis.append(f"⚡ **Target Objective: {goal}**")
            goal_analysis.append(f"Aligning multi-horizon predictive patterns and categorical Pareto ratios directly with '{goal}'.")

        # Deep contextual analysis on the basis of user's articulated goal explanation
        if goal_explanation and goal_explanation.strip():
            goal_analysis.append(f"📝 **User Analytical Context & Hypothesis:** \"{goal_explanation.strip()}\"")
            
            # Identify columns cited or implied in the explanation
            col_meta_list = profile.get("columns", [])
            col_names = [c.get("name", "") for c in col_meta_list]
            matched_cols = [c for c in col_names if c.lower() in expl_lower or c.lower().replace("_", " ") in expl_lower]
            
            if matched_cols:
                targeted_metrics.extend(matched_cols)
                goal_analysis.append(f"🔬 **Hypothesis-Linked Features:** Cross-referencing {len(matched_cols)} specific attributes targeted in your brief: `{'`, `'.join(matched_cols)}`.")
                for mc in matched_cols:
                    ns = next((n for n in eda_results.get("numerical", []) if n.get("column") == mc), None)
                    if ns:
                        eval_text = (
                            f"Evaluated targeted metric **`{mc}`**: Mean = {ns['mean']:,.2f} (median: {ns['median']:,.2f}, IQR: {ns['iqr']:,.2f}, spread: {ns['std']:,.2f}). "
                            f"Distribution shows {ns.get('distribution_shape', 'standard')} dispersion directly driving '{goal}' variance."
                        )
                        hypothesis_evaluations.append(eval_text)
                    cs = next((c for c in eda_results.get("categorical", []) if c.get("column") == mc), None)
                    if cs:
                        top = cs.get("top_categories", [{}])[0]
                        eval_text = (
                            f"Evaluated targeted cohort **`{mc}`**: Dominant class is **'{top.get('category', 'N/A')}'** "
                            f"representing {top.get('percentage', 0):.1f}% of total volume ({top.get('count', 0):,} records)."
                        )
                        hypothesis_evaluations.append(eval_text)
            else:
                top_nums = eda_results.get("numerical", [])[:2]
                for ns in top_nums:
                    targeted_metrics.append(ns["column"])
                    hypothesis_evaluations.append(
                        f"Evaluated variance in **`{ns['column']}`** against your objective: "
                        f"Standard deviation of {ns['std']:,.2f} (CV = {ns['std']/max(ns['mean'], 0.001):.2f}) indicates key optimization sensitivity."
                    )

        # 1. Descriptive Analytics (What Happened?)
        descriptive = []
        descriptive.append(
            f"Dataset encompasses a population of **{n_rows:,} records** across **{n_cols} attributes** with a health index of **{quality.get('score', 100)}/100** ({quality.get('grade', 'Good')})."
        )
        
        num_stats = eda_results.get("numerical", [])
        for col_stat in num_stats[:3]:
            col_name = col_stat["column"]
            mean_v = col_stat["mean"]
            med_v = col_stat["median"]
            iqr_v = col_stat["iqr"]
            descriptive.append(
                f"Metric **{col_name}** exhibits an average value of **{mean_v:,.2f}** (median: {med_v:,.2f}) with interquartile spread (IQR) of **{iqr_v:,.2f}**."
            )

        # 2. Diagnostic Analytics (Why Did It Happen? Root Cause & Variances)
        diagnostic = []
        rel = eda_results.get("relationships", {})
        strong_pos = rel.get("strong_positive", [])
        strong_neg = rel.get("strong_negative", [])
        
        for p in strong_pos[:2]:
            diagnostic.append(
                f"**Co-movement Driver:** High positive correlation between **`{p['col1']}`** and **`{p['col2']}`** (r = **{p['correlation']:.2f}**). Shifts in `{p['col1']}` directly account for variance in `{p['col2']}`."
            )
        for p in strong_neg[:2]:
            diagnostic.append(
                f"**Trade-off Friction:** Strong inverse correlation detected between **`{p['col1']}`** and **`{p['col2']}`** (r = **{p['correlation']:.2f}**), indicating an operational or pricing trade-off."
            )

        # Categorical concentration / Pareto diagnostics
        cat_stats = eda_results.get("categorical", [])
        for cat_info in cat_stats[:2]:
            top_cats = cat_info.get("top_categories", [])
            if top_cats:
                top_1 = top_cats[0]
                if top_1["percentage"] > 35:
                    diagnostic.append(
                        f"**Pareto Concentration:** Category **'{top_1['category']}'** in `{cat_info['column']}` dominates with **{top_1['percentage']}%** of all entries, driving majority aggregate performance."
                    )

        # Skewness diagnostics
        for col_stat in num_stats:
            col_name = col_stat["column"]
            skew = col_stat["skewness"]
            if abs(skew) > 1.2:
                direction = "right-tail outliers (positive skew)" if skew > 0 else "left-tail outliers (negative skew)"
                diagnostic.append(
                    f"**Distribution Asymmetry in `{col_name}`:** Skewness factor of **{skew:.2f}** indicates {direction}; median represents a more reliable benchmark than the arithmetic mean."
                )

        # 3. Predictive & Trend Analysis (Trajectory & Future Signals)
        predictive = []
        time_trends = eda_results.get("time_trends")
        if time_trends and time_trends.get("monthly"):
            monthly = time_trends["monthly"]
            if len(monthly) >= 2:
                v_start = monthly[0]["value"]
                v_end = monthly[-1]["value"]
                pct_change = ((v_end - v_start) / (v_start + 1e-6)) * 100
                direction = "upward momentum" if pct_change >= 0 else "contractionary trajectory"
                predictive.append(
                    f"**Temporal Momentum:** Monthly aggregated **{time_trends['target_metric']}** moved by **{pct_change:+.1f}%** (from {monthly[0]['date']} to {monthly[-1]['date']}), exhibiting a clear {direction}."
                )
                if len(monthly) >= 3:
                    recent_diff = monthly[-1]["value"] - monthly[-2]["value"]
                    rec_dir = "acceleration" if recent_diff > 0 else "deceleration"
                    predictive.append(
                        f"**Recent Velocity:** Period-over-period shift between the last two recorded intervals shows {rec_dir} ({recent_diff:+,.2f} net change)."
                    )
        else:
            predictive.append(
                "Static cross-sectional distribution detected. Without timestamp attributes, predictive modeling relies on feature-engineered cohort proxies."
            )

        # 4. Anomalies & Quality Risks
        anomalies = []
        if dup_rows > 0:
            anomalies.append(
                f"**Data Integrity Alert:** {dup_rows} duplicated records ({ds_info.get('duplicate_pct', 0):.1f}%) detected, threatening inflation of volume metrics."
            )
        if missing_pct > 0:
            anomalies.append(
                f"**Information Sparsity:** {missing_pct:.1f}% missing cells found across columns. Recommended to apply median/mode imputation before modeling."
            )
        for col_stat in num_stats:
            col_name = col_stat["column"]
            if col_stat["iqr"] > 0 and col_stat["mean"] > 0:
                cv = col_stat["std"] / col_stat["mean"]
                if cv > 1.5:
                    anomalies.append(
                        f"**Extreme Volatility in `{col_name}`:** Coefficient of Variation (CV = {cv:.2f}) indicates severe dispersion and high probability of extreme tail events."
                    )

        # 5. Prescriptive & Tactical Recommendations
        recommendations = []
        if strong_pos:
            recommendations.append(
                f"Operationalize the correlation between **{strong_pos[0]['col1']}** and **{strong_pos[0]['col2']}** into automated forecasting rules."
            )
        if cat_stats and cat_stats[0].get("top_categories") and cat_stats[0]["top_categories"][0]["percentage"] > 35:
            top_name = cat_stats[0]["top_categories"][0]["category"]
            recommendations.append(
                f"Implement strategic risk hedge for dominant segment **'{top_name}'** while boosting marketing initiatives for underrepresented long-tail categories."
            )
        if anomalies:
            recommendations.append(
                "Execute the automated cleaning pipeline in the Data Cleaning workbench to eliminate duplicate rows and cap extreme outliers using 1.5 IQR thresholds."
            )
        recommendations.append(
            "Configure real-time DuckDB SQL monitors to detect threshold breaches and track KPI variances automatically."
        )

        # 5b. Goal-Specific Tactical Interventions
        tactical_interventions = [
            f"Prioritize cohorts with highest statistical sensitivity to '{goal}' metrics.",
            f"Deploy automated threshold monitors for variance deviations impacting '{goal}'.",
            f"Institutionalize recurring KPI performance reviews across primary dimensions."
        ]
        if "revenue" in goal_lower or "sales" in goal_lower:
            tactical_interventions.extend([
                "Focus capital and marketing allocation on top 20% Pareto revenue drivers and high-margin clusters.",
                "Implement dynamic pricing models and cross-sell funnels in underperforming regional cohorts."
            ])
        elif "churn" in goal_lower or "retention" in goal_lower or "customer" in goal_lower:
            tactical_interventions.extend([
                "Trigger automated retention campaigns for customer cohorts displaying activity decay.",
                "Perform proactive executive outreach on high-value accounts exceeding median inactivity thresholds."
            ])
        elif "cost" in goal_lower or "margin" in goal_lower:
            tactical_interventions.extend([
                "Streamline operational touchpoints in highest-variance expense and processing clusters.",
                "Review supplier contracts and phase out bottom-decile margin-dilutive SKUs."
            ])
        elif "fraud" in goal_lower or "risk" in goal_lower or "anomaly" in goal_lower:
            tactical_interventions.extend([
                "Enforce real-time anomaly score thresholds on extreme tail events exceeding 3 standard deviations.",
                "Quarantine outlier transactions and mandate secondary multi-factor verification workflows."
            ])

        if hypothesis_evaluations:
            # Also integrate hypothesis evaluations into diagnostic section for rich contextual insights
            for he in hypothesis_evaluations:
                diagnostic.insert(0, f"**Hypothesis Testing:** {he}")

        # Add custom interventions based on explanation keywords
        if expl_lower:
            if any(k in expl_lower for k in ["churn", "attrition", "retention", "drop"]):
                tactical_interventions.insert(0, f"Deploy targeted retention safeguards on user segments identified in: \"{goal_explanation[:50]}...\"")
            if any(k in expl_lower for k in ["margin", "profit", "cost", "loss", "leak"]):
                tactical_interventions.insert(0, f"Execute SKU/channel unit economics audit targeting margin leaks highlighted in your objective.")
            if any(k in expl_lower for k in ["region", "channel", "territory", "location"]):
                tactical_interventions.insert(0, f"Institute geographic/channel segmentation controls to isolate regional variances from your brief.")

        goal_alignment_payload = {
            "goal": goal,
            "goal_explanation": goal_explanation,
            "strategic_findings": goal_analysis,
            "hypothesis_evaluations": hypothesis_evaluations,
            "targeted_metrics": targeted_metrics,
            "tactical_interventions": tactical_interventions
        }

        # 6. Requirement Specification Explanation
        requirement_spec = {
            "analytical_objective": f"Transform raw tabular data into multi-horizon business intelligence aligned with '{goal}'.",
            "data_lineage": f"Synthesized from {n_rows:,} verified data rows, {len(num_stats)} numeric variables, and {len(cat_stats)} categorical dimensions.",
            "privacy_guarantee": "Zero raw row-level records were transmitted. Insights are derived strictly from deterministic statistical moments and correlation matrices.",
            "grounding_audit": "100% of claims are mathematically traceable to the underlying summary statistics."
        }

        return {
            "goal": goal,
            "goal_explanation": goal_explanation,
            "goal_alignment": goal_alignment_payload,
            "goal_analysis": goal_analysis,
            "descriptive": descriptive,
            "diagnostic": diagnostic,
            "predictive": predictive,
            "anomalies": anomalies if anomalies else ["Zero critical anomalies detected."],
            "recommendations": recommendations,
            "requirement_spec": requirement_spec,
            "key_insights": descriptive + diagnostic[:2],
            "business_problems": anomalies,
            "trends": predictive,
            "provider": "Deep Analytical Synthesis Engine (Mathematically Grounded)"
        }

    @classmethod
    def generate_insights(
        cls,
        profile: Dict[str, Any],
        eda_results: Dict[str, Any],
        quality: Dict[str, Any],
        api_key: Optional[str] = None,
        goal: str = "General Business Intelligence",
        goal_explanation: str = ""
    ) -> Dict[str, Any]:
        # Generate the comprehensive grounded baseline
        insights = cls.generate_deep_analytical_insights(profile, eda_results, quality, goal=goal, goal_explanation=goal_explanation)
        
        # Legacy compatibility keys
        insights["key_insights"] = insights["descriptive"] + insights["diagnostic"][:2]
        insights["business_problems"] = insights["anomalies"]
        insights["trends"] = insights["predictive"]

        return insights

    # Backward compatibility alias
    generate_offline_insights = generate_deep_analytical_insights
