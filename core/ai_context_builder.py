import json
from typing import Dict, Any, List, Optional
from core.cache_manager import CacheManager

class AIContextBuilder:
    """Constructs token-compact (< 1,500 tokens) analytical context representations

    for LLM inference without transmitting raw multi-lakh tabular rows.
    """

    @classmethod
    def build_context(
        cls,
        profile: Dict[str, Any],
        quality: Dict[str, Any],
        quick_eda: Dict[str, Any],
        goal: str = "General Business Intelligence",
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Builds a structured prompt context summary."""
        total_rows = profile.get("rows", 0)
        total_cols = profile.get("columns", 0)
        col_names = profile.get("column_names", [])

        # Format stats concisely
        compact_stats = {}
        for c, st in list(profile.get("statistics", {}).items())[:6]:
            compact_stats[c] = {
                "mean": round(st.get("mean", 0), 2),
                "min": round(st.get("min", 0), 2),
                "max": round(st.get("max", 0), 2),
                "sum": round(st.get("sum", 0), 2)
            }

        # Format key findings
        key_findings = quick_eda.get("key_findings", [])

        # Format top correlation
        corr_info = quick_eda.get("modules", {}).get("correlation", {})
        top_corr = corr_info.get("top_pair")

        # Format outlier count
        outlier_info = quick_eda.get("modules", {}).get("outliers", {})

        context_obj = {
            "dataset_overview": {
                "total_records": total_rows,
                "total_features": total_cols,
                "feature_list": col_names[:25],
                "quality_score": quality.get("overall_score", 100),
                "strategic_goal": goal,
                "primary_target": target_col or "None designated"
            },
            "statistical_summary": compact_stats,
            "correlation_signal": f"{top_corr[0]} vs {top_corr[1]} (r={top_corr[2]})" if top_corr else "None",
            "outlier_signal": f"{outlier_info.get('outlier_count', 0):,} records ({outlier_info.get('outlier_pct', 0)}%)" if outlier_info else "0",
            "key_empirical_findings": key_findings[:5]
        }

        return context_obj

    @classmethod
    def format_as_prompt(cls, context_obj: Dict[str, Any]) -> str:
        """Serializes the context dictionary into a clean markdown prompt for LLMs."""
        ov = context_obj["dataset_overview"]
        lines = [
            f"### DATASET TELEMETRY PROFILE",
            f"- **Volume**: {ov['total_records']:,} observations | **Features**: {ov['total_features']} columns",
            f"- **Quality Score**: {ov['quality_score']}%",
            f"- **Strategic Goal**: {ov['strategic_goal']}",
            f"- **Primary Target Variable**: {ov['primary_target']}",
            "",
            "### KEY EMPIRICAL FINDINGS",
        ]
        for f in context_obj.get("key_empirical_findings", []):
            lines.append(f"- {f}")

        lines.extend([
            "",
            f"### CORRELATION & ANOMALIES",
            f"- Dominant Correlation: {context_obj.get('correlation_signal')}",
            f"- Outlier Footprint: {context_obj.get('outlier_signal')}",
            "",
            "### STATISTICAL BENCHMARKS",
            "```json",
            json.dumps(context_obj.get("statistical_summary", {}), indent=2),
            "```"
        ])
        return "\n".join(lines)
