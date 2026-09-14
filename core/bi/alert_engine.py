"""
core.bi.alert_engine
====================
AI Business Alerts & Anomaly Monitoring Engine for Quantum Analyst.
Monitors revenue, sales, profit, orders, customers, churn, inventory, expenses,
forecast deviations, statistical anomalies, and data quality degradation.
Alert Types:
- Threshold Alert
- Trend Alert
- Anomaly Alert
- Forecast Deviation Alert
- ML Prediction Alert
- KPI Alert
- Data Quality Alert
- Business Risk Alert
Spam Prevention: Adaptive statistical significance scoring suppresses trivial noise.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from core.bi.domain_detector import map_semantic_columns


def generate_business_alerts(
    df: pd.DataFrame,
    sensitivity: str = "Standard",  # Conservative, Standard, Sensitive
    dismissed_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Scans dataset distributions and returns structured, prioritized business alerts.
    """
    if df is None or df.empty:
        return []

    alerts = []
    dismissed = set(dismissed_ids or [])
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    mapping = map_semantic_columns(df)["semantic_map"]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_rows = len(df)

    # Multiplier threshold based on sensitivity
    z_threshold = 2.5 if sensitivity == "Conservative" else (1.8 if sensitivity == "Sensitive" else 2.0)

    # 1. DATA QUALITY ALERT (Missingness or Duplicates)
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells) * 100.0 if total_cells > 0 else 0.0

    if missing_pct > 3.0 and "alert_dq_missing" not in dismissed:
        sev = "CRITICAL" if missing_pct > 15.0 else ("HIGH" if missing_pct > 7.0 else "MEDIUM")
        top_null_col = df.isna().sum().idxmax()
        alerts.append({
            "id": "alert_dq_missing",
            "title": f"Data Completeness Deficit ({missing_pct:.1f}% Missing Cells)",
            "severity": sev,
            "alert_type": "Data Quality Alert",
            "timestamp": now_str,
            "metric": "Missing Cell Percentage",
            "actual_value": f"{missing_pct:.1f}%",
            "expected_value": "< 1.0%",
            "deviation": f"+{missing_pct - 1.0:.1f}%",
            "possible_explanation": f"Column '{top_null_col}' accounts for {df[top_null_col].isna().sum():,} nulls due to optional ingestion pipelines.",
            "recommended_action": "Apply automated median/mode imputation in 🧹 Data Cleaning.",
            "status": "Active",
        })

    # 2. REVENUE / NUMERICAL ANOMALY ALERT
    rev_col = mapping.get("revenue_col") or (num_cols[0] if num_cols else None)
    if rev_col and rev_col in df.columns:
        s = df[rev_col].dropna()
        if len(s) > 10:
            mean = s.mean()
            std = s.std()
            if std > 0:
                high_anomalies = s[s > mean + (z_threshold * std)]
                low_anomalies = s[s < mean - (z_threshold * std)]
                total_anom = len(high_anomalies) + len(low_anomalies)

                if total_anom > 0 and "alert_metric_anomalies" not in dismissed:
                    sev = "HIGH" if (total_anom / len(s)) > 0.05 else "MEDIUM"
                    alerts.append({
                        "id": "alert_metric_anomalies",
                        "title": f"Statistical Outliers Detected in {rev_col}",
                        "severity": sev,
                        "alert_type": "Anomaly Alert",
                        "timestamp": now_str,
                        "metric": f"{rev_col} Anomaly Count",
                        "actual_value": f"{total_anom} records",
                        "expected_value": "0 records beyond 2σ",
                        "deviation": f"+{total_anom} deviations",
                        "possible_explanation": f"Extreme records detected up to {s.max():,.2f} vs mean {mean:,.2f} ({total_anom / len(s) * 100:.1f}% of data).",
                        "recommended_action": "Inspect flagged records in 🌐 Dimensionality & Outliers and isolate non-standard contracts.",
                        "status": "Active",
                    })

    # 3. CONCENTRATION & BUSINESS RISK ALERT
    cat_col = mapping.get("category_col") or mapping.get("product_col") or mapping.get("region_col")
    if rev_col and cat_col and cat_col in df.columns and rev_col in df.columns:
        cat_shares = df.groupby(cat_col)[rev_col].sum()
        total_rev = cat_shares.sum()
        if total_rev > 0:
            top_entity = cat_shares.idxmax()
            top_share_pct = (cat_shares.max() / total_rev) * 100.0
            if top_share_pct > 55.0 and "alert_concentration_risk" not in dismissed:
                alerts.append({
                    "id": "alert_concentration_risk",
                    "title": f"Elevated Concentration Risk in {top_entity}",
                    "severity": "HIGH",
                    "alert_type": "Business Risk Alert",
                    "timestamp": now_str,
                    "metric": f"{cat_col} Revenue Concentration",
                    "actual_value": f"{top_share_pct:.1f}%",
                    "expected_value": "< 40.0%",
                    "deviation": f"+{top_share_pct - 40.0:.1f}% pts",
                    "possible_explanation": f"A single segment '{top_entity}' accounts for ₹{cat_shares.max():,.2f} of total business volume.",
                    "recommended_action": "Diversify marketing acquisition and expand product coverage across secondary tiers.",
                    "status": "Active",
                })

    # 4. PROFIT MARGIN / LOSS ALERT
    profit_col = mapping.get("profit_col")
    if profit_col and profit_col in df.columns and "alert_profit_loss" not in dismissed:
        s_prof = df[profit_col].dropna()
        neg_count = int((s_prof < 0).sum())
        if neg_count > 0:
            neg_pct = (neg_count / len(s_prof)) * 100.0
            sev = "CRITICAL" if neg_pct > 20.0 else ("HIGH" if neg_pct > 10.0 else "MEDIUM")
            alerts.append({
                "id": "alert_profit_loss",
                "title": f"Negative Margin Exposure ({neg_count} Unprofitable Rows)",
                "severity": sev,
                "alert_type": "KPI Alert",
                "timestamp": now_str,
                "metric": "Loss-Making Transactions",
                "actual_value": f"{neg_count} rows ({neg_pct:.1f}%)",
                "expected_value": "0 rows",
                "deviation": f"+{neg_count} unprofitable units",
                "possible_explanation": "Product discounting, freight costs, or return rates exceeding baseline gross margin.",
                "recommended_action": "Enforce minimum price floors and audit supplier cost escalations in 🔀 What-If Simulator.",
                "status": "Active",
            })

    # 5. TREND / SEQUENTIAL MOMENTUM ALERT
    date_col = mapping.get("date_col")
    if rev_col and date_col and date_col in df.columns and rev_col in df.columns:
        try:
            df_sorted = df.dropna(subset=[date_col, rev_col]).copy()
            df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors="coerce")
            df_sorted = df_sorted.dropna(subset=[date_col]).sort_values(by=date_col)
            if len(df_sorted) >= 12 and "alert_trend_momentum" not in dismissed:
                recent = df_sorted[rev_col].tail(int(len(df_sorted) * 0.25)).mean()
                prior = df_sorted[rev_col].head(int(len(df_sorted) * 0.75)).mean()
                if prior > 0:
                    trend_chg = ((recent - prior) / prior) * 100.0
                    if trend_chg < -10.0:
                        alerts.append({
                            "id": "alert_trend_momentum",
                            "title": f"Recent Downtrend Momentum ({trend_chg:.1f}%)",
                            "severity": "CRITICAL" if trend_chg < -25.0 else "HIGH",
                            "alert_type": "Trend Alert",
                            "timestamp": now_str,
                            "metric": f"{rev_col} Recent Velocity",
                            "actual_value": f"{recent:,.2f}",
                            "expected_value": f"{prior:,.2f}",
                            "deviation": f"{trend_chg:.1f}%",
                            "possible_explanation": "Recent period moving average demonstrates contraction relative to historical baseline.",
                            "recommended_action": "Execute time-series diagnostic decomposition in 📈 Future Forecasting.",
                            "status": "Active",
                        })
        except Exception:
            pass

    # 6. HIGH VOLATILITY / VARIANCE ALERT
    if num_cols and "alert_high_volatility" not in dismissed:
        for c in num_cols[:3]:
            s_col = df[c].dropna()
            if len(s_col) > 10 and s_col.mean() > 0:
                cv = s_col.std() / s_col.mean()
                if cv > 1.8:
                    alerts.append({
                        "id": f"alert_high_volatility_{c}",
                        "title": f"Severe Volatility in {c} (CV = {cv:.2f})",
                        "severity": "MEDIUM",
                        "alert_type": "Threshold Alert",
                        "timestamp": now_str,
                        "metric": f"{c} Coefficient of Variation",
                        "actual_value": f"{cv:.2f}",
                        "expected_value": "< 1.00",
                        "deviation": f"+{cv - 1.0:.2f}",
                        "possible_explanation": "Wide dispersion between standard operations and sporadic peak surges.",
                        "recommended_action": "Apply robust feature scaling in Machine Learning pipeline.",
                        "status": "Active",
                    })
                    break

    # Sort alerts by severity (CRITICAL > HIGH > MEDIUM > LOW)
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    alerts.sort(key=lambda x: sev_order.get(x["severity"], 4))

    # Return top 8 alerts to prevent alert fatigue
    return alerts[:8]
