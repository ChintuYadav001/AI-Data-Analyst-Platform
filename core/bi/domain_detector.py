"""
core.bi.domain_detector
=======================
Automated dataset business domain classification and semantic column mapping.
Identifies domain contexts (E-commerce, Retail, Finance, SaaS, Customer, Operations, HR, Sales)
and maps semantic business variables (revenue, profit, cost, date, customer, product, region, etc.).
Safeguard: Never invents missing dimensions; explicitly reports data limitations.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


DOMAIN_KEYWORDS = {
    "Sales & Distribution": ["sales", "revenue", "deal", "quota", "rep", "territory", "pipeline", "discount", "invoice"],
    "E-commerce & Retail": ["order", "cart", "sku", "product", "inventory", "shipping", "return", "item", "checkout"],
    "Finance & Banking": ["profit", "revenue", "expense", "cost", "margin", "ebitda", "budget", "asset", "liability", "tax", "fee"],
    "SaaS & Subscriptions": ["mrr", "arr", "churn", "subscriber", "subscription", "plan", "renewal", "cac", "ltv", "tier"],
    "Customer & CRM": ["customer", "client", "gender", "age", "income", "segment", "loyalty", "satisfaction", "nps", "account"],
    "Operations & Logistics": ["warehouse", "delivery", "carrier", "delay", "defect", "downtime", "transit", "freight", "logistics"],
    "HR & Workforce": ["employee", "salary", "attrition", "tenure", "department", "hiring", "compensation", "leave"],
    "Healthcare & Life Sciences": ["patient", "diagnosis", "treatment", "doctor", "hospital", "dosage", "claim", "admit"],
}


DOMAIN_PRIMARY_KPIS = {
    "Sales & Distribution": ["Gross Revenue", "Win Rate", "Average Order Value", "Sales Velocity", "Quota Realization"],
    "E-commerce & Retail": ["Gross Merchandise Value", "Cart Conversion Rate", "Average Order Value", "Customer Acquisition Cost", "Return Rate"],
    "Finance & Banking": ["Operating Margin", "EBITDA", "Net Profit Ratio", "Cost to Income Ratio", "Cash Flow Velocity"],
    "SaaS & Subscriptions": ["Monthly Recurring Revenue", "Net Revenue Retention", "Churn Rate", "Customer Lifetime Value", "CAC Payback"],
    "Customer & CRM": ["Customer Retention Rate", "Net Promoter Score", "Customer Lifetime Value", "Purchase Frequency", "Decile Concentration"],
    "Operations & Logistics": ["Order Cycle Time", "Capacity Utilization", "Defect Rate", "On-Time Delivery Rate", "Inventory Turnover"],
    "HR & Workforce": ["Employee Retention", "Turnover Rate", "Revenue per Employee", "Tenure Distribution", "Hiring Velocity"],
    "Healthcare & Life Sciences": ["Patient Readmission Rate", "Treatment Length of Stay", "Bed Utilization", "Cost per Patient Day", "Clinical Error Rate"],
    "General Analytics": ["Aggregate Volume", "Variance Ratio", "Distribution Kurtosis", "Completeness Index", "Stability Index"],
}


def detect_business_domain(df: pd.DataFrame) -> Dict[str, Any]:
    """Detects the primary business domain based on column names and semantic features."""
    if df is None or df.empty:
        mapping_info = map_semantic_columns(df)
        return {
            "domain": "General Analytics",
            "confidence": "Low",
            "scores": {},
            "available_domains": list(DOMAIN_KEYWORDS.keys()) + ["General Analytics"],
            "primary_kpis": DOMAIN_PRIMARY_KPIS["General Analytics"],
            "column_mapping": mapping_info["semantic_map"],
            "semantic_map": mapping_info["semantic_map"],
            "available_dimensions": mapping_info["available_dimensions"],
            "missing_dimensions": mapping_info["missing_dimensions"],
            "disclaimer": mapping_info["disclaimer"],
        }

    cols_lower = [str(c).lower().strip() for c in df.columns]
    cols_text = " ".join(cols_lower)

    scores = {}
    for domain, kws in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in cols_text)
        scores[domain] = score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_domain, top_score = sorted_scores[0]

    if top_score >= 3:
        confidence = "High"
    elif top_score >= 1:
        confidence = "Medium"
    else:
        top_domain = "General Analytics"
        confidence = "Low"

    mapping_info = map_semantic_columns(df)
    primary_kpis = DOMAIN_PRIMARY_KPIS.get(top_domain, DOMAIN_PRIMARY_KPIS["General Analytics"])

    return {
        "domain": top_domain,
        "confidence": confidence,
        "scores": dict(sorted_scores),
        "available_domains": list(DOMAIN_KEYWORDS.keys()) + ["General Analytics"],
        "primary_kpis": primary_kpis,
        "column_mapping": mapping_info["semantic_map"],
        "semantic_map": mapping_info["semantic_map"],
        "available_dimensions": mapping_info["available_dimensions"],
        "missing_dimensions": mapping_info["missing_dimensions"],
        "disclaimer": mapping_info["disclaimer"],
    }


def map_semantic_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Intelligently maps dataset columns to business roles:
    revenue, profit, cost, date, customer, product, region, churn, quantity, category.
    """
    if df is None or df.empty:
        return {
            "semantic_map": {},
            "available_dimensions": [],
            "missing_dimensions": ["revenue", "profit", "cost", "date", "customer", "product", "region", "churn", "quantity"],
            "disclaimer": "No dataset loaded to evaluate business dimensions.",
        }

    cols = list(df.columns)
    cols_lower = {c: str(c).lower().strip() for c in cols}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    semantic_map: Dict[str, Optional[str]] = {
        "revenue_col": None,
        "profit_col": None,
        "cost_col": None,
        "date_col": None,
        "customer_col": None,
        "product_col": None,
        "region_col": None,
        "churn_col": None,
        "quantity_col": None,
        "category_col": None,
    }

    # 1. Date column
    date_candidates = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if date_candidates:
        semantic_map["date_col"] = date_candidates[0]
    else:
        for c, cl in cols_lower.items():
            if any(k in cl for k in ["date", "time", "timestamp", "period", "year", "month", "day"]):
                semantic_map["date_col"] = c
                break

    # 2. Revenue column
    for c in num_cols:
        cl = cols_lower[c]
        if any(k in cl for k in ["revenue", "sales", "total_amount", "amount", "turnover", "bill_amount", "price"]):
            semantic_map["revenue_col"] = c
            break

    # 3. Profit column
    for c in num_cols:
        cl = cols_lower[c]
        if any(k in cl for k in ["profit", "net_profit", "earnings", "net_income", "margin"]):
            semantic_map["profit_col"] = c
            break

    # 4. Cost column
    for c in num_cols:
        cl = cols_lower[c]
        if any(k in cl for k in ["cost", "expense", "spend", "cogs", "expenditure"]):
            semantic_map["cost_col"] = c
            break

    # 5. Quantity column
    for c in num_cols:
        cl = cols_lower[c]
        if any(k in cl for k in ["qty", "quantity", "units", "volume", "order_count"]):
            semantic_map["quantity_col"] = c
            break

    # 6. Customer column
    for c, cl in cols_lower.items():
        if any(k in cl for k in ["customer", "client", "user", "buyer", "account_id"]):
            semantic_map["customer_col"] = c
            break

    # 7. Product column
    for c, cl in cols_lower.items():
        if any(k in cl for k in ["product", "item", "sku", "service", "device"]):
            semantic_map["product_col"] = c
            break

    # 8. Region column
    for c, cl in cols_lower.items():
        if any(k in cl for k in ["region", "country", "state", "city", "territory", "location", "zone", "branch"]):
            semantic_map["region_col"] = c
            break

    # 9. Churn / Status column
    for c, cl in cols_lower.items():
        if any(k in cl for k in ["churn", "status", "attrition", "is_active", "retained"]):
            semantic_map["churn_col"] = c
            break

    # 10. Category column
    for c, cl in cols_lower.items():
        if any(k in cl for k in ["category", "segment", "department", "type", "class", "group"]):
            semantic_map["category_col"] = c
            break

    # Calculate available vs missing
    available = [k.replace("_col", "") for k, v in semantic_map.items() if v is not None]
    all_dims = ["revenue", "profit", "cost", "date", "customer", "product", "region", "churn", "quantity", "category"]
    missing = [d for d in all_dims if d not in available]

    disclaimer = ""
    if missing:
        disclaimer = (
            f"Notice: The active dataset lacks explicit columns for: {', '.join(missing)}. "
            f"Analyses requiring these specific metrics will display 'Insufficient data in dataset' rather than fabricated estimates."
        )

    return {
        "semantic_map": semantic_map,
        "available_dimensions": available,
        "missing_dimensions": missing,
        "disclaimer": disclaimer,
    }


def get_domain_context(df: pd.DataFrame, manual_domain: Optional[str] = None) -> Dict[str, Any]:
    """Generates complete business context combining domain detection and semantic mappings."""
    domain_info = detect_business_domain(df)
    active_domain = manual_domain if manual_domain and manual_domain != "Auto-Detect" else domain_info["domain"]
    mapping_info = map_semantic_columns(df)

    return {
        "domain": active_domain,
        "domain_confidence": domain_info["confidence"],
        "semantic_map": mapping_info["semantic_map"],
        "available_dimensions": mapping_info["available_dimensions"],
        "missing_dimensions": mapping_info["missing_dimensions"],
        "disclaimer": mapping_info["disclaimer"],
    }
