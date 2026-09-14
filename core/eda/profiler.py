"""
Dataset Intelligence Profiler & Domain Classifier for Quantum Analyst AI EDA Engine.
Analyzes column types, semantic roles, identifiers, target candidates, data quality,
leakage risks, PII flags, class balance, and domain classification.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
from core.cache import AnalyticsCache


class EDAProfiler:
    """Extracts deep structural, semantic, and domain intelligence from a dataset."""

    PII_KEYWORDS = {
        "email", "phone", "ssn", "social_security", "password", "pwd", "credit_card",
        "first_name", "last_name", "full_name", "address", "street", "zipcode", "postal",
        "ip_address", "passport", "national_id"
    }

    IDENTIFIER_KEYWORDS = {
        "id", "uuid", "guid", "key", "pk", "fk", "code", "sku", "index", "hash"
    }

    DOMAIN_KEYWORDS = {
        "E-Commerce / Retail": [
            "order", "product", "customer", "cart", "sku", "revenue", "discount",
            "quantity", "shipping", "price", "unit_price", "purchase", "store", "sale"
        ],
        "Financial Analytics": [
            "transaction", "balance", "credit", "debit", "loan", "account", "interest",
            "fraud", "payment", "amount", "deposit", "withdrawal", "default", "equity"
        ],
        "Marketing Analytics": [
            "campaign", "click", "impression", "ctr", "cpc", "lead", "conversion",
            "roas", "cac", "channel", "ad_spend", "utm", "source", "medium", "bounce"
        ],
        "HR / People Analytics": [
            "employee", "salary", "attrition", "tenure", "department", "job_role",
            "performance", "bonus", "hire_date", "promotion", "satisfaction", "resigned"
        ],
        "Healthcare / Clinical": [
            "patient", "diagnosis", "treatment", "blood", "pressure", "heart", "glucose",
            "cholesterol", "symptom", "hospital", "doctor", "dosage", "bmi", "medical"
        ],
        "Web / App Analytics": [
            "session", "pageview", "page", "visit", "device", "browser", "referrer",
            "duration", "bounce_rate", "events", "screen", "os", "platform"
        ],
        "Customer / Subscription": [
            "churn", "churned", "subscriber", "mrr", "arr", "plan", "contract",
            "renewal", "monthly_charges", "total_charges", "tenure"
        ]
    }

    TARGET_PATTERNS = [
        (r"\b(churn|churned|is_churn)\b", "classification", 0.95),
        (r"\b(default|is_default|fraud|is_fraud)\b", "classification", 0.95),
        (r"\b(converted|conversion|is_converted|subscribe|bought)\b", "classification", 0.90),
        (r"\b(attrition|left_company|turnover)\b", "classification", 0.90),
        (r"\b(status|target|outcome|result|class|label)\b", "classification", 0.85),
        (r"\b(revenue|total_revenue|net_revenue)\b", "regression", 0.95),
        (r"\b(profit|net_profit|operating_profit)\b", "regression", 0.95),
        (r"\b(sales|total_sales|gross_sales)\b", "regression", 0.90),
        (r"\b(price|unit_price|selling_price|fare)\b", "regression", 0.88),
        (r"\b(amount|total_amount|spend|spending)\b", "regression", 0.85),
        (r"\b(salary|wage|compensation)\b", "regression", 0.85),
        (r"\b(score|rating|customer_rating|satisfaction)\b", "regression", 0.80),
        (r"\b(quantity|units_sold|volume)\b", "regression", 0.78)
    ]

    @classmethod
    def profile_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Performs a comprehensive dataset intelligence audit."""
        if df is None or df.empty:
            return cls._empty_profile()

        n_rows, n_cols = df.shape
        if n_rows > 10000:
            sample_n = min(2000, n_rows)
            sample_mem = int(df.sample(sample_n, random_state=42).memory_usage(deep=True).sum())
            memory_bytes = int((sample_mem / sample_n) * n_rows)
        else:
            memory_bytes = int(df.memory_usage(deep=True).sum())
        memory_mb = round(float(memory_bytes) / (1024 * 1024), 2)

        cached_dups = AnalyticsCache.get(df, "duplicate_rows")
        if cached_dups is not None:
            duplicate_rows = cached_dups
        else:
            duplicate_rows = int(df.duplicated().sum())
            AnalyticsCache.set(df, "duplicate_rows", duplicate_rows)
        duplicate_pct = round((duplicate_rows / n_rows) * 100, 2) if n_rows > 0 else 0.0

        total_cells = n_rows * n_cols
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

        numerical_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []
        boolean_cols: List[str] = []
        text_cols: List[str] = []
        identifier_cols: List[str] = []
        high_cardinality_cols: List[str] = []
        constant_cols: List[str] = []
        near_constant_cols: List[str] = []
        pii_cols: List[str] = []

        col_profiles: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            series = df[col]
            clean_name = str(col).lower().strip()
            n_unique = int(series.nunique(dropna=True))
            col_missing = int(series.isna().sum())
            col_missing_pct = round((col_missing / n_rows) * 100, 2) if n_rows > 0 else 0.0

            # 1. Detect Identifiers
            is_id = False
            has_id_keyword = any(clean_name == kw or clean_name.startswith(kw + "_") or clean_name.endswith("_" + kw) or ("_" + kw + "_") in clean_name for kw in cls.IDENTIFIER_KEYWORDS)
            if has_id_keyword:
                if n_unique > 0.5 * n_rows or n_unique > 20:
                    is_id = True
            elif not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
                if n_unique > 0.85 * n_rows and n_rows > 15:
                    is_id = True

            if is_id:
                identifier_cols.append(str(col))

            # 2. Detect PII
            if any(re.search(r"\b" + re.escape(kw) + r"\b", clean_name) for kw in cls.PII_KEYWORDS):
                pii_cols.append(str(col))

            # 3. Detect Constant & Near Constant
            if n_unique <= 1:
                constant_cols.append(str(col))
            elif n_unique > 1 and n_rows >= 20:
                if n_rows > 10000:
                    sample_head = series.dropna().iloc[:2000]
                    sample_vc = sample_head.value_counts(normalize=True)
                    if not sample_vc.empty and sample_vc.iloc[0] >= 0.90:
                        top_freq = series.value_counts(normalize=True, dropna=True)
                        if not top_freq.empty and top_freq.iloc[0] >= 0.95:
                            near_constant_cols.append(str(col))
                else:
                    top_freq = series.value_counts(normalize=True, dropna=True)
                    if not top_freq.empty and top_freq.iloc[0] >= 0.95:
                        near_constant_cols.append(str(col))

            # 4. Column Type Detection
            inferred_type = "unknown"
            if pd.api.types.is_bool_dtype(series):
                boolean_cols.append(str(col))
                inferred_type = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(series):
                datetime_cols.append(str(col))
                inferred_type = "datetime"
            elif pd.api.types.is_numeric_dtype(series):
                is_bool_num = False
                if n_unique <= 2:
                    unique_vals = set(series.dropna().unique())
                    if unique_vals.issubset({0, 1, 0.0, 1.0}):
                        is_bool_num = True
                if is_bool_num:
                    boolean_cols.append(str(col))
                    inferred_type = "boolean"
                elif is_id:
                    inferred_type = "identifier"
                else:
                    numerical_cols.append(str(col))
                    inferred_type = "numerical"
            else:
                is_date_candidate = False
                if any(kw in clean_name for kw in ["date", "time", "timestamp", "year", "month", "day", "created_at"]):
                    try:
                        sample = series.dropna().iloc[:20]
                        if len(sample) > 0:
                            pd.to_datetime(sample, errors="raise")
                            datetime_cols.append(str(col))
                            inferred_type = "datetime"
                            is_date_candidate = True
                    except Exception:
                        pass

                if not is_date_candidate:
                    if n_unique > 0:
                        sample_for_len = series.dropna().head(1000).astype(str)
                        avg_len = float(sample_for_len.str.len().mean()) if not sample_for_len.empty else 0.0
                    else:
                        avg_len = 0.0
                    if avg_len > 60:
                        text_cols.append(str(col))
                        inferred_type = "text"
                    elif is_id:
                        inferred_type = "identifier"
                    else:
                        categorical_cols.append(str(col))
                        inferred_type = "categorical"
                        if n_unique > 50 and (n_unique / n_rows) > 0.1:
                            high_cardinality_cols.append(str(col))

            col_profiles[str(col)] = {
                "name": str(col),
                "inferred_type": inferred_type,
                "n_unique": n_unique,
                "missing_count": col_missing,
                "missing_pct": col_missing_pct,
                "is_identifier": is_id,
                "is_pii": str(col) in pii_cols,
                "is_constant": str(col) in constant_cols or str(col) in near_constant_cols
            }

        class_imbalances = cls._detect_class_imbalances(df, categorical_cols + boolean_cols)
        date_coverage = cls._extract_date_coverage(df, datetime_cols)
        target_candidates = cls._detect_target_candidates(df, numerical_cols, categorical_cols, boolean_cols)
        domain, domain_confidence = cls._classify_domain(df)

        summary_narrative = cls._generate_intelligence_summary(
            n_rows=n_rows,
            n_cols=n_cols,
            domain=domain,
            target_candidates=target_candidates,
            datetime_cols=datetime_cols,
            identifier_cols=identifier_cols,
            missing_pct=missing_pct,
            constant_cols=constant_cols,
            pii_cols=pii_cols
        )

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "memory_mb": memory_mb,
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": duplicate_pct,
            "total_missing": total_missing,
            "missing_pct": missing_pct,
            "columns_by_type": {
                "numerical": numerical_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols,
                "boolean": boolean_cols,
                "text": text_cols,
                "identifier": identifier_cols
            },
            "flags": {
                "high_cardinality": high_cardinality_cols,
                "constant": constant_cols,
                "near_constant": near_constant_cols,
                "pii_sensitive": pii_cols,
                "class_imbalances": class_imbalances
            },
            "date_coverage": date_coverage,
            "target_candidates": target_candidates,
            "domain": domain,
            "domain_confidence": domain_confidence,
            "column_profiles": col_profiles,
            "summary_narrative": summary_narrative
        }

    @classmethod
    def _detect_class_imbalances(cls, df: pd.DataFrame, cat_cols: List[str]) -> List[Dict[str, Any]]:
        imbalances = []
        n_rows = len(df)
        if n_rows < 20:
            return imbalances

        for col in cat_cols:
            series = df[col].dropna()
            eval_series = series.sample(10000, random_state=42) if n_rows > 25000 else series
            if eval_series.nunique() in [2, 3]:
                dist = eval_series.value_counts(normalize=True)
                majority_pct = float(dist.iloc[0]) * 100
                minority_pct = float(dist.iloc[-1]) * 100
                if majority_pct >= 80.0:
                    imbalances.append({
                        "column": str(col),
                        "majority_class": str(dist.index[0]),
                        "majority_pct": round(majority_pct, 1),
                        "minority_class": str(dist.index[-1]),
                        "minority_pct": round(minority_pct, 1),
                        "severity": "Severe" if majority_pct >= 90 else "Moderate"
                    })
        return imbalances

    @classmethod
    def _extract_date_coverage(cls, df: pd.DataFrame, datetime_cols: List[str]) -> Optional[Dict[str, Any]]:
        if not datetime_cols:
            return None

        primary_date_col = datetime_cols[0]
        try:
            s = df[primary_date_col].dropna()
            if s.empty:
                return None
            if pd.api.types.is_datetime64_any_dtype(s):
                min_date = s.min()
                max_date = s.max()
            else:
                first_str = str(s.iloc[0]).strip()
                if len(first_str) >= 10 and (first_str[4] == '-' or first_str[4] == '/'):
                    min_date = pd.to_datetime(s.min())
                    max_date = pd.to_datetime(s.max())
                else:
                    parsed = pd.to_datetime(s, errors="coerce").dropna()
                    if parsed.empty:
                        return None
                    min_date = parsed.min()
                    max_date = parsed.max()
            span_days = int((max_date - min_date).days)
            return {
                "column": primary_date_col,
                "min_date": min_date.strftime("%Y-%m-%d"),
                "max_date": max_date.strftime("%Y-%m-%d"),
                "span_days": span_days,
                "span_years": round(span_days / 365.25, 1)
            }
        except Exception:
            return None

    @classmethod
    def _detect_target_candidates(
        cls, df: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        bool_cols: List[str]
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen_cols: Set[str] = set()
        all_cols = num_cols + cat_cols + bool_cols

        for col in all_cols:
            col_lower = str(col).lower().strip()
            for pattern, task_type, confidence in cls.TARGET_PATTERNS:
                if re.search(pattern, col_lower) and col not in seen_cols:
                    if task_type == "classification" and (col in cat_cols or col in bool_cols or (col in num_cols and df[col].nunique() <= 10)):
                        candidates.append({
                            "column": str(col),
                            "task_type": "classification",
                            "confidence": confidence,
                            "unique_classes": int(df[col].nunique()),
                            "rationale": f"Identified as candidate classification target via pattern '{pattern}'"
                        })
                        seen_cols.add(col)
                        break
                    elif task_type == "regression" and col in num_cols:
                        candidates.append({
                            "column": str(col),
                            "task_type": "regression",
                            "confidence": confidence,
                            "rationale": f"Identified as continuous business KPI/target via pattern '{pattern}'"
                        })
                        seen_cols.add(col)
                        break

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates

    @classmethod
    def _classify_domain(cls, df: pd.DataFrame) -> Tuple[str, float]:
        col_names = [str(c).lower() for c in df.columns]
        domain_scores: Dict[str, int] = {d: 0 for d in cls.DOMAIN_KEYWORDS}

        for col in col_names:
            for domain, keywords in cls.DOMAIN_KEYWORDS.items():
                for kw in keywords:
                    if kw in col:
                        domain_scores[domain] += 1

        best_domain = "General Enterprise Tabular"
        max_matches = 0

        for domain, score in domain_scores.items():
            if score > max_matches:
                max_matches = score
                best_domain = domain

        if max_matches >= 3:
            confidence = 0.95
        elif max_matches >= 1:
            confidence = 0.75
        else:
            confidence = 0.50

        return best_domain, confidence

    @classmethod
    def _generate_intelligence_summary(
        cls,
        n_rows: int,
        n_cols: int,
        domain: str,
        target_candidates: List[Dict[str, Any]],
        datetime_cols: List[str],
        identifier_cols: List[str],
        missing_pct: float,
        constant_cols: List[str],
        pii_cols: List[str]
    ) -> str:
        parts: List[str] = []
        parts.append(f"Dataset contains {n_rows:,} records across {n_cols} attributes.")
        parts.append(f"Domain intelligence classifies the schema as **{domain}**.")

        if target_candidates:
            top_target = target_candidates[0]
            parts.append(
                f"Primary analytical target identified as **`{top_target['column']}`** "
                f"({top_target['task_type'].title()} objective)."
            )
        else:
            parts.append("No explicit single target detected; multi-dimensional exploration recommended.")

        if datetime_cols:
            parts.append(f"Temporal dimension **`{datetime_cols[0]}`** enables chronological wave and velocity analytics.")

        if identifier_cols:
            id_names = ", ".join([f"`{c}`" for c in identifier_cols[:3]])
            parts.append(f"Entity identifiers detected ({id_names}) and isolated from numerical correlation modeling.")

        if missing_pct > 0:
            parts.append(f"Overall missingness stands at {missing_pct:.1f}%.")

        if constant_cols:
            parts.append(f"Zero-variance attributes ({', '.join(constant_cols[:2])}) quarantined.")

        return " ".join(parts)

    @classmethod
    def _empty_profile(cls) -> Dict[str, Any]:
        return {
            "n_rows": 0,
            "n_cols": 0,
            "memory_mb": 0.0,
            "duplicate_rows": 0,
            "duplicate_pct": 0.0,
            "total_missing": 0,
            "missing_pct": 0.0,
            "columns_by_type": {
                "numerical": [], "categorical": [], "datetime": [],
                "boolean": [], "text": [], "identifier": []
            },
            "flags": {
                "high_cardinality": [], "constant": [], "near_constant": [],
                "pii_sensitive": [], "class_imbalances": []
            },
            "date_coverage": None,
            "target_candidates": [],
            "domain": "Empty Dataset",
            "domain_confidence": 0.0,
            "column_profiles": {},
            "summary_narrative": "No dataset loaded."
        }
