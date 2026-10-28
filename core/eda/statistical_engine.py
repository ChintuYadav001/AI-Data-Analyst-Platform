"""
Statistical Moments, Distribution Intelligence & Hypothesis Testing Engine.
Computes univariate moments, skewness/kurtosis, distribution shapes, recommended transforms,
categorical Pareto distributions, and automated assumption-checked hypothesis testing
with effect sizes and strict non-causal language.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats


class EDAStatisticalEngine:
    """Enterprise statistical analysis engine providing moments, distributions, and inferential tests."""

    # -------------------------------------------------------------------------
    # 1. UNIVARIATE NUMERICAL ANALYSIS
    # -------------------------------------------------------------------------
    @classmethod
    def analyze_numerical_moments(cls, df: pd.DataFrame, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Calculates moments, quartiles, percentiles, skewness, kurtosis, and distribution shapes."""
        if df is None or df.empty:
            return []

        if columns is not None:
            num_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        else:
            num_cols = list(df.select_dtypes(include=[np.number]).columns)

        results: List[Dict[str, Any]] = []

        for col in num_cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            n_valid = len(series)
            if n_valid < 2:
                continue

            try:
                mean_val = float(series.mean())
                median_val = float(series.median())
                std_val = float(series.std()) if n_valid > 1 else 0.0
                var_val = float(series.var()) if n_valid > 1 else 0.0
                min_val = float(series.min())
                max_val = float(series.max())
                q_vals = series.quantile([0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99])
                p5 = float(q_vals.loc[0.05])
                p10 = float(q_vals.loc[0.10])
                q25 = float(q_vals.loc[0.25])
                q50 = median_val
                q75 = float(q_vals.loc[0.75])
                p90 = float(q_vals.loc[0.90])
                p95 = float(q_vals.loc[0.95])
                p99 = float(q_vals.loc[0.99])
                iqr_val = q75 - q25

                skew_val = float(series.skew()) if n_valid > 2 else 0.0
                kurt_val = float(series.kurtosis()) if n_valid > 3 else 0.0

                if n_valid > 10000:
                    mode_s = series.head(5000).mode()
                    mode_val = float(mode_s.iloc[0]) if not mode_s.empty else median_val
                else:
                    mode_s = series.mode()
                    mode_val = float(mode_s.iloc[0]) if not mode_s.empty else median_val

                zero_count = int((series == 0).sum())
                zero_pct = round((zero_count / n_valid) * 100, 2)

                if zero_pct >= 25.0:
                    dist_shape = "Zero-Inflated"
                elif kurt_val > 4.0:
                    dist_shape = "Heavy-Tailed (Leptokurtic)"
                elif abs(skew_val) < 0.5 and abs(kurt_val) < 1.0:
                    dist_shape = "Normal-like (Gaussian)"
                elif skew_val >= 0.5:
                    dist_shape = "Right-Skewed (Positive Skew)"
                elif skew_val <= -0.5:
                    dist_shape = "Left-Skewed (Negative Skew)"
                elif kurt_val < -1.2:
                    dist_shape = "Platykurtic (Flat/Uniform)"
                else:
                    dist_shape = "Moderately Skewed"

                if dist_shape == "Right-Skewed (Positive Skew)" and min_val >= 0:
                    rec_transform = "Log Transformation (np.log1p) or Square Root (np.sqrt)"
                elif dist_shape == "Heavy-Tailed (Leptokurtic)":
                    rec_transform = "Robust Scaling (IQR) or Yeo-Johnson Power Transform"
                elif dist_shape == "Left-Skewed (Negative Skew)":
                    rec_transform = "Power Transform (Box-Cox/Yeo-Johnson) or Reflection Log"
                elif dist_shape == "Zero-Inflated":
                    rec_transform = "Two-part hurdle model or binary indicator + positive continuous"
                else:
                    rec_transform = "Standardization (z-score) or None required"

                n_bins = min(20, max(5, int(math.sqrt(n_valid))))
                counts, bin_edges = np.histogram(series, bins=n_bins)
                hist_data = [
                    {"bin_start": round(float(bin_edges[i]), 2), "bin_end": round(float(bin_edges[i+1]), 2), "count": int(counts[i])}
                    for i in range(len(counts))
                ]

                results.append({
                    "column": str(col),
                    "count": n_valid,
                    "missing": int(df[col].isna().sum()),
                    "mean": round(mean_val, 4),
                    "median": round(median_val, 4),
                    "mode": round(mode_val, 4),
                    "std": round(std_val, 4),
                    "variance": round(var_val, 4),
                    "min": round(min_val, 4),
                    "max": round(max_val, 4),
                    "q25": round(q25, 4),
                    "q50": round(q50, 4),
                    "q75": round(q75, 4),
                    "iqr": round(iqr_val, 4),
                    "p5": round(p5, 4),
                    "p10": round(p10, 4),
                    "p90": round(p90, 4),
                    "p95": round(p95, 4),
                    "p99": round(p99, 4),
                    "skewness": round(skew_val, 4),
                    "kurtosis": round(kurt_val, 4),
                    "zero_pct": zero_pct,
                    "distribution_shape": dist_shape,
                    "recommended_transform": rec_transform,
                    "histogram": hist_data
                })
            except Exception:
                continue

        return results

    # -------------------------------------------------------------------------
    # 2. UNIVARIATE CATEGORICAL ANALYSIS
    # -------------------------------------------------------------------------
    @classmethod
    def analyze_categorical_distributions(cls, df: pd.DataFrame, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Calculates categorical frequency, Pareto distribution, concentration, and rare tails."""
        if df is None or df.empty:
            return []

        if columns is not None:
            cat_cols = [
                c for c in columns
                if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
            ]
        else:
            cat_cols = list(df.select_dtypes(include=["object", "string", "category", "bool"]).columns)

        total_rows = len(df)
        results: List[Dict[str, Any]] = []

        for col in cat_cols:
            if col not in df.columns:
                continue
            try:
                series = df[col].dropna().astype(str)
            except Exception:
                series = df[col].dropna().map(lambda x: x.decode("latin-1", errors="replace") if isinstance(x, bytes) else str(x))
            n_valid = len(series)
            if n_valid == 0:
                continue

            val_counts = series.value_counts()
            n_unique = len(val_counts)
            if n_unique > 300:
                continue

            top_cats = [
                {"category": str(k), "count": int(v), "percentage": round((v / total_rows) * 100, 2)}
                for k, v in val_counts.head(10).items()
            ]

            top3_share = sum(item["percentage"] for item in top_cats[:3])

            rare_threshold = 0.02 * total_rows
            rare_cats = [
                {"category": str(k), "count": int(v), "percentage": round((v / total_rows) * 100, 2)}
                for k, v in val_counts.items() if v <= rare_threshold
            ]

            cum_pct = (val_counts.cumsum() / n_valid) * 100
            cats_for_80 = int((cum_pct <= 80).sum()) + 1
            pareto_ratio = round((cats_for_80 / n_unique) * 100, 1) if n_unique > 0 else 100.0

            results.append({
                "column": str(col),
                "unique_count": n_unique,
                "missing": int(df[col].isna().sum()),
                "top_categories": top_cats,
                "top3_concentration_pct": round(top3_share, 2),
                "pareto_categories_for_80pct": cats_for_80,
                "pareto_ratio_pct": pareto_ratio,
                "rare_categories": rare_cats[:10],
                "rare_count_total": len(rare_cats)
            })

        return results

    # -------------------------------------------------------------------------
    # 3. AUTOMATED HYPOTHESIS TESTING (2 GROUPS)
    # -------------------------------------------------------------------------
    @classmethod
    def test_two_groups(
        cls,
        df: pd.DataFrame,
        num_col: str,
        cat_col: str,
        group1: Optional[str] = None,
        group2: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Performs automated assumption-checked comparison between two cohorts (Welch t-test vs Mann-Whitney U)."""
        if num_col not in df.columns or cat_col not in df.columns:
            return None
        if not pd.api.types.is_numeric_dtype(df[num_col]):
            return None

        clean_df = df[[num_col, cat_col]].dropna()
        top_cats = clean_df[cat_col].value_counts().index.tolist()
        if len(top_cats) < 2:
            return None

        g1 = group1 if group1 in top_cats else str(top_cats[0])
        g2 = group2 if group2 in top_cats and group2 != g1 else str(top_cats[1])

        s1 = clean_df[clean_df[cat_col] == g1][num_col]
        s2 = clean_df[clean_df[cat_col] == g2][num_col]

        n1, n2 = len(s1), len(s2)
        if n1 < 3 or n2 < 3:
            return None

        m1, m2 = float(s1.mean()), float(s2.mean())
        std1, std2 = float(s1.std()), float(s2.std())

        skew1 = abs(s1.skew()) if n1 > 3 else 0
        skew2 = abs(s2.skew()) if n2 > 3 else 0
        is_normal = (n1 >= 30 and n2 >= 30 and skew1 < 1.0 and skew2 < 1.0)

        if is_normal:
            test_name = "Welch's Two-Sample t-Test (Unequal Variances)"
            stat_res = stats.ttest_ind(s1, s2, equal_var=False)
            t_stat = float(stat_res.statistic)
            p_val = float(stat_res.pvalue)

            pooled_std = math.sqrt(((std1 ** 2) + (std2 ** 2)) / 2) if (std1 + std2) > 0 else 1.0
            effect_size = abs(m1 - m2) / pooled_std if pooled_std > 0 else 0.0
            effect_label = "Small" if effect_size < 0.5 else ("Medium" if effect_size < 0.8 else "Large")
        else:
            test_name = "Mann-Whitney U Test (Non-Parametric)"
            u_s1 = s1.sample(20000, random_state=42) if n1 > 20000 else s1
            u_s2 = s2.sample(20000, random_state=42) if n2 > 20000 else s2
            stat_res = stats.mannwhitneyu(u_s1, u_s2, alternative="two-sided")
            t_stat = float(stat_res.statistic)
            p_val = float(stat_res.pvalue)

            effect_size = abs(1.0 - (2.0 * t_stat / (len(u_s1) * len(u_s2)))) if (len(u_s1) * len(u_s2)) > 0 else 0.0
            effect_label = "Small" if effect_size < 0.3 else ("Medium" if effect_size < 0.5 else "Large")

        is_sig = p_val < 0.05
        higher_group = g1 if m1 >= m2 else g2
        diff_pct = abs(m1 - m2) / (min(m1, m2) if min(m1, m2) > 0 else 1.0) * 100

        if is_sig:
            interpretation = (
                f"Statistically significant association detected (p = {p_val:.4g}, effect size = {effect_size:.2f} [{effect_label}]). "
                f"Group '{higher_group}' exhibits a higher mean {num_col} ({max(m1, m2):.2f} vs {min(m1, m2):.2f}, Δ {diff_pct:.1f}%). "
                f"This statistical association indicates significant group disparity but does not establish causation."
            )
        else:
            interpretation = (
                f"No statistically significant difference observed between '{g1}' and '{g2}' on {num_col} "
                f"(p = {p_val:.4g} > 0.05). Any observed variance is consistent with sampling noise."
            )

        return {
            "test_name": test_name,
            "metric": str(num_col),
            "group_variable": str(cat_col),
            "group1": {"name": str(g1), "n": n1, "mean": round(m1, 2), "std": round(std1, 2)},
            "group2": {"name": str(g2), "n": n2, "mean": round(m2, 2), "std": round(std2, 2)},
            "statistic": round(t_stat, 4),
            "p_value": round(p_val, 5),
            "is_significant": is_sig,
            "effect_size": round(effect_size, 3),
            "effect_size_label": effect_label,
            "interpretation": interpretation
        }

    # -------------------------------------------------------------------------
    # 4. MULTI-GROUP TESTING (ANOVA / KRUSKAL-WALLIS)
    # -------------------------------------------------------------------------
    @classmethod
    def test_multiple_groups(cls, df: pd.DataFrame, num_col: str, cat_col: str) -> Optional[Dict[str, Any]]:
        """Performs One-Way ANOVA or Kruskal-Wallis across multiple categories."""
        if num_col not in df.columns or cat_col not in df.columns:
            return None
        if not pd.api.types.is_numeric_dtype(df[num_col]):
            return None

        clean_df = df[[num_col, cat_col]].dropna()
        if len(clean_df) > 30000:
            sample_df = clean_df.sample(30000, random_state=42)
        else:
            sample_df = clean_df

        grouped_dict = {k: v.values for k, v in sample_df.groupby(cat_col)[num_col] if len(v) >= 3}
        groups = list(grouped_dict.values())
        group_names = [str(k) for k in grouped_dict.keys()]

        try:
            kw_res = stats.kruskal(*groups)
            stat = float(kw_res.statistic)
            p_val = float(kw_res.pvalue)

            k = len(groups)
            N = sum(len(g) for g in groups)
            eta_sq = max(0.0, min(1.0, (stat - k + 1) / (N - k))) if (N - k) > 0 else 0.0
            effect_label = "Small" if eta_sq < 0.06 else ("Medium" if eta_sq < 0.14 else "Large")

            is_sig = p_val < 0.05
            interpretation = (
                f"Kruskal-Wallis analysis across {k} groups in '{cat_col}' demonstrates "
                f"{'a statistically significant variance' if is_sig else 'no significant variance'} "
                f"in {num_col} (H = {stat:.2f}, p = {p_val:.4g}, η² = {eta_sq:.3f} [{effect_label}]). "
                f"Cohort differences represent correlational variation rather than confirmed causal mechanisms."
            )

            return {
                "test_name": "Kruskal-Wallis Multi-Cohort Test",
                "metric": str(num_col),
                "dimension": str(cat_col),
                "groups_count": k,
                "group_names": group_names[:8],
                "statistic": round(stat, 4),
                "p_value": round(p_val, 5),
                "is_significant": is_sig,
                "effect_size_eta_sq": round(eta_sq, 4),
                "effect_label": effect_label,
                "interpretation": interpretation
            }
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # 5. CATEGORICAL INDEPENDENCE (CHI-SQUARE & CRAMER'S V)
    # -------------------------------------------------------------------------
    @classmethod
    def test_categorical_independence(cls, df: pd.DataFrame, cat1: str, cat2: str) -> Optional[Dict[str, Any]]:
        """Evaluates contingency independence via Chi-Square and Cramer's V association."""
        if cat1 not in df.columns or cat2 not in df.columns or cat1 == cat2:
            return None

        clean_df = df[[cat1, cat2]].dropna()
        if clean_df[cat1].nunique() > 20 or clean_df[cat2].nunique() > 20:
            return None

        contingency = pd.crosstab(clean_df[cat1], clean_df[cat2])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return None

        try:
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            n = contingency.sum().sum()
            min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
            cramer_v = math.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0

            assoc_label = "Weak" if cramer_v < 0.2 else ("Moderate" if cramer_v < 0.4 else "Strong")
            is_sig = p_val < 0.05

            interpretation = (
                f"Chi-Square test indicates {'a significant association' if is_sig else 'no significant association'} "
                f"between '{cat1}' and '{cat2}' (χ² = {chi2:.2f}, df = {dof}, p = {p_val:.4g}, Cramer's V = {cramer_v:.2f} [{assoc_label}]). "
                f"Cross-tabulation indicates non-random distribution across categories."
            )

            return {
                "test_name": "Pearson's Chi-Square Test of Independence",
                "var1": str(cat1),
                "var2": str(cat2),
                "chi2_statistic": round(float(chi2), 3),
                "p_value": round(float(p_val), 5),
                "degrees_of_freedom": int(dof),
                "cramers_v": round(float(cramer_v), 3),
                "association_strength": assoc_label,
                "is_significant": is_sig,
                "interpretation": interpretation
            }
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # 6. CORRELATION LATTICE (PEARSON & SPEARMAN)
    # -------------------------------------------------------------------------
    @classmethod
    def calculate_correlations(cls, df: pd.DataFrame, num_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculates Pearson and Spearman correlation matrices with non-causal disclosures."""
        cols = num_cols if num_cols is not None else list(df.select_dtypes(include=[np.number]).columns)
        cols = [c for c in cols if not any(k in str(c).lower() for k in ["_id", "id_", "identifier", "row_number"])]

        if len(cols) < 2:
            return {"columns": [], "pearson_matrix": {}, "spearman_matrix": {}, "strong_pairs": [], "disclaimer": ""}

        num_df = df[cols].dropna()
        if len(num_df) < 5:
            return {"columns": [], "pearson_matrix": {}, "spearman_matrix": {}, "strong_pairs": [], "disclaimer": ""}

        if len(num_df) > 30000:
            sample_corr = num_df.sample(30000, random_state=42)
        else:
            sample_corr = num_df

        pearson_mat = sample_corr.corr(method="pearson").round(3)
        spearman_mat = sample_corr.corr(method="spearman").round(3)

        strong_pairs: List[Dict[str, Any]] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r_p = float(pearson_mat.iloc[i, j])
                r_s = float(spearman_mat.iloc[i, j])
                if not np.isnan(r_p) and abs(r_p) >= 0.4:
                    pair_type = "Positive Association" if r_p > 0 else "Inverse Trade-off"
                    strong_pairs.append({
                        "feature1": str(cols[i]),
                        "feature2": str(cols[j]),
                        "pearson_r": round(r_p, 3),
                        "spearman_rho": round(r_s, 3),
                        "strength": "Strong" if abs(r_p) >= 0.7 else "Moderate",
                        "relationship_type": pair_type,
                        "disclaimer": f"Higher {cols[i]} tends to coincide with {'higher' if r_p > 0 else 'lower'} {cols[j]}, but does not imply causal effect."
                    })

        strong_pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

        return {
            "columns": cols,
            "pearson_matrix": pearson_mat.to_dict(),
            "spearman_matrix": spearman_mat.to_dict(),
            "strong_pairs": strong_pairs,
            "disclaimer": "CRITICAL DATA SCIENCE DIRECTIVE: Correlation reflects statistical co-occurrence and does NOT establish causation."
        }
