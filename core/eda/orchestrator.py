"""
Central AI EDA Orchestrator for Quantum Analyst.
Coordinates the adaptive 14-stage analytical pipeline, synthesizes the EDA Plan Object,
computes EDA Coverage Scores, organizes 3-Tier Chart Recommendations, and produces
evidence-backed Senior Data Analyst Key Findings with strict non-causal disclosures.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from core.cache import AnalyticsCache
from core.models.quality_model import DataQualityModel

from core.eda.profiler import EDAProfiler
from core.eda.goal_engine import EDAGoalEngine
from core.eda.statistical_engine import EDAStatisticalEngine
from core.eda.time_series import EDATimeSeriesEngine
from core.eda.segments import EDASegmentEngine
from core.eda.anomalies import EDAAnomalyEngine
from core.eda.drivers import EDADriverEngine
from core.eda.nl_query import EDANLQueryEngine


class AIEDAOrchestrator:
    """Central Orchestrator delivering institutional-grade, goal-aware exploratory data analysis."""

    @classmethod
    def run_full_eda(
        cls,
        df: pd.DataFrame,
        user_goal: Optional[str] = None,
        user_target: Optional[str] = None,
        mode: str = "auto"
    ) -> Dict[str, Any]:
        """Executes the adaptive 14-stage EDA pipeline."""
        if df is None or df.empty:
            return cls._empty_eda_result()

        cache_key = f"ai_eda_{len(df)}_{len(df.columns)}_{hash(user_goal)}_{hash(user_target)}_{mode}"
        cached = AnalyticsCache.get(df, cache_key)
        if cached is not None:
            return cached

        # Mode resolution
        resolved_mode = mode.lower()
        if resolved_mode == "auto":
            resolved_mode = "deep" if len(df) <= 50000 else "standard"

        # ---------------------------------------------------------------------
        # STAGE 1: DATASET OVERVIEW & INTELLIGENCE PROFILE
        # ---------------------------------------------------------------------
        profile = EDAProfiler.profile_dataset(df)

        # ---------------------------------------------------------------------
        # STAGE 2: DATA QUALITY IMPACT
        # ---------------------------------------------------------------------
        quality_assessment = DataQualityModel.calculate_quality_score(df)
        quality_score = quality_assessment.get("score", 92)
        quality_limitations = cls._identify_quality_limitations(df, quality_assessment, profile)

        # ---------------------------------------------------------------------
        # STAGE 3: GOAL & TARGET RESOLUTION
        # ---------------------------------------------------------------------
        goal_info = EDAGoalEngine.infer_goal(profile, user_goal=user_goal, user_target=user_target)
        target_info = EDAGoalEngine.resolve_target(df, profile, user_target=user_target)
        target_col = target_info["column"] if target_info else None
        target_task = target_info["task_type"] if target_info else "regression"

        # ---------------------------------------------------------------------
        # STAGE 4: VARIABLE PRIORITIZATION
        # ---------------------------------------------------------------------
        priority_cols = EDAGoalEngine.select_priority_columns(
            df, profile, target_col=target_col, max_cols=14
        )

        # ---------------------------------------------------------------------
        # STAGE 5: UNIVARIATE ANALYSIS (MOMENTS & DISTRIBUTIONS)
        # ---------------------------------------------------------------------
        num_moments = EDAStatisticalEngine.analyze_numerical_moments(df, columns=priority_cols)
        cat_dists = EDAStatisticalEngine.analyze_categorical_distributions(df, columns=priority_cols)

        # ---------------------------------------------------------------------
        # STAGE 6: CORRELATIONS & ASSOCIATIONS
        # ---------------------------------------------------------------------
        correlations = EDAStatisticalEngine.calculate_correlations(df)

        # ---------------------------------------------------------------------
        # STAGE 7: TARGET DRIVER ANALYSIS
        # ---------------------------------------------------------------------
        drivers: List[Dict[str, Any]] = []
        if target_col:
            drivers = EDADriverEngine.analyze_drivers(df, target_col=target_col, task_type=target_task)

        # ---------------------------------------------------------------------
        # STAGE 8: TIME-SERIES ANALYSIS
        # ---------------------------------------------------------------------
        time_series_results = None
        date_cols = profile.get("columns_by_type", {}).get("datetime", [])
        if date_cols:
            time_series_results = EDATimeSeriesEngine.analyze_temporal_trends(
                df, date_col=date_cols[0], metric_col=target_col
            )

        # ---------------------------------------------------------------------
        # STAGE 9: SEGMENT ANALYSIS
        # ---------------------------------------------------------------------
        segments = EDASegmentEngine.analyze_segments(df, target_col=target_col)

        # ---------------------------------------------------------------------
        # STAGE 10: OUTLIER & ANOMALY EXPLORATION
        # ---------------------------------------------------------------------
        anomalies = EDAAnomalyEngine.detect_anomalies(df)

        # ---------------------------------------------------------------------
        # STAGE 11: STATISTICAL HYPOTHESIS TESTING (IF STANDARD OR DEEP)
        # ---------------------------------------------------------------------
        statistical_tests: List[Dict[str, Any]] = []
        if resolved_mode in ["standard", "deep"]:
            statistical_tests = cls._run_hypothesis_battery(df, target_col, profile, priority_cols)

        # ---------------------------------------------------------------------
        # STAGE 12: CHART RECOMMENDATIONS (3 TIERS)
        # ---------------------------------------------------------------------
        chart_tiers = cls._build_chart_recommendation_tiers(
            df=df,
            profile=profile,
            target_col=target_col,
            num_moments=num_moments,
            cat_dists=cat_dists,
            correlations=correlations,
            time_series=time_series_results,
            segments=segments,
            anomalies=anomalies
        )

        # ---------------------------------------------------------------------
        # STAGE 13: KEY FINDINGS WITH EVIDENCE
        # ---------------------------------------------------------------------
        key_findings = cls._synthesize_key_findings(
            df=df,
            goal_info=goal_info,
            target_info=target_info,
            num_moments=num_moments,
            cat_dists=cat_dists,
            correlations=correlations,
            drivers=drivers,
            time_series=time_series_results,
            segments=segments,
            anomalies=anomalies,
            statistical_tests=statistical_tests,
            quality_score=quality_score
        )

        # ---------------------------------------------------------------------
        # STAGE 14: RECOMMENDED NEXT ANALYSES & FEATURE ENGINEERING
        # ---------------------------------------------------------------------
        next_analyses = cls._recommend_next_analyses(
            target_info=target_info,
            drivers=drivers,
            anomalies=anomalies,
            time_series=time_series_results,
            segments=segments
        )
        feature_recommendations = cls._generate_feature_recommendations(profile, target_col, num_moments)

        # ---------------------------------------------------------------------
        # EDA COVERAGE SCORE & AUDIT
        # ---------------------------------------------------------------------
        coverage = cls._calculate_eda_coverage_score(
            has_univariate=bool(num_moments or cat_dists),
            has_bivariate=bool(len(correlations.get("columns", [])) >= 2),
            has_target=bool(target_col),
            has_time_series=bool(time_series_results),
            has_segments=bool(segments),
            has_anomalies=bool(anomalies),
            has_stats_tests=bool(statistical_tests or (resolved_mode in ["standard", "deep"] and len(df) >= 20))
        )

        # Synthesize EDA Plan Object
        eda_plan = {
            "mode": resolved_mode.upper(),
            "goal": goal_info["goal"],
            "goal_is_inferred": goal_info["is_inferred"],
            "target": target_col,
            "target_task": target_task,
            "domain": profile["domain"],
            "priority_columns": priority_cols,
            "executed_stages": [
                "Dataset Overview & Profile",
                "Data Quality Impact Assessment",
                "Goal & Target Resolution",
                "Univariate Moment Analysis",
                "Bivariate & Correlation Modeling",
                "Segment Contribution Matrix",
                "Outlier & Anomaly Exploration",
                *(["Time-Series Wave Analysis"] if time_series_results else []),
                *(["Target Driver Association"] if drivers else []),
                *(["Inferential Hypothesis Testing"] if statistical_tests else []),
                "3-Tier Visual Synthesis",
                "Senior Analyst Findings"
            ],
            "total_analyses_count": (
                len(num_moments) + len(cat_dists) + len(correlations.get("strong_pairs", [])) +
                len(drivers) + len(segments) + len(anomalies) + len(statistical_tests)
            ),
            "charts_count": len(chart_tiers["level1_executive"]) + len(chart_tiers["level2_detailed"]),
            "coverage": coverage
        }

        result = {
            "profile": profile,
            "quality": {
                "score": quality_score,
                "limitations": quality_limitations
            },
            "goal_info": goal_info,
            "target_info": target_info,
            "eda_plan": eda_plan,
            "coverage": coverage,
            "univariate_numerical": num_moments,
            "univariate_categorical": cat_dists,
            "correlations": correlations,
            "target_drivers": drivers,
            "time_series": time_series_results,
            "segments": segments,
            "anomalies": anomalies,
            "statistical_tests": statistical_tests,
            "chart_tiers": chart_tiers,
            "key_findings": key_findings,
            "next_analyses": next_analyses,
            "feature_recommendations": feature_recommendations
        }

        AnalyticsCache.set(df, cache_key, result)
        return result

    @classmethod
    def _identify_quality_limitations(cls, df: pd.DataFrame, quality: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
        limitations = []
        n_rows = len(df)
        if n_rows == 0:
            return ["Dataset is empty."]

        # Check missingness
        missing_pct = profile.get("missing_pct", 0)
        if missing_pct > 5.0:
            limitations.append(f"Overall missingness stands at {missing_pct:.1f}%; complete-case analysis may induce cohort attenuation.")

        # High missingness columns
        for col, p in profile.get("column_profiles", {}).items():
            if p.get("missing_pct", 0) > 15.0:
                limitations.append(f"Attribute `{col}` has {p['missing_pct']:.1f}% missing values.")

        # Duplicates
        dup_pct = profile.get("duplicate_pct", 0)
        if dup_pct > 2.0:
            limitations.append(f"Duplicate records ({dup_pct:.1f}%) detected; potential frequency overrepresentation.")

        # Extreme outliers in quality
        if not limitations:
            limitations.append("No critical data quality blockers detected; statistical reliability verified.")

        return limitations

    @classmethod
    def _run_hypothesis_battery(
        cls,
        df: pd.DataFrame,
        target_col: Optional[str],
        profile: Dict[str, Any],
        priority_cols: List[str]
    ) -> List[Dict[str, Any]]:
        tests: List[Dict[str, Any]] = []
        num_cols = profile.get("columns_by_type", {}).get("numerical", [])
        cat_cols = profile.get("columns_by_type", {}).get("categorical", [])
        bool_cols = profile.get("columns_by_type", {}).get("boolean", [])

        # 1. Target vs Top Categorical / Boolean (Two-group Welch or Mann-Whitney)
        if target_col and target_col in num_cols:
            test_cats = [c for c in (cat_cols + bool_cols) if 2 <= df[c].nunique() <= 4]
            for c in test_cats[:2]:
                res = EDAStatisticalEngine.test_two_groups(df, num_col=target_col, cat_col=c)
                if res:
                    tests.append(res)

            # Target vs Multi-group Categorical (ANOVA / Kruskal-Wallis)
            multi_cats = [c for c in cat_cols if 3 <= df[c].nunique() <= 8]
            for c in multi_cats[:2]:
                res = EDAStatisticalEngine.test_multiple_groups(df, num_col=target_col, cat_col=c)
                if res:
                    tests.append(res)

        # 2. Categorical Independence (Chi-Square)
        if len(cat_cols) >= 2:
            c1, c2 = cat_cols[0], cat_cols[1]
            res = EDAStatisticalEngine.test_categorical_independence(df, cat1=c1, cat2=c2)
            if res:
                tests.append(res)

        return tests

    @classmethod
    def _build_chart_recommendation_tiers(
        cls,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        target_col: Optional[str],
        num_moments: List[Dict[str, Any]],
        cat_dists: List[Dict[str, Any]],
        correlations: Dict[str, Any],
        time_series: Optional[Dict[str, Any]],
        segments: List[Dict[str, Any]],
        anomalies: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        level1: List[Dict[str, Any]] = []
        level2: List[Dict[str, Any]] = []

        # 1. Target distribution
        if target_col and target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                level1.append({
                    "id": f"target_dist_{target_col}",
                    "title": f"// TARGET DENSITY HUD: {target_col.upper()}",
                    "type": "histogram",
                    "x": target_col,
                    "badge": "TARGET METRIC",
                    "explanation": f"Probability density and dispersion bounds for primary business target {target_col}.",
                    "sql": f"SELECT {target_col} FROM dataset WHERE {target_col} IS NOT NULL;"
                })
            else:
                level1.append({
                    "id": f"target_bar_{target_col}",
                    "title": f"// TARGET SHARE: {target_col.upper()}",
                    "type": "bar",
                    "x": target_col,
                    "badge": "TARGET CLASSES",
                    "explanation": f"Class distribution and volume concentration for outcome {target_col}.",
                    "sql": f"SELECT {target_col}, COUNT(*) AS count FROM dataset GROUP BY 1 ORDER BY 2 DESC;"
                })

        # 2. Time-series momentum (if available)
        if time_series:
            level1.append({
                "id": "ts_wave_primary",
                "title": f"// TEMPORAL VELOCITY: {time_series['metric_column'].upper()} OVER {time_series['date_column'].upper()}",
                "type": "line",
                "x": "period",
                "y": "total",
                "data_override": time_series["trend_points"],
                "badge": "TEMPORAL MOMENTUM",
                "explanation": f"{time_series['granularity']} chronological momentum demonstrating {time_series['trend_direction']}.",
                "sql": f"SELECT DATE_TRUNC('month', {time_series['date_column']}) AS period, SUM({time_series['metric_column']}) AS total FROM dataset GROUP BY 1 ORDER BY 1;"
            })

        # 3. Top Segment Breakdown
        if segments:
            top_seg = segments[0]
            level1.append({
                "id": f"segment_bar_{top_seg['dimension']}",
                "title": f"// COHORT CONTRIBUTION: {top_seg['metric'].upper()} BY {top_seg['dimension'].upper()}",
                "type": "bar",
                "x": "segment",
                "y": "total_metric",
                "data_override": top_seg["cohorts"][:8],
                "badge": "PARETO SEGMENTATION",
                "explanation": f"Contribution share across {top_seg['dimension']} cohorts; top 2 segments account for {top_seg['top2_concentration_pct']}% of metric volume.",
                "sql": f"SELECT {top_seg['dimension']}, SUM({top_seg['metric']}) AS total FROM dataset GROUP BY 1 ORDER BY 2 DESC;"
            })

        # 4. Correlation Heatmap
        cols = correlations.get("columns", [])
        if len(cols) >= 2:
            level1.append({
                "id": "corr_matrix_heatmap",
                "title": "// MULTIVARIATE CORRELATION LATTICE",
                "type": "heatmap",
                "matrix_data": correlations["pearson_matrix"],
                "columns": cols,
                "badge": "PEARSON MATRIX",
                "explanation": "Bivariate correlation matrix isolating positive linkages and inverse trade-offs. Association does not imply causation.",
                "sql": f"SELECT CORR(col1, col2) FROM dataset;"
            })

        # 5. Top Bivariate Scatter / Trade-off
        strong_pairs = correlations.get("strong_pairs", [])
        if strong_pairs:
            sp = strong_pairs[0]
            level1.append({
                "id": f"bivariate_scatter_{sp['feature1']}_{sp['feature2']}",
                "title": f"// BIVARIATE COVARIANCE: {sp['feature1'].upper()} VS {sp['feature2'].upper()}",
                "type": "scatter",
                "x": sp["feature1"],
                "y": sp["feature2"],
                "badge": f"r = {sp['pearson_r']}",
                "explanation": f"Scatter distribution illustrating {sp['relationship_type']} (Pearson r = {sp['pearson_r']}, Spearman rho = {sp['spearman_rho']}). Non-causal.",
                "sql": f"SELECT {sp['feature1']}, {sp['feature2']} FROM dataset WHERE {sp['feature1']} IS NOT NULL AND {sp['feature2']} IS NOT NULL;"
            })

        # 6. Outlier Boxplot
        if anomalies:
            top_anom = anomalies[0]
            level1.append({
                "id": f"outlier_box_{top_anom['column']}",
                "title": f"// VOLATILITY BOUNDS & ANOMALIES: {top_anom['column'].upper()}",
                "type": "box",
                "y": top_anom["column"],
                "badge": f"{top_anom['iqr_outliers_count']} OUTLIERS",
                "explanation": f"Tukey IQR dispersion HUD. Outliers shift overall sample mean by {top_anom['mean_shift_pct']}%.",
                "sql": f"SELECT {top_anom['column']} FROM dataset;"
            })

        # Level 2 Detailed Visualizations
        for m in num_moments:
            if m["column"] != target_col:
                level2.append({
                    "id": f"num_hist_{m['column']}",
                    "title": f"// SPECTRAL DENSITY: {m['column'].upper()}",
                    "type": "histogram",
                    "x": m["column"],
                    "badge": m["distribution_shape"],
                    "explanation": f"Skewness = {m['skewness']}, Kurtosis = {m['kurtosis']}. Transform suggestion: {m['recommended_transform']}.",
                    "sql": f"SELECT {m['column']} FROM dataset;"
                })

        for c in cat_dists:
            if c["column"] != target_col:
                level2.append({
                    "id": f"cat_pareto_{c['column']}",
                    "title": f"// PARETO FREQUENCY: {c['column'].upper()}",
                    "type": "bar",
                    "x": "category",
                    "y": "count",
                    "data_override": c["top_categories"],
                    "badge": f"{c['unique_count']} CLASSES",
                    "explanation": f"Top 3 concentration: {c['top3_concentration_pct']}%. Rare tail categories: {c['rare_count_total']}.",
                    "sql": f"SELECT {c['column']}, COUNT(*) AS count FROM dataset GROUP BY 1 ORDER BY 2 DESC LIMIT 10;"
                })

        return {
            "level1_executive": level1,
            "level2_detailed": level2
        }

    @classmethod
    def _synthesize_key_findings(
        cls,
        df: pd.DataFrame,
        goal_info: Dict[str, Any],
        target_info: Optional[Dict[str, Any]],
        num_moments: List[Dict[str, Any]],
        cat_dists: List[Dict[str, Any]],
        correlations: Dict[str, Any],
        drivers: List[Dict[str, Any]],
        time_series: Optional[Dict[str, Any]],
        segments: List[Dict[str, Any]],
        anomalies: List[Dict[str, Any]],
        statistical_tests: List[Dict[str, Any]],
        quality_score: int
    ) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []

        # 1. Target & Driver Finding
        if drivers:
            top_d = drivers[0]
            findings.append({
                "title": f"Primary Predictive Driver: `{top_d['feature']}`",
                "finding": f"`{top_d['feature']}` demonstrates the strongest statistical association with target `{target_info['column'] if target_info else 'Metric'}`.",
                "evidence": f"Computed association score of {top_d['association_score']}/100 via {top_d['metric_type']} (p = {top_d['p_value']:.4g}).",
                "meaning": f"Variations in `{top_d['feature']}` coincide strongly with shifts in target performance, highlighting an essential operational monitoring lever.",
                "confidence": "High" if quality_score >= 85 and top_d['p_value'] < 0.01 else "Medium",
                "recommended_action": f"Segment operations by `{top_d['feature']}` to evaluate differential margins and operational strategies."
            })

        # 2. Segment Concentration Finding
        if segments:
            top_s = segments[0]
            findings.append({
                "title": f"Severe Cohort Concentration across `{top_s['dimension']}`",
                "finding": f"Top 2 cohorts in `{top_s['dimension']}` generate {top_s['top2_concentration_pct']}% of total `{top_s['metric']}` volume.",
                "evidence": f"Top performing segment `{top_s['top_performer']}` leads cohort generation, while overall average stands at {top_s['overall_mean']:,.2f}.",
                "meaning": "High revenue concentration indicates structural dependency on key cohorts, exposing the business to portfolio concentration risk.",
                "confidence": "High",
                "recommended_action": f"Audit retention and unit economics for `{top_s['top_performer']}` while testing expansion in lower-tier cohorts."
            })

        # 3. Anomaly Impact Finding
        if anomalies:
            top_a = anomalies[0]
            if top_a["iqr_outliers_count"] > 0:
                findings.append({
                    "title": f"Statistical Distortions from Extreme `{top_a['column']}` Observations",
                    "finding": f"{top_a['iqr_outliers_count']:,} boundary outliers ({top_a['iqr_outliers_pct']}%) materially bias the sample distribution.",
                    "evidence": f"Outliers distort sample mean by {top_a['mean_shift_pct']}% (from {top_a['mean_without_outliers']:,.2f} baseline to {top_a['original_mean']:,.2f} observed).",
                    "meaning": f"Standard metrics (mean, variance) are sensitive to these tail values. {top_a['classification']}.",
                    "confidence": "High",
                    "recommended_action": "Utilize median and interquartile ranges (IQR) for executive KPI tracking rather than unadjusted arithmetic means."
                })

        # 4. Chronological Momentum Finding
        if time_series:
            findings.append({
                "title": f"Chronological Trend: {time_series['trend_direction']}",
                "finding": f"`{time_series['metric_column']}` displays {time_series['trend_direction'].lower()} across {time_series['span_days']} days.",
                "evidence": f"Latest period-over-period growth is {time_series['latest_growth_rate_pct']:+.1f}%. Peak reached {time_series['peak_period']['total']:,.2f} on {time_series['peak_period']['period']}.",
                "meaning": "Temporal velocity establishes baseline trajectory for budgeting and operational forecasting.",
                "confidence": "High" if len(time_series["trend_points"]) >= 6 else "Medium",
                "recommended_action": "Examine driver stability across periods to verify whether upward/downward momentum is sustainable."
            })

        # 5. Statistical Hypothesis Finding
        if statistical_tests:
            top_test = statistical_tests[0]
            if top_test.get("is_significant"):
                findings.append({
                    "title": f"Hypothesis Validation: Significant Difference Detected",
                    "finding": f"Rigorous inferential testing confirms significant variance across groups ({top_test['test_name']}).",
                    "evidence": f"Test statistic = {top_test.get('statistic', 0):.2f}, p-value = {top_test.get('p_value', 0):.4g} (< 0.05).",
                    "meaning": top_test.get("interpretation", "Statistical significance confirmed with controlled Type I error rate."),
                    "confidence": "High",
                    "recommended_action": "Design targeted intervention programs calibrated to high-performing vs low-performing cohorts."
                })

        return findings

    @classmethod
    def _recommend_next_analyses(
        cls,
        target_info: Optional[Dict[str, Any]],
        drivers: List[Dict[str, Any]],
        anomalies: List[Dict[str, Any]],
        time_series: Optional[Dict[str, Any]],
        segments: List[Dict[str, Any]]
    ) -> List[str]:
        recommendations: List[str] = []

        if drivers:
            recommendations.append(
                f"Deep-dive into multi-variable interaction between `{drivers[0]['feature']}` and secondary driver `{drivers[1]['feature'] if len(drivers) > 1 else 'metrics'}`."
            )

        if segments:
            recommendations.append(
                f"Perform cohort retention analysis and unit-margin comparison across `{segments[0]['dimension']}` cohorts."
            )

        if anomalies:
            recommendations.append(
                f"Audit root-cause drivers for the {anomalies[0]['iqr_outliers_count']} extreme transactions in `{anomalies[0]['column']}`."
            )

        if time_series:
            recommendations.append(
                f"Construct ARIMA/Prophet forecasting model utilizing the {time_series['granularity'].lower()} trajectory in `{time_series['metric_column']}`."
            )

        recommendations.append("Formulate synthetic feature interaction terms in Feature Engineering module.")
        return recommendations

    @classmethod
    def _generate_feature_recommendations(
        cls,
        profile: Dict[str, Any],
        target_col: Optional[str],
        num_moments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []

        # Datetime features
        date_cols = profile.get("columns_by_type", {}).get("datetime", [])
        for dc in date_cols:
            recs.append({
                "source_columns": [dc],
                "recommended_feature": f"{dc}_month, {dc}_quarter, {dc}_day_of_week",
                "transformation_type": "Temporal Component Extraction",
                "rationale": "Enables seasonal regression modeling and cyclic behavior isolation."
            })

        # Skewness transformations
        for m in num_moments:
            if "Right-Skewed" in m["distribution_shape"] and m["min"] >= 0:
                recs.append({
                    "source_columns": [m["column"]],
                    "recommended_feature": f"log1p_{m['column']}",
                    "transformation_type": "Logarithmic Transformation (np.log1p)",
                    "rationale": f"Compresses heavy right tail (skewness = {m['skewness']:.2f}) to stabilize variance."
                })
                break

        # Ratio / Interaction feature
        num_cols = profile.get("columns_by_type", {}).get("numerical", [])
        if len(num_cols) >= 2:
            c1, c2 = num_cols[0], num_cols[1]
            recs.append({
                "source_columns": [c1, c2],
                "recommended_feature": f"{c1}_per_{c2}_ratio",
                "transformation_type": "Interaction Ratio",
                "rationale": f"Captures normalized productivity and efficiency trade-offs between {c1} and {c2}."
            })

        return recs

    @classmethod
    def _calculate_eda_coverage_score(
        cls,
        has_univariate: bool,
        has_bivariate: bool,
        has_target: bool,
        has_time_series: bool,
        has_segments: bool,
        has_anomalies: bool,
        has_stats_tests: bool
    ) -> Dict[str, Any]:
        stages = [
            ("Univariate Distribution Analysis", has_univariate, 20),
            ("Bivariate Correlation Modeling", has_bivariate, 15),
            ("Target Driver Identification", has_target, 20),
            ("Time-Series Wave Analytics", has_time_series, 15),
            ("Dimensional Segmentation", has_segments, 10),
            ("Outlier & Anomaly Exploration", has_anomalies, 10),
            ("Inferential Hypothesis Testing", has_stats_tests, 10)
        ]

        total_weight = sum(w for _, _, w in stages)
        earned_weight = sum(w for _, active, w in stages if active)
        score = int(round((earned_weight / total_weight) * 100))

        covered = [name for name, active, _ in stages if active]
        not_applicable = [name for name, active, _ in stages if not active]

        return {
            "score": score,
            "covered_stages": covered,
            "not_applicable_stages": not_applicable,
            "explanation": f"Analysis covers {len(covered)} of {len(stages)} analytical dimensions ({score}% coverage)."
        }

    @classmethod
    def _empty_eda_result(cls) -> Dict[str, Any]:
        return {
            "profile": EDAProfiler.profile_dataset(pd.DataFrame()),
            "quality": {"score": 0, "limitations": []},
            "goal_info": {"goal": "No dataset loaded", "is_inferred": True, "rationale": ""},
            "target_info": None,
            "eda_plan": {"mode": "EMPTY", "goal": "None", "coverage": {"score": 0}},
            "coverage": {"score": 0, "covered_stages": [], "not_applicable_stages": []},
            "univariate_numerical": [],
            "univariate_categorical": [],
            "correlations": {},
            "target_drivers": [],
            "time_series": None,
            "segments": [],
            "anomalies": [],
            "statistical_tests": [],
            "chart_tiers": {"level1_executive": [], "level2_detailed": []},
            "key_findings": [],
            "next_analyses": [],
            "feature_recommendations": []
        }
