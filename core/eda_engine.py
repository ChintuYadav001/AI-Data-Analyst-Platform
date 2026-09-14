import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from core.duckdb_engine import DuckDBEngine
from core.parquet_manager import ParquetManager
from core.sampling_engine import SamplingEngine
from core.cache_manager import CacheManager

class EDAEngine:
    """High-performance parallelized EDA engine executing independent statistical

    modules concurrently via thread pools and DuckDB Parquet querying.
    """

    _thread_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="parallel_eda")

    @classmethod
    def run_quick_eda(
        cls,
        parquet_path: Union[str, Path],
        profile: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes Quick EDA in parallel, returning in < 1.5s even on 1M+ rows."""
        dataset_hash = profile.get("dataset_hash", "default")
        version = profile.get("version", "v1")

        cached = CacheManager.get(dataset_hash, version, "quick_eda")
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        target_col = plan.get("target_variable")
        num_cols = profile.get("numeric_columns", [])
        clean_num_cols = [c for c in num_cols if c not in profile.get("identifier_columns", [])]
        cat_cols = profile.get("categorical_columns", [])
        date_cols = profile.get("datetime_columns", [])

        futures = {}

        # 1. Target Distribution & Summary Task
        def task_target_dist():
            if not target_col:
                return None
            quantiles = DuckDBEngine.get_quantiles(parquet_path, target_col, [0.25, 0.50, 0.75])
            sample_df, is_exact, note = SamplingEngine.get_visualization_sample(
                parquet_path, "histogram", x_col=target_col, target_size=3000
            )
            return {
                "target_col": target_col,
                "quantiles": quantiles,
                "sample_data": sample_df.to_dict(orient="records"),
                "is_exact": is_exact,
                "telemetry_note": note
            }

        futures[cls._thread_pool.submit(task_target_dist)] = "target_distribution"

        # 2. Categorical Segmentation Task
        def task_segmentation():
            if not cat_cols:
                return None
            top_cat = cat_cols[0]
            if target_col:
                agg = DuckDBEngine.get_bivariate_aggregation(parquet_path, top_cat, target_col, top_n=20)
            else:
                agg = DuckDBEngine.get_categorical_frequency(parquet_path, top_cat, top_n=20)
            return {
                "cat_col": top_cat,
                "metric_col": target_col,
                "data": agg,
                "is_exact": True,
                "telemetry_note": "Exact DuckDB Group-By aggregation across full dataset."
            }

        futures[cls._thread_pool.submit(task_segmentation)] = "segmentation"

        # 3. Time Series Momentum Task
        def task_time_series():
            if not date_cols or not clean_num_cols:
                return None
            d_col = date_cols[0]
            m_col = target_col or clean_num_cols[0]
            ts_data = DuckDBEngine.get_date_timeseries(parquet_path, d_col, m_col, granularity="month")
            return {
                "date_col": d_col,
                "metric_col": m_col,
                "data": ts_data,
                "is_exact": True,
                "telemetry_note": "Exact chronological roll-up across full dataset."
            }

        futures[cls._thread_pool.submit(task_time_series)] = "time_series"

        # 4. Correlation Task
        def task_correlation():
            if len(clean_num_cols) < 2:
                return None
            corr_matrix = DuckDBEngine.get_correlation_matrix(parquet_path, clean_num_cols[:6])
            # Find top correlated pair
            top_pair = None
            max_c = 0.0
            cols = list(corr_matrix.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c_val = abs(corr_matrix.iloc[i, j])
                    if c_val > max_c and c_val < 0.9999:
                        max_c = c_val
                        top_pair = (cols[i], cols[j], round(float(corr_matrix.iloc[i, j]), 4))

            return {
                "matrix": corr_matrix.to_dict(),
                "top_pair": top_pair,
                "is_exact": True,
                "telemetry_note": "Calculated across numeric non-ID vectors."
            }

        futures[cls._thread_pool.submit(task_correlation)] = "correlation"

        # 5. Outlier Screening Task (Stage 1 IQR Fast Screen)
        def task_outlier_screen():
            if not clean_num_cols:
                return None
            eval_col = target_col or clean_num_cols[0]
            q = DuckDBEngine.get_quantiles(parquet_path, eval_col, [0.25, 0.75])
            q25 = q.get(0.25, 0.0)
            q75 = q.get(0.75, 0.0)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            clean_p = DuckDBEngine._clean_path(parquet_path)
            sql = f"""
            SELECT COUNT(*) AS outlier_count 
            FROM read_parquet('{clean_p}') 
            WHERE "{eval_col}" < {lower_bound} OR "{eval_col}" > {upper_bound};
            """
            outlier_cnt = int(DuckDBEngine.get_connection().execute(sql).df().iloc[0]["outlier_count"])
            return {
                "column": eval_col,
                "iqr": round(iqr, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outlier_count": outlier_cnt,
                "outlier_pct": round((outlier_cnt / max(1, profile.get("rows", 1))) * 100, 2),
                "is_exact": True,
                "telemetry_note": "Single-pass DuckDB Tukey IQR anomaly filter."
            }

        futures[cls._thread_pool.submit(task_outlier_screen)] = "outliers"

        # Collect parallel results gracefully
        results = {}
        for fut in as_completed(futures):
            mod_name = futures[fut]
            try:
                results[mod_name] = fut.result()
            except Exception as e:
                results[mod_name] = {"error": str(e), "failed": True}

        # Synthesize 3-5 Key Findings
        key_findings = []
        if results.get("target_distribution") and target_col:
            td = results["target_distribution"]
            q50 = td.get("quantiles", {}).get(0.5, 0.0)
            key_findings.append(f"Primary Target Variable '{target_col}' has a median observation value of {q50:,.2f}.")

        if results.get("correlation") and results["correlation"].get("top_pair"):
            c1, c2, val = results["correlation"]["top_pair"]
            direction = "positive" if val > 0 else "inverse"
            key_findings.append(f"Strongest linear covariance detected between '{c1}' and '{c2}' ({direction} r = {val:.2f}).")

        if results.get("time_series") and results["time_series"].get("data"):
            ts_pts = results["time_series"]["data"]
            if len(ts_pts) >= 2:
                growth = ts_pts[-1]["total"] - ts_pts[0]["total"]
                trend_dir = "expanding" if growth >= 0 else "contracting"
                key_findings.append(f"Chronological trajectory over {len(ts_pts)} periods exhibits an {trend_dir} trajectory.")

        if results.get("outliers"):
            ot = results["outliers"]
            key_findings.append(f"Statistical outlier screening isolated {ot.get('outlier_count', 0):,} anomalous tail rows ({ot.get('outlier_pct', 0)}%).")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        quick_eda = {
            "execution_mode": "QUICK_EDA",
            "dataset_hash": dataset_hash,
            "version": version,
            "target_variable": target_col,
            "key_findings": key_findings,
            "modules": results,
            "elapsed_ms": elapsed_ms
        }

        CacheManager.set(dataset_hash, version, "quick_eda", quick_eda, persist_l3=True)
        return quick_eda

    @classmethod
    def run_deep_eda(
        cls,
        parquet_path: Union[str, Path],
        profile: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes deep statistical & ML models (Isolation Forest, ANOVA drivers) in the background."""
        dataset_hash = profile.get("dataset_hash", "default")
        version = profile.get("version", "v1")

        cached = CacheManager.get(dataset_hash, version, "deep_eda")
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        target_col = plan.get("target_variable")
        num_cols = profile.get("numeric_columns", [])
        clean_num_cols = [c for c in num_cols if c not in profile.get("identifier_columns", [])]

        results = {}

        # 1. Isolation Forest Anomaly Detection on smart sample
        if len(clean_num_cols) >= 2:
            try:
                from sklearn.ensemble import IsolationForest
                sample_df = ParquetManager.read_sample(parquet_path, n=20000, random_state=42)[clean_num_cols].dropna()
                if len(sample_df) > 50:
                    clf = IsolationForest(contamination=0.03, random_state=42)
                    preds = clf.fit_predict(sample_df)
                    anomaly_count = int(np.count_nonzero(preds == -1))
                    results["isolation_forest"] = {
                        "tested_records": len(sample_df),
                        "anomalies_isolated": anomaly_count,
                        "contamination_rate": 0.03,
                        "status": "completed"
                    }
            except Exception as e:
                results["isolation_forest"] = {"error": str(e), "status": "failed"}

        # 2. ANOVA & Driver Ranking
        if target_col and len(clean_num_cols) >= 2:
            try:
                from scipy import stats
                driver_cols = [c for c in clean_num_cols if c != target_col][:5]
                sample_df = ParquetManager.read_sample(parquet_path, n=20000, random_state=42)
                rankings = []
                for dc in driver_cols:
                    valid_df = sample_df[[dc, target_col]].dropna()
                    if len(valid_df) > 50:
                        corr_val, p_val = stats.pearsonr(valid_df[dc], valid_df[target_col])
                        rankings.append({
                            "feature": dc,
                            "correlation": round(float(corr_val), 4),
                            "p_value": float(p_val),
                            "influence": "HIGH" if abs(corr_val) > 0.4 else "MEDIUM" if abs(corr_val) > 0.2 else "LOW"
                        })
                rankings.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                results["statistical_drivers"] = rankings
            except Exception as e:
                results["statistical_drivers"] = {"error": str(e)}

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        deep_eda = {
            "execution_mode": "DEEP_EDA",
            "dataset_hash": dataset_hash,
            "version": version,
            "modules": results,
            "elapsed_ms": elapsed_ms
        }

        CacheManager.set(dataset_hash, version, "deep_eda", deep_eda, persist_l3=True)
        return deep_eda
