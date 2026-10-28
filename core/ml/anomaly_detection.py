"""
core.ml.anomaly_detection
=========================
Institutional Anomaly & Outlier Detection Engine for Quantum Analyst.
Supports Isolation Forest, Local Outlier Factor (LOF), One-Class SVM,
and Elliptic Envelope with anomaly scoring, 2D PCA visualization, and profiling.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


ANOMALY_ALGORITHMS = {
    "isolation_forest": {
        "name": "Isolation Forest",
        "family": "Tree Ensemble",
        "description": "Isolates anomalies by randomly partitioning feature space; anomalies require fewer splits to isolate.",
    },
    "local_outlier_factor": {
        "name": "Local Outlier Factor (LOF)",
        "family": "Density-Based",
        "description": "Measures local density deviation of an observation relative to its k-nearest neighbors.",
    },
    "one_class_svm": {
        "name": "One-Class SVM",
        "family": "Support Vector Boundary",
        "description": "Constructs a non-linear hyper-spherical boundary in kernel feature space to enclose nominal data.",
    },
    "elliptic_envelope": {
        "name": "Elliptic Envelope",
        "family": "Robust Covariance",
        "description": "Fits a robust Mahalanobis elliptical covariance envelope, optimal for Gaussian multivariate distributions.",
    },
}


def run_anomaly_detection(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    algorithm: str = "isolation_forest",
    contamination: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes unsupervised anomaly detection on numeric columns and returns
    anomaly flags, continuous anomaly scores, PCA projection coords, and top outlier records.
    """
    algo_key = algorithm.lower()
    if algo_key not in ANOMALY_ALGORITHMS:
        raise ValueError(f"Unsupported anomaly algorithm '{algorithm}'. Supported: {list(ANOMALY_ALGORITHMS.keys())}")

    # Determine numeric feature columns
    if not feature_cols:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(feature_cols) < 2:
        raise ValueError("Anomaly detection requires at least 2 numeric feature columns.")

    if len(df) < 10:
        raise ValueError("Anomaly detection requires at least 10 observations.")

    # Impute and Standardize (sampled for large datasets to prevent OOM/slowdown)
    work_df = df
    if len(df) > 30_000:
        work_df = df.sample(30_000, random_state=random_state)
    X_raw = work_df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)

    contam = max(0.01, min(0.20, float(contamination)))

    # Fit selected anomaly estimator
    if algo_key == "isolation_forest":
        est = IsolationForest(contamination=contam, random_state=random_state, n_estimators=100, n_jobs=-1)
        raw_preds = est.fit_predict(X_std)
        raw_scores = -est.score_samples(X_std)  # Higher is more anomalous

    elif algo_key == "local_outlier_factor":
        k_neighbors = min(20, max(5, len(df) // 5))
        est = LocalOutlierFactor(n_neighbors=k_neighbors, contamination=contam, n_jobs=-1)
        raw_preds = est.fit_predict(X_std)
        raw_scores = -est.negative_outlier_factor_  # Higher is more anomalous

    elif algo_key == "one_class_svm":
        est = OneClassSVM(nu=contam, kernel="rbf", gamma="scale")
        raw_preds = est.fit_predict(X_std)
        raw_scores = -est.decision_function(X_std).flatten()

    elif algo_key == "elliptic_envelope":
        est = EllipticEnvelope(contamination=contam, random_state=random_state)
        raw_preds = est.fit_predict(X_std)
        raw_scores = -est.decision_function(X_std).flatten()

    # Normalization of anomaly scores to [0.0, 1.0]
    min_sc = np.min(raw_scores)
    max_sc = np.max(raw_scores)
    if max_sc - min_sc > 1e-9:
        norm_scores = (raw_scores - min_sc) / (max_sc - min_sc)
    else:
        norm_scores = np.zeros(len(df))

    # Binary flags (-1 is anomaly in scikit-learn, 1 is inlier)
    is_anomaly = (raw_preds == -1)
    anomaly_count = int(np.sum(is_anomaly))
    anomaly_pct = round((anomaly_count / len(df)) * 100, 2)

    # 2D PCA Projection for Visual Clustering
    pca = PCA(n_components=2, random_state=random_state)
    coords_2d = pca.fit_transform(X_std)

    # Subsample for lightweight UI rendering
    sub_step = max(1, len(df) // 600)
    indices = list(range(0, len(df), sub_step))
    # Ensure anomalies are always included in the scatter sample
    anomaly_indices = np.where(is_anomaly)[0].tolist()
    all_plot_indices = sorted(list(set(indices + anomaly_indices[:100])))

    scatter_points = []
    for idx in all_plot_indices:
        scatter_points.append({
            "dim1": round(float(coords_2d[idx, 0]), 4),
            "dim2": round(float(coords_2d[idx, 1]), 4),
            "is_anomaly": bool(is_anomaly[idx]),
            "anomaly_score": round(float(norm_scores[idx]), 4),
            "index": int(idx),
        })

    # Extract Top Anomalous Records with Feature Values
    top_indices = np.argsort(norm_scores)[::-1][:15]
    top_records = []
    for rank, idx in enumerate(top_indices, start=1):
        if not is_anomaly[idx] and rank > 5:
            break
        row_dict = {
            "rank": rank,
            "row_index": int(idx),
            "anomaly_score": round(float(norm_scores[idx]), 4),
            "status": "Anomaly" if is_anomaly[idx] else "Nominal",
        }
        # Add key features
        for f in feature_cols[:6]:
            val = df[f].iloc[idx]
            row_dict[f] = round(float(val), 2) if isinstance(val, (int, float, np.number)) else str(val)
        top_records.append(row_dict)

    summary = (
        f"Analyzed {len(df):,} records using {ANOMALY_ALGORITHMS[algo_key]['name']}. "
        f"Identified {anomaly_count:,} anomalies ({anomaly_pct}% of total records) at contamination threshold {contam:.1%}."
    )

    return {
        "algorithm": algo_key,
        "metadata": ANOMALY_ALGORITHMS[algo_key],
        "contamination": contam,
        "total_records": len(df),
        "anomaly_count": anomaly_count,
        "anomaly_pct": anomaly_pct,
        "is_anomaly": is_anomaly.tolist(),
        "anomaly_scores": [round(float(s), 4) for s in norm_scores],
        "scatter_points": scatter_points,
        "top_records": top_records,
        "summary": summary,
    }
