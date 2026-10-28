"""
core.ml.clustering
==================
Institutional Unsupervised Clustering Suite for Quantum Analyst.
Supports K-Means, DBSCAN, and Agglomerative Hierarchical Clustering with
automated optimal k evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz),
cluster profiling, and 2D PCA space projection.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


CLUSTERING_ALGORITHMS = {
    "kmeans": {
        "name": "K-Means Clustering",
        "family": "Centroid-Based",
        "description": "Partitions data into k Voronoi cells minimizing within-cluster sum-of-squares (inertia).",
    },
    "dbscan": {
        "name": "DBSCAN Clustering",
        "family": "Density-Based",
        "description": "Discovers clusters of arbitrary non-convex shapes and isolates noise points via local spatial density.",
    },
    "agglomerative": {
        "name": "Agglomerative Hierarchical",
        "family": "Connectivity-Based",
        "description": "Builds a nested bottom-up hierarchical tree using Ward minimum variance linkage.",
    },
}


def run_clustering(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    algorithm: str = "kmeans",
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
    auto_k: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes unsupervised clustering on numeric columns.
    Returns cluster labels, evaluation metrics, cluster profiles, and 2D PCA projection coordinates.
    """
    algo_key = algorithm.lower()
    if algo_key not in CLUSTERING_ALGORITHMS:
        raise ValueError(f"Unsupported clustering algorithm '{algorithm}'. Supported: {list(CLUSTERING_ALGORITHMS.keys())}")

    if not feature_cols:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(feature_cols) < 2:
        raise ValueError("Clustering requires at least 2 numeric feature columns.")

    if len(df) < 6:
        raise ValueError("Clustering requires at least 6 observations.")

    # Impute missing values and standardize features
    # Sample for large datasets — clustering is O(n²) for DBSCAN/Agglomerative
    work_df = df
    if len(df) > 30_000:
        work_df = df.sample(30_000, random_state=random_state)
    X_raw = work_df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)

    n_samples = len(df)
    optimal_k = n_clusters
    k_evaluations = []

    # Automated optimal k discovery
    if auto_k and algo_key in ("kmeans", "agglomerative"):
        max_test_k = min(8, n_samples - 1)
        best_sil = -1.0
        for test_k in range(2, max_test_k + 1):
            if algo_key == "kmeans":
                test_est = KMeans(n_clusters=test_k, random_state=random_state, n_init="auto")
            else:
                test_est = AgglomerativeClustering(n_clusters=test_k)
            test_labels = test_est.fit_predict(X_std)
            if len(set(test_labels)) > 1:
                sil = float(silhouette_score(X_std, test_labels))
                k_evaluations.append({"k": test_k, "silhouette": round(sil, 4)})
                if sil > best_sil:
                    best_sil = sil
                    optimal_k = test_k

    chosen_k = max(2, min(n_samples - 1, optimal_k))

    # Fit selected algorithm
    if algo_key == "kmeans":
        estimator = KMeans(n_clusters=chosen_k, random_state=random_state, n_init="auto")
        labels = estimator.fit_predict(X_std)

    elif algo_key == "dbscan":
        adjusted_min_samples = min(min_samples, max(2, n_samples // 4))
        estimator = DBSCAN(eps=eps, min_samples=adjusted_min_samples)
        labels = estimator.fit_predict(X_std)

    elif algo_key == "agglomerative":
        estimator = AgglomerativeClustering(n_clusters=chosen_k)
        labels = estimator.fit_predict(X_std)

    # Calculate cluster statistics
    unique_labels = sorted(list(set(labels)))
    n_clusters_found = len([lbl for lbl in unique_labels if lbl != -1])
    noise_count = int(np.sum(labels == -1))

    # Quality metrics
    non_noise_mask = labels != -1
    if len(set(labels[non_noise_mask])) > 1 and np.sum(non_noise_mask) > 3:
        try:
            sil_score = round(float(silhouette_score(X_std[non_noise_mask], labels[non_noise_mask])), 4)
        except Exception:
            sil_score = 0.0
        try:
            db_score = round(float(davies_bouldin_score(X_std[non_noise_mask], labels[non_noise_mask])), 4)
        except Exception:
            db_score = 0.0
        try:
            ch_score = round(float(calinski_harabasz_score(X_std[non_noise_mask], labels[non_noise_mask])), 2)
        except Exception:
            ch_score = 0.0
    else:
        sil_score = 0.0
        db_score = 0.0
        ch_score = 0.0

    # Cluster Profiles
    profile_df = df[feature_cols].copy()
    profile_df["Cluster"] = [f"Cluster {lbl}" if lbl != -1 else "Noise (Outlier)" for lbl in labels]
    cluster_summary = profile_df.groupby("Cluster").mean().round(3).reset_index()
    cluster_counts = profile_df["Cluster"].value_counts().to_dict()
    cluster_summary["Record Count"] = cluster_summary["Cluster"].map(cluster_counts)

    # 2D PCA Space Projection
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_std)
    proj_df = pd.DataFrame({
        "pca_x": np.round(X_pca[:, 0], 4),
        "pca_y": np.round(X_pca[:, 1], 4),
        "cluster": [f"Cluster {lbl}" if lbl != -1 else "Noise" for lbl in labels],
        "index": df.index.astype(str),
    })

    return {
        "algorithm": algo_key,
        "algorithm_name": CLUSTERING_ALGORITHMS[algo_key]["name"],
        "n_clusters": n_clusters_found,
        "labels": [int(l) for l in labels],
        "noise_count": noise_count,
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "calinski_harabasz_score": ch_score,
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "k_evaluations": k_evaluations,
        "pca_projection": proj_df.to_dict(orient="records"),
        "variance_explained_pct": round(float(np.sum(pca.explained_variance_ratio_)) * 100, 2),
        "summary": (
            f"Partitioned data into {n_clusters_found} clusters "
            f"with Silhouette Score of {sil_score} (Variance Explained: {round(float(np.sum(pca.explained_variance_ratio_)) * 100, 1)}%)."
        ),
    }
