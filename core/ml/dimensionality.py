"""
core.ml.dimensionality
======================
Institutional Dimensionality Reduction & Decomposition Suite for Quantum Analyst.
Supports PCA, Truncated SVD, Kernel PCA, Fast ICA, NMF, and t-SNE with scree plots,
cumulative explained variance, 2D/3D projections, loadings, and feature comparison.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, FastICA, NMF
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


METHOD_METADATA = {
    "pca": {
        "name": "Principal Component Analysis (PCA)",
        "type": "Linear Orthogonal Projection",
        "supports_variance": True,
        "supports_inverse": True,
        "description": "Transforms correlated features into linearly uncorrelated principal components maximizing captured variance.",
    },
    "truncated_svd": {
        "name": "Truncated SVD (LSA)",
        "type": "Matrix Factorization",
        "supports_variance": True,
        "supports_inverse": True,
        "description": "Factorizes data without explicit centering, ideal for sparse, zero-inflated, or count matrices.",
    },
    "kernel_pca": {
        "name": "Kernel PCA (RBF)",
        "type": "Non-Linear Manifold Projection",
        "supports_variance": False,
        "supports_inverse": False,
        "description": "Projects data into a high-dimensional reproducing Hilbert space using non-linear radial basis kernels.",
    },
    "fast_ica": {
        "name": "Independent Component Analysis (ICA)",
        "type": "Blind Source Separation",
        "supports_variance": False,
        "supports_inverse": True,
        "description": "Decomposes multivariate observations into statistically independent, non-Gaussian source signals.",
    },
    "nmf": {
        "name": "Non-Negative Matrix Factorization (NMF)",
        "type": "Part-Based Additive Factorization",
        "supports_variance": False,
        "supports_inverse": True,
        "description": "Factorizes non-negative feature matrices into purely additive component parts with strictly positive weights.",
    },
    "tsne": {
        "name": "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
        "type": "Non-Linear Neighborhood Embedding",
        "supports_variance": False,
        "supports_inverse": False,
        "description": "Preserves pairwise local affinities in 2D/3D manifold space, optimal for non-linear cluster visualization.",
    },
}


def run_dimensionality_reduction(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    method: str = "pca",
    n_components: int = 2,
    target_col: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes dimensionality reduction with rigorous statistical reporting:
    - Explained & cumulative variance ratios (PCA/SVD)
    - Scree curve coordinate data
    - 2D & 3D projection points
    - Component loadings matrix and top driver attributes
    - Original vs Transformed feature comparison & reconstruction error
    """
    method = method.lower()
    if method not in METHOD_METADATA:
        raise ValueError(f"Unsupported dimensionality reduction method '{method}'. Supported: {list(METHOD_METADATA.keys())}")

    # Determine numeric feature columns
    if not feature_cols:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in feature_cols:
            feature_cols.remove(target_col)

    if len(feature_cols) < 2:
        raise ValueError("Dimensionality reduction requires at least 2 numeric feature columns.")

    # Bound n_components to min(features, rows)
    max_k = min(len(feature_cols), len(df) - 1, 10)
    n_comp = max(2, min(n_components, max_k))

    # Extract and impute data (sampled for large datasets to prevent OOM/slowdown)
    work_df = df
    if len(df) > 30_000:
        work_df = df.sample(30_000, random_state=random_state)
    X_raw = work_df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    # Scaling
    if method == "nmf":
        # NMF strictly requires non-negative data
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

    # Instantiate and fit estimator
    explained_variance_ratio = []
    cumulative_variance_ratio = []
    reconstruction_error = None
    loadings_df = pd.DataFrame()
    top_drivers = {}

    if method == "pca":
        model = PCA(n_components=n_comp, random_state=random_state)
        X_transformed = model.fit_transform(X_scaled)
        explained_variance_ratio = [float(v) for v in model.explained_variance_ratio_]
        cumulative_variance_ratio = [float(v) for v in np.cumsum(model.explained_variance_ratio_)]

        # Reconstruction error
        X_recon = model.inverse_transform(X_transformed)
        reconstruction_error = float(np.mean((X_scaled - X_recon) ** 2))

        # Loadings: components_ has shape (n_components, n_features)
        loadings_df = pd.DataFrame(
            model.components_,
            columns=feature_cols,
            index=[f"PC{i+1}" for i in range(n_comp)],
        )

    elif method == "truncated_svd":
        model = TruncatedSVD(n_components=n_comp, random_state=random_state)
        X_transformed = model.fit_transform(X_scaled)
        explained_variance_ratio = [float(v) for v in model.explained_variance_ratio_]
        cumulative_variance_ratio = [float(v) for v in np.cumsum(model.explained_variance_ratio_)]

        X_recon = model.inverse_transform(X_transformed)
        reconstruction_error = float(np.mean((X_scaled - X_recon) ** 2))

        loadings_df = pd.DataFrame(
            model.components_,
            columns=feature_cols,
            index=[f"SVD{i+1}" for i in range(n_comp)],
        )

    elif method == "kernel_pca":
        model = KernelPCA(n_components=n_comp, kernel="rbf", random_state=random_state)
        X_transformed = model.fit_transform(X_scaled)

    elif method == "fast_ica":
        model = FastICA(n_components=n_comp, random_state=random_state, max_iter=500)
        X_transformed = model.fit_transform(X_scaled)
        if hasattr(model, "components_"):
            loadings_df = pd.DataFrame(
                model.components_,
                columns=feature_cols,
                index=[f"IC{i+1}" for i in range(n_comp)],
            )

    elif method == "nmf":
        model = NMF(n_components=n_comp, random_state=random_state, max_iter=400)
        X_transformed = model.fit_transform(X_scaled)
        if hasattr(model, "components_"):
            loadings_df = pd.DataFrame(
                model.components_,
                columns=feature_cols,
                index=[f"NMF{i+1}" for i in range(n_comp)],
            )

    elif method == "tsne":
        perp = min(30.0, max(5.0, (len(df) - 1) / 3.0))
        # Cap rows for t-SNE performance if dataset is massive
        sample_indices = np.arange(len(df))
        if len(df) > 1000:
            np.random.seed(random_state)
            sample_indices = np.random.choice(len(df), size=1000, replace=False)
            X_scaled_sample = X_scaled[sample_indices]
        else:
            X_scaled_sample = X_scaled

        model = TSNE(n_components=min(3, n_comp), random_state=random_state, perplexity=perp)
        X_tsne_sub = model.fit_transform(X_scaled_sample)

        # Pad back to full length for alignment
        X_transformed = np.zeros((len(df), min(3, n_comp)))
        X_transformed[sample_indices] = X_tsne_sub

    # Extract Top Drivers per Component from Loadings
    if not loadings_df.empty:
        for comp_name in loadings_df.index:
            weights = loadings_df.loc[comp_name]
            # Top positive and top negative
            pos_top = weights.sort_values(ascending=False).head(3).to_dict()
            neg_top = weights.sort_values(ascending=True).head(3).to_dict()
            top_drivers[comp_name] = {
                "positive": {k: round(float(v), 4) for k, v in pos_top.items()},
                "negative": {k: round(float(v), 4) for k, v in neg_top.items()},
            }

    # Scree Plot Data
    scree_data = []
    for i, (ev, cv) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
        scree_data.append({
            "component": f"PC{i+1}",
            "index": i + 1,
            "variance_pct": round(ev * 100, 2),
            "cumulative_pct": round(cv * 100, 2),
        })

    # Prepare 2D and 3D Projections with Subsampling for fast interactive charts
    n_pts = len(df)
    sub_step = max(1, n_pts // 500)
    indices = list(range(0, n_pts, sub_step))

    target_values = []
    if target_col and target_col in df.columns:
        target_series = df[target_col].iloc[indices].astype(str).tolist()
        target_values = target_series
    else:
        target_values = [f"Record {idx}" for idx in indices]

    proj_2d = []
    for i, idx in enumerate(indices):
        proj_2d.append({
            "dim1": round(float(X_transformed[idx, 0]), 4),
            "dim2": round(float(X_transformed[idx, 1]), 4),
            "target": str(target_values[i]),
            "index": int(idx),
        })

    proj_3d = []
    if X_transformed.shape[1] >= 3:
        for i, idx in enumerate(indices):
            proj_3d.append({
                "dim1": round(float(X_transformed[idx, 0]), 4),
                "dim2": round(float(X_transformed[idx, 1]), 4),
                "dim3": round(float(X_transformed[idx, 2]), 4),
                "target": str(target_values[i]),
                "index": int(idx),
            })

    # Original vs PCA Comparison Metadata
    total_variance_retained = cumulative_variance_ratio[-1] * 100 if cumulative_variance_ratio else None
    compression_ratio = round((1.0 - (n_comp / len(feature_cols))) * 100, 1)

    comparison_summary = {
        "original_features_count": len(feature_cols),
        "reduced_dimensions_count": n_comp,
        "compression_ratio_pct": compression_ratio,
        "variance_retained_pct": round(total_variance_retained, 2) if total_variance_retained else "N/A (Non-linear)",
        "reconstruction_error": round(reconstruction_error, 4) if reconstruction_error is not None else "N/A",
    }

    # Transformed DataFrame representation
    transformed_cols = [f"{method.upper()}_{i+1}" for i in range(n_comp)]
    transformed_df = pd.DataFrame(X_transformed, columns=transformed_cols, index=df.index)

    return {
        "method": method,
        "metadata": METHOD_METADATA[method],
        "n_components": n_comp,
        "feature_cols": feature_cols,
        "scree_data": scree_data,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance_ratio": cumulative_variance_ratio,
        "loadings": loadings_df.round(4).to_dict() if not loadings_df.empty else {},
        "top_drivers": top_drivers,
        "proj_2d": proj_2d,
        "proj_3d": proj_3d,
        "comparison_summary": comparison_summary,
        "transformed_df": transformed_df,
    }
