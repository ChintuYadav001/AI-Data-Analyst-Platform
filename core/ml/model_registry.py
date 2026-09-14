"""
core.ml.model_registry
======================
Institutional Model Catalog and Factory supporting Regression, Classification,
Clustering, Anomaly Detection, and Dimensionality Reduction.
"""

from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass

# Base Scikit-Learn Estimators
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
    BayesianRidge,
    LogisticRegression,
    SGDClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVR, SVC, OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, FastICA, NMF
from sklearn.manifold import TSNE

# Optional Advanced Libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


@dataclass
class ModelInfo:
    id: str
    name: str
    task: str  # regression, binary_classification, multiclass_classification, clustering, anomaly, decomposition
    family: str  # linear, tree, ensemble, boosting, distance, probabilistic, svm, anomaly, decomposition
    speed: str  # Fast, Moderate, Slow
    complexity: str  # Low, Medium, High
    description: str
    default_params: Dict[str, Any]
    param_grid: Dict[str, List[Any]]


# Complete Model Metadata Registry
REGISTRY: Dict[str, ModelInfo] = {
    # ------------------ REGRESSION ------------------
    "linear_regression": ModelInfo(
        id="linear_regression",
        name="Linear Regression (OLS)",
        task="regression",
        family="linear",
        speed="Fast",
        complexity="Low",
        description="Classical ordinary least squares regression baseline.",
        default_params={},
        param_grid={},
    ),
    "ridge_regression": ModelInfo(
        id="ridge_regression",
        name="Ridge Regression (L2)",
        task="regression",
        family="linear",
        speed="Fast",
        complexity="Low",
        description="Linear regression with L2 regularization to counter multicollinearity.",
        default_params={"alpha": 1.0, "random_state": 42},
        param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
    "lasso_regression": ModelInfo(
        id="lasso_regression",
        name="Lasso Regression (L1)",
        task="regression",
        family="linear",
        speed="Fast",
        complexity="Low",
        description="Linear regression with L1 regularization inducing feature sparsity.",
        default_params={"alpha": 1.0, "random_state": 42},
        param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    ),
    "elastic_net": ModelInfo(
        id="elastic_net",
        name="Elastic Net",
        task="regression",
        family="linear",
        speed="Fast",
        complexity="Low",
        description="Linear regression combining L1 and L2 penalties.",
        default_params={"alpha": 1.0, "l1_ratio": 0.5, "random_state": 42},
        param_grid={"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
    ),
    "huber_regressor": ModelInfo(
        id="huber_regressor",
        name="Huber Regressor",
        task="regression",
        family="linear",
        speed="Fast",
        complexity="Low",
        description="Robust linear regression resistant to heavy-tailed outliers.",
        default_params={"epsilon": 1.35, "max_iter": 300},
        param_grid={"epsilon": [1.1, 1.35, 1.75], "alpha": [0.0001, 0.001, 0.01]},
    ),
    "bayesian_ridge": ModelInfo(
        id="bayesian_ridge",
        name="Bayesian Ridge Regression",
        task="regression",
        family="linear",
        speed="Fast",
        complexity="Low",
        description="Probabilistic regression estimating parameter posterior distributions.",
        default_params={"max_iter": 300},
        param_grid={"alpha_1": [1e-6, 1e-4], "lambda_1": [1e-6, 1e-4]},
    ),
    "knn_regressor": ModelInfo(
        id="knn_regressor",
        name="KNN Regressor",
        task="regression",
        family="distance",
        speed="Moderate",
        complexity="Low",
        description="Instance-based regression predicting average of k-nearest neighbors.",
        default_params={"n_neighbors": 5},
        param_grid={"n_neighbors": [3, 5, 7, 11]},
    ),
    "svr": ModelInfo(
        id="svr",
        name="Support Vector Regression (SVR)",
        task="regression",
        family="svm",
        speed="Moderate",
        complexity="Medium",
        description="Epsilon-support vector regression with RBF non-linear kernel.",
        default_params={"C": 1.0, "epsilon": 0.1},
        param_grid={"C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 0.2]},
    ),
    "decision_tree_regressor": ModelInfo(
        id="decision_tree_regressor",
        name="Decision Tree Regressor",
        task="regression",
        family="tree",
        speed="Fast",
        complexity="Low",
        description="Non-linear decision tree regressor.",
        default_params={"max_depth": 8, "random_state": 42},
        param_grid={"max_depth": [4, 6, 8, 12, None], "min_samples_split": [2, 5, 10]},
    ),
    "random_forest_regressor": ModelInfo(
        id="random_forest_regressor",
        name="Random Forest Regressor",
        task="regression",
        family="ensemble",
        speed="Moderate",
        complexity="Medium",
        description="Bagged ensemble of randomized decision trees.",
        default_params={"n_estimators": 100, "max_depth": 12, "random_state": 42, "n_jobs": -1},
        param_grid={"n_estimators": [50, 100, 200], "max_depth": [6, 10, 15, None]},
    ),
    "extra_trees_regressor": ModelInfo(
        id="extra_trees_regressor",
        name="Extra Trees Regressor",
        task="regression",
        family="ensemble",
        speed="Moderate",
        complexity="Medium",
        description="Extremely Randomized Trees with higher variance reduction.",
        default_params={"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        param_grid={"n_estimators": [50, 100, 200], "max_depth": [6, 10, None]},
    ),
    "gradient_boosting_regressor": ModelInfo(
        id="gradient_boosting_regressor",
        name="Gradient Boosting Regressor",
        task="regression",
        family="boosting",
        speed="Moderate",
        complexity="High",
        description="Sequential gradient boosted regression trees.",
        default_params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
        param_grid={"learning_rate": [0.03, 0.1, 0.2], "max_depth": [3, 4, 6], "n_estimators": [50, 100]},
    ),
    "hist_gradient_boosting_regressor": ModelInfo(
        id="hist_gradient_boosting_regressor",
        name="HistGradientBoosting Regressor",
        task="regression",
        family="boosting",
        speed="Fast",
        complexity="High",
        description="Histogram-based gradient boosting optimized for larger datasets.",
        default_params={"max_iter": 100, "learning_rate": 0.1, "random_state": 42},
        param_grid={"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63]},
    ),
    "xgboost_regressor": ModelInfo(
        id="xgboost_regressor",
        name="XGBoost Regressor",
        task="regression",
        family="boosting",
        speed="Fast",
        complexity="High",
        description="Extreme Gradient Boosting with regularized objective.",
        default_params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42, "verbosity": 0},
        param_grid={"learning_rate": [0.03, 0.1, 0.2], "max_depth": [4, 6, 8]},
    ),
    "lightgbm_regressor": ModelInfo(
        id="lightgbm_regressor",
        name="LightGBM Regressor",
        task="regression",
        family="boosting",
        speed="Fast",
        complexity="High",
        description="Light Gradient Boosting Machine with leaf-wise tree growth.",
        default_params={"n_estimators": 100, "learning_rate": 0.1, "random_state": 42, "verbosity": -1},
        param_grid={"learning_rate": [0.03, 0.1, 0.2], "num_leaves": [15, 31, 63]},
    ),

    # ------------------ CLASSIFICATION ------------------
    "logistic_regression": ModelInfo(
        id="logistic_regression",
        name="Logistic Regression",
        task="binary_classification",
        family="linear",
        speed="Fast",
        complexity="Low",
        description="Linear classifier estimating probability via sigmoid/softmax.",
        default_params={"C": 1.0, "max_iter": 1000, "random_state": 42},
        param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "sgd_classifier": ModelInfo(
        id="sgd_classifier",
        name="SGD Classifier (Log Loss)",
        task="binary_classification",
        family="linear",
        speed="Fast",
        complexity="Low",
        description="Stochastic Gradient Descent classifier with probabilistic log loss.",
        default_params={"loss": "log_loss", "max_iter": 1000, "random_state": 42},
        param_grid={"alpha": [1e-4, 1e-3, 1e-2], "penalty": ["l2", "l1", "elasticnet"]},
    ),
    "naive_bayes": ModelInfo(
        id="naive_bayes",
        name="Gaussian Naive Bayes",
        task="binary_classification",
        family="probabilistic",
        speed="Fast",
        complexity="Low",
        description="Fast probabilistic classifier assuming feature independence.",
        default_params={},
        param_grid={},
    ),
    "knn_classifier": ModelInfo(
        id="knn_classifier",
        name="K-Nearest Neighbors",
        task="binary_classification",
        family="distance",
        speed="Moderate",
        complexity="Low",
        description="Non-parametric instance-based classifier.",
        default_params={"n_neighbors": 5},
        param_grid={"n_neighbors": [3, 5, 7, 11]},
    ),
    "svm_classifier": ModelInfo(
        id="svm_classifier",
        name="Support Vector Classifier (SVC)",
        task="binary_classification",
        family="svm",
        speed="Moderate",
        complexity="Medium",
        description="Support vector classification with RBF kernel and probability calibration.",
        default_params={"C": 1.0, "probability": True, "random_state": 42},
        param_grid={"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
    ),
    "decision_tree_classifier": ModelInfo(
        id="decision_tree_classifier",
        name="Decision Tree Classifier",
        task="binary_classification",
        family="tree",
        speed="Fast",
        complexity="Low",
        description="Interpretable decision tree with Gini or entropy split criteria.",
        default_params={"max_depth": 8, "random_state": 42},
        param_grid={"max_depth": [4, 6, 8, 12, None], "criterion": ["gini", "entropy"]},
    ),
    "random_forest_classifier": ModelInfo(
        id="random_forest_classifier",
        name="Random Forest Classifier",
        task="binary_classification",
        family="ensemble",
        speed="Moderate",
        complexity="Medium",
        description="Robust ensemble of bagging decision trees.",
        default_params={"n_estimators": 100, "max_depth": 12, "random_state": 42, "n_jobs": -1},
        param_grid={"n_estimators": [50, 100, 200], "max_depth": [6, 10, 15, None]},
    ),
    "extra_trees_classifier": ModelInfo(
        id="extra_trees_classifier",
        name="Extra Trees Classifier",
        task="binary_classification",
        family="ensemble",
        speed="Moderate",
        complexity="Medium",
        description="Extremely Randomized Trees with increased randomness.",
        default_params={"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        param_grid={"n_estimators": [50, 100, 200], "max_depth": [6, 10, None]},
    ),
    "gradient_boosting_classifier": ModelInfo(
        id="gradient_boosting_classifier",
        name="Gradient Boosting Classifier",
        task="binary_classification",
        family="boosting",
        speed="Moderate",
        complexity="High",
        description="Iterative gradient boosting algorithm for classification.",
        default_params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
        param_grid={"learning_rate": [0.03, 0.1, 0.2], "max_depth": [3, 4, 6]},
    ),
    "hist_gradient_boosting_classifier": ModelInfo(
        id="hist_gradient_boosting_classifier",
        name="HistGradientBoosting Classifier",
        task="binary_classification",
        family="boosting",
        speed="Fast",
        complexity="High",
        description="High performance histogram-based gradient boosting classifier.",
        default_params={"max_iter": 100, "learning_rate": 0.1, "random_state": 42},
        param_grid={"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63]},
    ),
    "xgboost_classifier": ModelInfo(
        id="xgboost_classifier",
        name="XGBoost Classifier",
        task="binary_classification",
        family="boosting",
        speed="Fast",
        complexity="High",
        description="Extreme Gradient Boosting classifier with GPU/multi-core acceleration.",
        default_params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42, "verbosity": 0, "eval_metric": "logloss"},
        param_grid={"learning_rate": [0.03, 0.1, 0.2], "max_depth": [4, 6, 8]},
    ),
    "lightgbm_classifier": ModelInfo(
        id="lightgbm_classifier",
        name="LightGBM Classifier",
        task="binary_classification",
        family="boosting",
        speed="Fast",
        complexity="High",
        description="Light Gradient Boosting Machine classifier for tabular efficiency.",
        default_params={"n_estimators": 100, "learning_rate": 0.1, "random_state": 42, "verbosity": -1},
        param_grid={"learning_rate": [0.03, 0.1, 0.2], "num_leaves": [15, 31, 63]},
    ),

    # ------------------ UNSUPERVISED CLUSTERING ------------------
    "kmeans": ModelInfo(
        id="kmeans",
        name="K-Means Clustering",
        task="clustering",
        family="clustering",
        speed="Fast",
        complexity="Low",
        description="Centroid-based partition clustering minimizing inertia.",
        default_params={"n_clusters": 3, "random_state": 42, "n_init": "auto"},
        param_grid={"n_clusters": [2, 3, 4, 5, 8]},
    ),
    "dbscan": ModelInfo(
        id="dbscan",
        name="DBSCAN Clustering",
        task="clustering",
        family="clustering",
        speed="Moderate",
        complexity="Medium",
        description="Density-based spatial clustering discovering arbitrary shapes and noise.",
        default_params={"eps": 0.5, "min_samples": 5},
        param_grid={"eps": [0.3, 0.5, 0.8], "min_samples": [3, 5, 10]},
    ),
    "agglomerative": ModelInfo(
        id="agglomerative",
        name="Agglomerative Hierarchical",
        task="clustering",
        family="clustering",
        speed="Slow",
        complexity="Medium",
        description="Hierarchical bottom-up tree clustering using Ward linkage.",
        default_params={"n_clusters": 3},
        param_grid={"n_clusters": [2, 3, 4, 5]},
    ),

    # ------------------ ANOMALY DETECTION ------------------
    "isolation_forest": ModelInfo(
        id="isolation_forest",
        name="Isolation Forest",
        task="anomaly",
        family="anomaly",
        speed="Fast",
        complexity="Low",
        description="Ensemble isolating anomalies via recursive random feature splits.",
        default_params={"n_estimators": 100, "contamination": 0.05, "random_state": 42},
        param_grid={"contamination": [0.01, 0.05, 0.1]},
    ),
    "local_outlier_factor": ModelInfo(
        id="local_outlier_factor",
        name="Local Outlier Factor (LOF)",
        task="anomaly",
        family="anomaly",
        speed="Moderate",
        complexity="Medium",
        description="Density-based anomaly detection measuring local density deviation.",
        default_params={"n_neighbors": 20, "contamination": 0.05, "novelty": True},
        param_grid={"n_neighbors": [10, 20, 35]},
    ),
    "one_class_svm": ModelInfo(
        id="one_class_svm",
        name="One-Class SVM",
        task="anomaly",
        family="anomaly",
        speed="Moderate",
        complexity="Medium",
        description="Kernel-based support vector boundary enclosing the normal data envelope.",
        default_params={"nu": 0.05, "kernel": "rbf", "gamma": "scale"},
        param_grid={"nu": [0.01, 0.05, 0.1]},
    ),
    "elliptic_envelope": ModelInfo(
        id="elliptic_envelope",
        name="Elliptic Envelope",
        task="anomaly",
        family="anomaly",
        speed="Fast",
        complexity="Low",
        description="Robust Mahalanobis covariance estimation for Gaussian outlier detection.",
        default_params={"contamination": 0.05, "random_state": 42},
        param_grid={"contamination": [0.01, 0.05, 0.1]},
    ),

    # ------------------ DIMENSIONALITY REDUCTION ------------------
    "pca": ModelInfo(
        id="pca",
        name="Principal Component Analysis (PCA)",
        task="decomposition",
        family="decomposition",
        speed="Fast",
        complexity="Low",
        description="Linear orthogonal projection maximizing explained variance.",
        default_params={"n_components": 2, "random_state": 42},
        param_grid={"n_components": [2, 3, 5]},
    ),
    "truncated_svd": ModelInfo(
        id="truncated_svd",
        name="Truncated SVD",
        task="decomposition",
        family="decomposition",
        speed="Fast",
        complexity="Low",
        description="Singular Value Decomposition robust to sparse and non-centered matrices.",
        default_params={"n_components": 2, "random_state": 42},
        param_grid={"n_components": [2, 3, 5]},
    ),
    "kernel_pca": ModelInfo(
        id="kernel_pca",
        name="Kernel PCA",
        task="decomposition",
        family="decomposition",
        speed="Moderate",
        complexity="Medium",
        description="Non-linear dimensionality reduction via RBF kernel trick.",
        default_params={"n_components": 2, "kernel": "rbf", "random_state": 42},
        param_grid={"n_components": [2, 3], "kernel": ["rbf", "poly", "cosine"]},
    ),
    "fast_ica": ModelInfo(
        id="fast_ica",
        name="Independent Component Analysis (ICA)",
        task="decomposition",
        family="decomposition",
        speed="Fast",
        complexity="Medium",
        description="Separates multivariate signals into statistically independent subcomponents.",
        default_params={"n_components": 2, "random_state": 42},
        param_grid={"n_components": [2, 3]},
    ),
    "nmf": ModelInfo(
        id="nmf",
        name="Non-Negative Matrix Factorization (NMF)",
        task="decomposition",
        family="decomposition",
        speed="Moderate",
        complexity="Medium",
        description="Factorizes non-negative matrices into additive part-based representations.",
        default_params={"n_components": 2, "random_state": 42, "max_iter": 300},
        param_grid={"n_components": [2, 3]},
    ),
    "tsne": ModelInfo(
        id="tsne",
        name="t-Distributed Stochastic Neighbor Embedding (t-SNE)",
        task="decomposition",
        family="decomposition",
        speed="Slow",
        complexity="High",
        description="Non-linear manifold visualization preserving local neighborhood structure.",
        default_params={"n_components": 2, "random_state": 42, "perplexity": 30.0},
        param_grid={"perplexity": [15.0, 30.0, 50.0]},
    ),
}


# Compute Profile Candidate Subsets
PROFILE_CANDIDATES = {
    "Fast": {
        "regression": [
            "linear_regression",
            "ridge_regression",
            "decision_tree_regressor",
            "hist_gradient_boosting_regressor",
        ],
        "classification": [
            "logistic_regression",
            "naive_bayes",
            "decision_tree_classifier",
            "hist_gradient_boosting_classifier",
        ],
        "anomaly": ["isolation_forest", "elliptic_envelope"],
    },
    "Balanced": {
        "regression": [
            "ridge_regression",
            "huber_regressor",
            "decision_tree_regressor",
            "random_forest_regressor",
            "extra_trees_regressor",
            "hist_gradient_boosting_regressor",
        ],
        "classification": [
            "logistic_regression",
            "sgd_classifier",
            "decision_tree_classifier",
            "random_forest_classifier",
            "extra_trees_classifier",
            "hist_gradient_boosting_classifier",
        ],
        "anomaly": ["isolation_forest", "local_outlier_factor", "elliptic_envelope"],
    },
    "Deep": {
        "regression": [
            "ridge_regression",
            "bayesian_ridge",
            "knn_regressor",
            "svr",
            "random_forest_regressor",
            "gradient_boosting_regressor",
            "hist_gradient_boosting_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
        ],
        "classification": [
            "logistic_regression",
            "knn_classifier",
            "svm_classifier",
            "random_forest_classifier",
            "gradient_boosting_classifier",
            "hist_gradient_boosting_classifier",
            "xgboost_classifier",
            "lightgbm_classifier",
        ],
        "anomaly": ["isolation_forest", "local_outlier_factor", "one_class_svm", "elliptic_envelope"],
    },
}


def get_candidate_models(task: str, compute_profile: str = "Balanced") -> List[str]:
    """Get list of model IDs according to task and compute profile."""
    if "classification" in task:
        category = "classification"
    elif "anomaly" in task:
        category = "anomaly"
    else:
        category = "regression"

    profile_dict = PROFILE_CANDIDATES.get(compute_profile, PROFILE_CANDIDATES["Balanced"])
    candidates = list(profile_dict.get(category, []))

    # Filter out XGBoost or LightGBM if not installed
    filtered = []
    for model_id in candidates:
        if "xgboost" in model_id and not XGB_AVAILABLE:
            continue
        if "lightgbm" in model_id and not LGBM_AVAILABLE:
            continue
        filtered.append(model_id)

    return filtered


def create_model_instance(model_id: str, custom_params: Optional[Dict[str, Any]] = None) -> Any:
    """Instantiate the raw estimator for a given model_id with merged parameters."""
    info = REGISTRY.get(model_id)
    if not info:
        raise ValueError(f"Unknown model_id: {model_id}")

    params = dict(info.default_params)
    if custom_params:
        params.update(custom_params)

    # 1. Regressors
    if model_id == "linear_regression":
        return LinearRegression(**params)
    elif model_id == "ridge_regression":
        return Ridge(**params)
    elif model_id == "lasso_regression":
        return Lasso(**params)
    elif model_id == "elastic_net":
        return ElasticNet(**params)
    elif model_id == "huber_regressor":
        return HuberRegressor(**params)
    elif model_id == "bayesian_ridge":
        return BayesianRidge(**params)
    elif model_id == "knn_regressor":
        return KNeighborsRegressor(**params)
    elif model_id == "svr":
        return SVR(**params)
    elif model_id == "decision_tree_regressor":
        return DecisionTreeRegressor(**params)
    elif model_id == "random_forest_regressor":
        return RandomForestRegressor(**params)
    elif model_id == "extra_trees_regressor":
        return ExtraTreesRegressor(**params)
    elif model_id == "gradient_boosting_regressor":
        return GradientBoostingRegressor(**params)
    elif model_id == "hist_gradient_boosting_regressor":
        return HistGradientBoostingRegressor(**params)
    elif model_id == "xgboost_regressor":
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        return xgb.XGBRegressor(**params)
    elif model_id == "lightgbm_regressor":
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed.")
        return lgb.LGBMRegressor(**params)

    # 2. Classifiers
    elif model_id == "logistic_regression":
        return LogisticRegression(**params)
    elif model_id == "sgd_classifier":
        return SGDClassifier(**params)
    elif model_id == "naive_bayes":
        return GaussianNB(**params)
    elif model_id == "knn_classifier":
        return KNeighborsClassifier(**params)
    elif model_id == "svm_classifier":
        return SVC(**params)
    elif model_id == "decision_tree_classifier":
        return DecisionTreeClassifier(**params)
    elif model_id == "random_forest_classifier":
        return RandomForestClassifier(**params)
    elif model_id == "extra_trees_classifier":
        return ExtraTreesClassifier(**params)
    elif model_id == "gradient_boosting_classifier":
        return GradientBoostingClassifier(**params)
    elif model_id == "hist_gradient_boosting_classifier":
        return HistGradientBoostingClassifier(**params)
    elif model_id == "xgboost_classifier":
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        return xgb.XGBClassifier(**params)
    elif model_id == "lightgbm_classifier":
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed.")
        return lgb.LGBMClassifier(**params)

    # 3. Unsupervised Clustering
    elif model_id == "kmeans":
        return KMeans(**params)
    elif model_id == "dbscan":
        return DBSCAN(**params)
    elif model_id == "agglomerative":
        return AgglomerativeClustering(**params)

    # 4. Anomaly Detection
    elif model_id == "isolation_forest":
        return IsolationForest(**params)
    elif model_id == "local_outlier_factor":
        return LocalOutlierFactor(**params)
    elif model_id == "one_class_svm":
        return OneClassSVM(**params)
    elif model_id == "elliptic_envelope":
        return EllipticEnvelope(**params)

    # 5. Dimensionality Reduction
    elif model_id == "pca":
        return PCA(**params)
    elif model_id == "truncated_svd":
        return TruncatedSVD(**params)
    elif model_id == "kernel_pca":
        return KernelPCA(**params)
    elif model_id == "fast_ica":
        return FastICA(**params)
    elif model_id == "nmf":
        return NMF(**params)
    elif model_id == "tsne":
        return TSNE(**params)

    raise ValueError(f"No instantiation implementation for model_id: {model_id}")
