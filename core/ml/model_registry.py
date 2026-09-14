"""
core.ml.model_registry
======================
Model Catalog and Factory supporting Regression, Classification, Clustering,
and Dimensionality Reduction with Fast / Balanced / Deep compute profiles.
"""

from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass

# Base Scikit-Learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
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
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

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
    task: str  # regression, binary_classification, multiclass_classification, clustering, pca
    family: str  # linear, tree, ensemble, boosting, distance, probabilistic, clustering, decomposition
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

    # ------------------ UNSUPERVISED ------------------
    "kmeans": ModelInfo(
        id="kmeans",
        name="K-Means Clustering",
        task="clustering",
        family="clustering",
        speed="Fast",
        complexity="Low",
        description="Centroid-based partition clustering.",
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
        description="Hierarchical bottom-up tree clustering.",
        default_params={"n_clusters": 3},
        param_grid={"n_clusters": [2, 3, 4, 5]},
    ),
    "pca": ModelInfo(
        id="pca",
        name="Principal Component Analysis",
        task="pca",
        family="decomposition",
        speed="Fast",
        complexity="Low",
        description="Orthogonal linear transformation for dimensionality reduction.",
        default_params={"n_components": 2, "random_state": 42},
        param_grid={"n_components": [2, 3, 5]},
    ),
}


# Compute Profile Candidate Subsets
PROFILE_CANDIDATES = {
    "Fast": {
        "regression": ["linear_regression", "ridge_regression", "decision_tree_regressor", "hist_gradient_boosting_regressor"],
        "classification": ["logistic_regression", "naive_bayes", "decision_tree_classifier", "hist_gradient_boosting_classifier"],
    },
    "Balanced": {
        "regression": [
            "ridge_regression",
            "decision_tree_regressor",
            "random_forest_regressor",
            "extra_trees_regressor",
            "hist_gradient_boosting_regressor",
        ],
        "classification": [
            "logistic_regression",
            "decision_tree_classifier",
            "random_forest_classifier",
            "extra_trees_classifier",
            "hist_gradient_boosting_classifier",
        ],
    },
    "Deep": {
        "regression": [
            "ridge_regression",
            "random_forest_regressor",
            "gradient_boosting_regressor",
            "hist_gradient_boosting_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
        ],
        "classification": [
            "logistic_regression",
            "knn_classifier",
            "random_forest_classifier",
            "gradient_boosting_classifier",
            "hist_gradient_boosting_classifier",
            "xgboost_classifier",
            "lightgbm_classifier",
        ],
    },
}


def get_candidate_models(task: str, compute_profile: str = "Balanced") -> List[str]:
    """Get list of model IDs according to task and compute profile."""
    category = "classification" if "classification" in task else "regression"
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

    # Factory Mapping
    if model_id == "linear_regression":
        return LinearRegression(**params)
    elif model_id == "ridge_regression":
        return Ridge(**params)
    elif model_id == "lasso_regression":
        return Lasso(**params)
    elif model_id == "elastic_net":
        return ElasticNet(**params)
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

    # Classifiers
    elif model_id == "logistic_regression":
        return LogisticRegression(**params)
    elif model_id == "naive_bayes":
        return GaussianNB(**params)
    elif model_id == "knn_classifier":
        return KNeighborsClassifier(**params)
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

    # Unsupervised
    elif model_id == "kmeans":
        return KMeans(**params)
    elif model_id == "dbscan":
        return DBSCAN(**params)
    elif model_id == "agglomerative":
        return AgglomerativeClustering(**params)
    elif model_id == "pca":
        return PCA(**params)

    raise ValueError(f"No instantiation implementation for model_id: {model_id}")
