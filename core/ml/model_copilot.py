"""
core.ml.model_copilot
=====================
High-Quality AI Model Knowledge & Explanation Engine for Quantum Analyst.
Explains every machine learning model in simple, intuitive layman terms,
provides real-world business use cases, pros & cons, and contextual insights
grounded in the active dataset and champion models.
"""

from typing import Dict, Any, List, Optional
import re
import pandas as pd
import numpy as np

# Structured Encyclopedia of ML Models with Simple Analogies and Business Use Cases
MODEL_ENCYCLOPEDIA: Dict[str, Dict[str, Any]] = {
    # ------------------ REGRESSION ------------------
    "linear_regression": {
        "name": "Linear Regression (OLS)",
        "family": "Linear Baseline",
        "analogy": "Like drawing a straight ruler through scattered points on graph paper so the ruler is as close as possible to all points at once.",
        "simple_explanation": (
            "Linear Regression assumes that when one thing increases, the other thing goes up or down at a steady, "
            "fixed rate (like how spending more on advertising steadily increases sales)."
        ),
        "business_uses": [
            "📈 Sales & Revenue Forecasting based on marketing spend.",
            "🏠 Real Estate Pricing based on square footage and bedroom count.",
            "💰 Customer Lifetime Value (CLV) estimation from purchase frequency.",
        ],
        "pros": ["Super fast to compute", "Extremely easy to explain to executives and stakeholders", "Provides exact dollar/unit multipliers for each feature"],
        "cons": ["Cannot capture curved, wavy, or complex non-linear relationships", "Easily distorted by extreme outliers"],
    },
    "ridge_regression": {
        "name": "Ridge Regression (L2)",
        "family": "Regularized Linear",
        "analogy": "Like putting a gentle leash on every feature so none of them can get too bossy or over-confident.",
        "simple_explanation": (
            "When many features are telling almost the same story (like temperature in Celsius vs Fahrenheit), "
            "standard regression gets confused. Ridge shrinks their influence slightly so all features play nicely together."
        ),
        "business_uses": [
            "📊 Portfolio Risk Analysis when financial stocks move together.",
            "🛍️ Retail Demand Forecasting with hundreds of related promo indicators.",
        ],
        "pros": ["Prevents overfitting when you have many correlated features", "Stable and mathematically reliable"],
        "cons": ["Keeps all features; it shrinks weights toward zero but never turns them completely off"],
    },
    "lasso_regression": {
        "name": "Lasso Regression (L1)",
        "family": "Sparse Regularized Linear",
        "analogy": "Like a strict judge that fires all lazy or redundant workers, keeping only the absolute best contributors.",
        "simple_explanation": (
            "Lasso does automatic feature selection. If a feature is not genuinely helping the prediction, "
            "Lasso sets its weight to exactly zero, eliminating it from the model."
        ),
        "business_uses": [
            "🧬 Medical Diagnostics selecting 10 critical biomarkers out of 10,000 genes.",
            "🎯 Marketing Attribution determining which 3 channels actually drive conversions.",
        ],
        "pros": ["Automatically discards useless features", "Creates clean, compact models with fewer variables"],
        "cons": ["Can arbitrarily pick one feature and discard another if two features are nearly identical"],
    },
    "elastic_net": {
        "name": "Elastic Net",
        "family": "Hybrid Linear",
        "analogy": "Like a hybrid team leader combining the discipline of Lasso and the teamwork of Ridge.",
        "simple_explanation": (
            "Elastic Net blends L1 (Lasso) and L2 (Ridge) penalties. It removes useless noise features like Lasso, "
            "but keeps groups of strongly correlated features working together like Ridge."
        ),
        "business_uses": [
            "📈 High-dimensional financial modeling with thousands of macroeconomic signals.",
            "🏢 Macroeconomic KPI projections across multiple correlated economic indices.",
        ],
        "pros": ["Best of both worlds: handles multicollinearity and performs feature selection", "More stable than pure Lasso"],
        "cons": ["Requires tuning two knobs: overall penalty strength and the balance ratio"],
    },
    "huber_regressor": {
        "name": "Huber Regressor",
        "family": "Robust Linear",
        "analogy": "Like a fair referee who listens to normal complaints but ignores wild, screaming temper tantrums.",
        "simple_explanation": (
            "Standard regression squares errors, meaning one wild outlier can ruin the entire model. "
            "Huber treats normal errors linearly and ignores crazy extreme values."
        ),
        "business_uses": [
            "💳 Transaction Volume Forecasting where occasional billion-dollar wire transfers would warp standard models.",
            "🏭 Manufacturing Sensor Calibration where faulty sensor spikes occur.",
        ],
        "pros": ["Highly resistant to wild outliers without needing to throw data away", "More stable than OLS"],
        "cons": ["Only works on linear patterns"],
    },
    "bayesian_ridge": {
        "name": "Bayesian Ridge",
        "family": "Probabilistic Linear",
        "analogy": "Like a cautious forecaster who doesn't just give you one number, but also tells you how confident they feel about it.",
        "simple_explanation": (
            "Instead of assuming a single fixed formula, Bayesian Ridge treats weights as probability distributions, "
            "updating its beliefs as more data arrives."
        ),
        "business_uses": [
            "🛡️ Insurance Risk Assessment where uncertainty quantification is mandatory.",
            "🔬 Clinical Trial Outcome Prediction with small sample sizes.",
        ],
        "pros": ["Adapts well to small datasets", "Provides confidence ranges for its predictions"],
        "cons": ["Can be slower to compute on large datasets"],
    },
    "knn_regressor": {
        "name": "K-Nearest Neighbors (KNN) Regressor",
        "family": "Instance / Distance",
        "analogy": "Like pricing a house by asking: 'What did the 5 houses most similar to this one sell for?'",
        "simple_explanation": (
            "KNN doesn't learn a formula. Whenever you ask it to predict, it searches your dataset for the $k$ most "
            "similar historical records and averages their values."
        ),
        "business_uses": [
            "🏡 Real Estate Valuations based on neighboring comparable properties.",
            "🎬 Recommender Systems (collaborative filtering by user taste similarity).",
        ],
        "pros": ["Zero assumptions about data shape (handles crazy curves easily)", "Very intuitive logic"],
        "cons": ["Slow at prediction time because it has to search through all historical data", "Sensitive to unscaled features"],
    },
    "svr": {
        "name": "Support Vector Regression (SVR)",
        "family": "Kernel Support Vector",
        "analogy": "Like laying down a flexible rubber tube through the data cloud—points inside the tube don't trigger any penalties.",
        "simple_explanation": (
            "SVR finds a smooth non-linear curve that keeps as many training points as possible within an allowable "
            "margin band (epsilon), ignoring small errors."
        ),
        "business_uses": [
            "⚡ Energy Grid Load & Power Demand Forecasting.",
            "🧪 Chemical Yield & Concentration Modeling.",
        ],
        "pros": ["Exceptional accuracy on complex non-linear curves", "Guaranteed global optimum without local minima traps"],
        "cons": ["Computationally heavy on datasets with more than 50,000 rows", "Tuning C, gamma, and epsilon requires care"],
    },
    "decision_tree_regressor": {
        "name": "Decision Tree Regressor",
        "family": "Tree",
        "analogy": "Like a flowchart or a game of 20 Questions: 'Is age > 40? If Yes -> Is income > $80k? If Yes -> Predict $120k.'",
        "simple_explanation": (
            "Splits your dataset into smaller and smaller boxes by asking simple yes/no questions about features, "
            "then averages the target values in each final box."
        ),
        "business_uses": [
            "🏥 Patient Hospital Length-of-Stay Estimation.",
            "🚚 Logistics Delivery Time Estimation based on distance, traffic, and weather.",
        ],
        "pros": ["100% human readable and explainable", "Handles non-linear patterns and interactions automatically without feature scaling"],
        "cons": ["A single tree easily memorizes training data (overfitting)", "Can be unstable: small data changes change the whole tree"],
    },
    "random_forest_regressor": {
        "name": "Random Forest Regressor",
        "family": "Bagging Ensemble",
        "analogy": "Like taking advice from a council of 100 independent experts and taking their democratic average.",
        "simple_explanation": (
            "Builds a large forest of randomized decision trees. Each tree sees a slightly different subset of data and features. "
            "While individual trees make errors, their combined vote cancels out mistakes."
        ),
        "business_uses": [
            "💳 Credit Card Spending Limit Prediction.",
            "🌾 Agricultural Crop Yield Forecasting.",
            "🏢 Employee Salary & Compensation Benchmarking.",
        ],
        "pros": ["One of the most robust, reliable 'workhorse' algorithms in machine learning", "Almost impossible to overfit severely", "Handles missing data well"],
        "cons": ["Slower to train than a single tree", "Larger file size when exported"],
    },
    "extra_trees_regressor": {
        "name": "Extra Trees Regressor",
        "family": "Extremely Randomized Ensemble",
        "analogy": "Like a Random Forest, but each expert makes split decisions with even more spontaneity and speed.",
        "simple_explanation": (
            "Extra Trees takes randomness one step further by choosing split points completely at random rather than searching "
            "for the best mathematical split. Surprisingly, this often reduces variance and speeds up training."
        ),
        "business_uses": [
            "📡 Real-time Telemetry & IoT Sensor Metric Prediction.",
            "🚗 Vehicle Fuel Consumption Modeling.",
        ],
        "pros": ["Trains faster than standard Random Forest", "Often reduces variance on noisy datasets"],
        "cons": ["Can suffer if only a tiny handful of features have real predictive signal"],
    },
    "gradient_boosting_regressor": {
        "name": "Gradient Boosting Regressor",
        "family": "Boosting Ensemble",
        "analogy": "Like an archery student: each new shot is aimed specifically to correct where the previous arrow missed the target.",
        "simple_explanation": (
            "Instead of training independent trees in parallel, Gradient Boosting trains trees sequentially. "
            "Each new tree focuses 100% on predicting the residual errors (mistakes) made by all prior trees combined."
        ),
        "business_uses": [
            "💰 Dynamic Hotel & Airline Ticket Pricing.",
            "📈 Stock Volatility & Options Valuation.",
        ],
        "pros": ["Often achieves the highest accuracy on tabular business data", "Learns intricate non-linear relationships"],
        "cons": ["Can overfit if trained for too many rounds without learning rate shrinkage"],
    },
    "hist_gradient_boosting_regressor": {
        "name": "HistGradientBoosting Regressor",
        "family": "Histogram Boosting",
        "analogy": "Like rounding messy decimal numbers into neat integer bins so you can sort and analyze them 10x faster.",
        "simple_explanation": (
            "Scikit-learn's modern boosting engine. It groups continuous features into 256 discrete bins, "
            "slashing training time from minutes to seconds while preserving top-tier accuracy."
        ),
        "business_uses": [
            "⚡ Large-scale E-Commerce Order Value Estimation (hundreds of thousands of rows).",
            "📦 Freight Shipping Rate Calculations.",
        ],
        "pros": ["Blazing fast on large datasets (100k+ rows)", "Native support for missing values without manual imputation"],
        "cons": ["Slight loss of precision on very small datasets (< 100 rows)"],
    },
    "xgboost_regressor": {
        "name": "XGBoost Regressor",
        "family": "Extreme Gradient Boosting",
        "analogy": "Like a Formula 1 race car engineered specifically for winning tabular data competitions with built-in speed brakes.",
        "simple_explanation": (
            "The industry champion algorithm. XGBoost optimizes gradient boosting with hardware acceleration, "
            "clever tree pruning, and mathematical L1/L2 penalties that prevent overfitting."
        ),
        "business_uses": [
            "🏆 Production Recommendation Engines.",
            "🏦 Banking Risk & Loss Given Default (LGD) Modeling.",
            "📉 Insurance Claim Cost Modeling.",
        ],
        "pros": ["Gold standard for structured tabular datasets", "Built-in regularization prevents runaway overfitting", "Multi-core accelerated"],
        "cons": ["Has dozens of hyperparameters that benefit from tuning"],
    },
    "lightgbm_regressor": {
        "name": "LightGBM Regressor",
        "family": "Leaf-Wise Boosting",
        "analogy": "Like a tree that grows only the single most fruitful branch first, rather than growing all branches symmetrically.",
        "simple_explanation": (
            "Developed by Microsoft, LightGBM grows trees leaf-wise rather than depth-wise. "
            "It runs faster, consumes less RAM, and handles high-cardinality data effortlessly."
        ),
        "business_uses": [
            "🛒 Supermarket Inventory Demand Forecasting across 50,000 product SKUs.",
            "📱 Ad Click-Through Rate (CTR) Value Estimation.",
        ],
        "pros": ["Lowest memory footprint of any boosting library", "Fastest training on millions of rows"],
        "cons": ["Leaf-wise growth can overfit on very small datasets if `max_depth` is not constrained"],
    },

    # ------------------ CLASSIFICATION ------------------
    "logistic_regression": {
        "name": "Logistic Regression",
        "family": "Linear Classification",
        "analogy": "Like a probability thermometer: calculates a score between 0% and 100%, and if it passes 50%, calls it a 'Yes'.",
        "simple_explanation": (
            "Despite its name, it is a classification algorithm. It takes input features and squashes the result through "
            "an S-shaped curve (sigmoid) to output a clean probability (e.g., 82% chance customer will churn)."
        ),
        "business_uses": [
            "🚨 Loan Default Classification (Will borrower pay back or default?).",
            "💌 Email Spam Filtering (Is this message spam or legitimate?).",
            "🩺 Medical Screening (Does patient test positive or negative?).",
        ],
        "pros": ["Produces well-calibrated probabilities", "Fast and interpretable (coefficients tell you the odds ratio)"],
        "cons": ["Assumes linear decision boundaries; cannot capture intricate curves without feature engineering"],
    },
    "sgd_classifier": {
        "name": "SGD Classifier",
        "family": "Linear Online Classifier",
        "analogy": "Like a fast student who reads one flashcard at a time and tweaks their knowledge immediately, rather than reading the whole textbook at once.",
        "simple_explanation": (
            "Stochastic Gradient Descent classifier updates weights incrementally row-by-row. "
            "Ideal for streaming data or massive datasets that don't fit in computer RAM."
        ),
        "business_uses": [
            "🌐 Real-time Clickstream Classification as users browse a website.",
            "📰 Streaming News Sentiment Categorization.",
        ],
        "pros": ["Extremely lightweight and fast", "Can learn continuously from streaming data"],
        "cons": ["Sensitive to feature scaling and learning rate schedules"],
    },
    "naive_bayes": {
        "name": "Gaussian Naive Bayes",
        "family": "Probabilistic",
        "analogy": "Like a detective who assumes every clue is independent: 'You have mud on your shoes AND an umbrella, so it's 95% likely it rained.'",
        "simple_explanation": (
            "Uses Bayes' Rule from probability theory. It calculates how often each feature occurs in Class A vs Class B, "
            "assuming (naively) that all features operate independently."
        ),
        "business_uses": [
            "📝 Text Classification & Sentiment Analysis (Positive, Neutral, Negative).",
            "🛡️ Cyber Intrusion Detection based on packet header attributes.",
        ],
        "pros": ["Incredible speed (trains in milliseconds)", "Works well even with small sample sizes"],
        "cons": ["Its 'independence assumption' is almost never true in real life, so probability scores can be overconfident"],
    },
    "knn_classifier": {
        "name": "K-Nearest Neighbors (KNN) Classifier",
        "family": "Distance Classification",
        "analogy": "Like the old saying: 'Birds of a feather flock together.' You are classified based on the majority vote of your 5 nearest neighbors.",
        "simple_explanation": (
            "Plots all data points in space. To classify a new person, it looks at the 5 closest people. "
            "If 4 of them are 'Subscribers' and 1 is 'Cancelled', it classifies the new person as 'Subscriber'."
        ),
        "business_uses": [
            "🏷️ Customer Persona Clustering & Tagging.",
            "🖐️ Handwriting & Digit Recognition.",
        ],
        "pros": ["Naturally handles multi-class problems", "Zero training time (lazy learner)"],
        "cons": ["Slow to predict on large datasets", "Fails if features are not properly normalized"],
    },
    "svm_classifier": {
        "name": "Support Vector Classifier (SVC)",
        "family": "Kernel Support Vector",
        "analogy": "Like drawing a wide, paved highway between two warring cities with the widest possible neutral zone (margin) between them.",
        "simple_explanation": (
            "Finds the mathematical boundary (hyperplane) that maximizes the safety margin between classes. "
            "With the RBF kernel trick, it can warp 2D data into 3D space to separate classes that seem impossible to separate in 2D."
        ),
        "business_uses": [
            "🔬 Cancer Tissue Biopsy Image Classification.",
            "💳 High-Security Facial Recognition & Biometric Verification.",
        ],
        "pros": ["Phenomenal accuracy in high-dimensional feature spaces", "Robust to overfitting if margin $C$ is tuned properly"],
        "cons": ["Slow on datasets with more than 30,000 rows", "Black box: difficult to explain why a single prediction was made"],
    },
    "decision_tree_classifier": {
        "name": "Decision Tree Classifier",
        "family": "Tree",
        "analogy": "Like a clinical diagnosis checklist: 'Fever > 100°F? Yes -> Cough present? Yes -> Diagnosed with Flu.'",
        "simple_explanation": (
            "Creates clear if-then rules to partition data into pure class buckets using Gini Impurity or Entropy."
        ),
        "business_uses": [
            "📋 Automated Loan Underwriting Rulebooks.",
            "🛡️ Regulatory Compliance Approval Checklists.",
        ],
        "pros": ["Highest human interpretability of any machine learning model", "Can be directly translated into SQL `CASE WHEN` statements"],
        "cons": ["Prone to memorizing noise (overfitting) if max depth is not capped"],
    },
    "random_forest_classifier": {
        "name": "Random Forest Classifier",
        "family": "Bagging Ensemble",
        "analogy": "Like having a jury of 100 diverse jurors deliberate and cast votes on a verdict.",
        "simple_explanation": (
            "Builds dozens of decision trees trained on random subsets of data and features. "
            "To classify a record, every tree votes, and the majority class wins."
        ),
        "business_uses": [
            "🛑 Customer Churn Prevention (Identifying customers likely to cancel next month).",
            "🏧 Credit Card Fraud Detection.",
            "🏭 Predictive Machine Failure Warning in factories.",
        ],
        "pros": ["Consistently delivers high accuracy without complex parameter tuning", "Provides reliable feature importance rankings"],
        "cons": ["Cannot be summarized into a single simple flowchart"],
    },
    "xgboost_classifier": {
        "name": "XGBoost Classifier",
        "family": "Extreme Gradient Boosting",
        "analogy": "Like an elite coaching team where each specialist coach works only on fixing the specific flaws left by the previous coaches.",
        "simple_explanation": (
            "Iteratively builds decision trees where each new tree focuses on the misclassified records of previous trees. "
            "Equipped with second-order gradient optimization and regularized tree depth."
        ),
        "business_uses": [
            "🎯 Search Ranking & Ad Conversion Optimization.",
            "🚨 Anti-Money Laundering (AML) Transaction Surveillance.",
        ],
        "pros": ["Industry champion for tabular classification", "High precision on imbalanced datasets"],
        "cons": ["Requires careful tuning of learning rate and tree depth to avoid overfitting"],
    },

    # ------------------ UNSUPERVISED & ANOMALY ------------------
    "kmeans": {
        "name": "K-Means Clustering",
        "family": "Clustering",
        "analogy": "Like placing 3 magnets on a metal tray and letting metal shavings gather around the closest magnet.",
        "simple_explanation": (
            "Unsupervised algorithm that groups data points into $k$ distinct clusters without any target labels. "
            "It moves cluster centers (centroids) until the distance from points to their center is minimized."
        ),
        "business_uses": [
            "👥 Customer Market Segmentation (Budget Buyers, Value Seekers, VIP Spenders).",
            "🗺️ Warehouse & Store Location Optimization based on customer address clusters.",
        ],
        "pros": ["Simple, lightning fast, scales to millions of records"],
        "cons": ["You must specify $k$ in advance", "Assumes clusters are circular/spherical in shape"],
    },
    "dbscan": {
        "name": "DBSCAN Clustering",
        "family": "Density-Based Clustering",
        "analogy": "Like identifying islands in an ocean: dense land masses become clusters, while open water is marked as noise/outliers.",
        "simple_explanation": (
            "Finds clusters of arbitrary shapes by grouping points that are densely packed together. "
            "Points in lonely, low-density regions are automatically tagged as outliers."
        ),
        "business_uses": [
            "🛰️ GPS Fleet Tracking & Delivery Route Discovery.",
            "🔍 Spatial Anomaly & Hotspot Detection.",
        ],
        "pros": ["Does not require specifying cluster count in advance", "Can discover horseshoe, ring, and irregular shapes", "Detects noise automatically"],
        "cons": ["Struggles if clusters have wildly different densities"],
    },
    "isolation_forest": {
        "name": "Isolation Forest",
        "family": "Tree-based Anomaly Detection",
        "analogy": "Like spotting a 7-foot tall basketball player in a crowd of average people: it takes just 1 quick cut to isolate them.",
        "simple_explanation": (
            "Normal points are packed tightly together and require many random cuts to isolate. "
            "Anomalies are unusual and distant, so they get isolated in very few random splits."
        ),
        "business_uses": [
            "💳 Credit Card Fraud & Stolen Identity Detection.",
            "💻 Cybersecurity Network Intrusion & Botnet Identification.",
            "🏭 Manufacturing Defect & Quality Control Inspection.",
        ],
        "pros": ["Blazing fast linear time complexity", "Works amazingly well on high-dimensional data without needing labels"],
        "cons": ["User needs to provide an estimate of contamination rate (e.g. 1% or 5%)"],
    },
    "local_outlier_factor": {
        "name": "Local Outlier Factor (LOF)",
        "family": "Density-based Anomaly",
        "analogy": "Like checking if you are standing uncomfortably far from your friends compared to how close they stand to each other.",
        "simple_explanation": (
            "Compares the local density of an object to the local densities of its neighbors. "
            "An object with significantly lower density than its neighbors is flagged as an outlier."
        ),
        "business_uses": [
            "🏥 Rare Disease Patient Diagnostic Outliers.",
            "📊 Financial Market Trading Flash Crash Pre-warning.",
        ],
        "pros": ["Detects local anomalies that global methods miss", "Does not assume a normal bell-curve distribution"],
        "cons": ["Slower on massive datasets because of $k$-nearest neighbor calculations"],
    },

    # ------------------ DIMENSIONALITY REDUCTION ------------------
    "pca": {
        "name": "Principal Component Analysis (PCA)",
        "family": "Dimensionality Reduction",
        "analogy": "Like taking a 2D photograph of a complex 3D sculpture from the exact angle that captures the maximum amount of shadow and detail.",
        "simple_explanation": (
            "Compresses dozens of correlated columns down into 2 or 3 'Principal Components' that preserve 80%–95% "
            "of the original information, making complex high-dimensional data easy to visualize on a 2D screen."
        ),
        "business_uses": [
            "📉 Data Visualization: Plotting 50 customer survey questions on a 2D scatter plot.",
            "⚡ Preprocessing: Compressing 500 financial variables into 10 orthogonal factors to speed up downstream models.",
            "🔊 Audio & Sensor Noise Reduction.",
        ],
        "pros": ["Eliminates multicollinearity completely (components are 100% uncorrelated)", "Dramatically speeds up modeling"],
        "cons": ["The new components (PC1, PC2) are mathematical combinations, so they lose original column names"],
    },

    # ------------------ TIME SERIES FORECASTING ------------------
    "arima": {
        "name": "ARIMA / SARIMAX",
        "family": "Statistical Time Series",
        "analogy": "Like predicting tomorrow's temperature by combining yesterday's temperature, last week's average, and whether you made a bad prediction yesterday.",
        "simple_explanation": (
            "ARIMA stands for Auto-Regressive Integrated Moving Average. It models trends and temporal correlations "
            "by looking at historical lags and recent prediction errors. SARIMAX adds seasonal cycles and external drivers."
        ),
        "business_uses": [
            "📦 Retail Weekly Inventory Reorder Point Forecasting.",
            "⚡ Utility Electricity Consumption Projections.",
            "📈 Monthly Budget & Cash Flow Forecasts.",
        ],
        "pros": ["Rigorous statistical foundation", "Outputs clean mathematical confidence intervals"],
        "cons": ["Requires the time series to be stationary (or differenced to become stationary)"],
    },
    "holt_winters": {
        "name": "Holt-Winters Exponential Smoothing",
        "family": "Exponential Smoothing",
        "analogy": "Like predicting next quarter's revenue by giving 50% weight to last month, 25% to two months ago, and factoring in the annual Christmas rush.",
        "simple_explanation": (
            "Applies exponentially decaying weights to past data (recent observations count much more than old observations) "
            "while explicitly modeling an upward/downward trend and recurring seasonal rhythm."
        ),
        "business_uses": [
            "🎄 Holiday E-Commerce Sales Projections.",
            "🏨 Hotel Room Booking & Seasonal Occupancy Rates.",
        ],
        "pros": ["Very intuitive and robust to missing dates", "Computes in fractions of a second"],
        "cons": ["Cannot easily incorporate external features (like marketing campaigns) without ML lag extensions"],
    },
}


class ModelCopilotEngine:
    """Conversational reasoning engine specialized in explaining machine learning algorithms."""

    def __init__(self):
        self.encyclopedia = MODEL_ENCYCLOPEDIA

    def answer_model_question(
        self,
        query: str,
        df: Optional[pd.DataFrame] = None,
        champ_info: Optional[Dict[str, Any]] = None,
        benchmark_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes a structured, simple plain-English response to any user question about
        models, comparisons, metrics, PCA, forecasting, or active dataset recommendations.
        """
        q = query.lower().strip()

        # 1. Ask about active champion or best model on the current dataset
        if any(w in q for w in ["best model", "champion", "why did", "winner", "my model", "current model"]) and (champ_info or benchmark_results):
            return self._explain_current_champion(champ_info, benchmark_results, df)

        # 2. General request to explain ALL models or list models
        if any(w in q for w in ["all models", "list of models", "what models", "explain all", "every model"]):
            return self._explain_all_models_summary()

        # 3. PCA / Dimensionality Reduction explanation
        if any(w in q for w in ["pca", "principal component", "dimensionality", "scree"]):
            return self._explain_pca_in_plain_english()

        # 4. Anomaly Detection explanation
        if any(w in q for w in ["anomaly", "outlier", "fraud", "isolation forest", "lof"]):
            return self._explain_anomaly_in_plain_english()

        # 5. Future Forecasting explanation
        if any(w in q for w in ["forecast", "time series", "arima", "holt", "future"]):
            return self._explain_forecasting_in_plain_english()

        # 6. Metric Explanations (Precision, Recall, F1, R2, RMSE, Overfitting)
        if any(w in q for w in ["metric", "precision", "recall", "f1", "r2", "r-squared", "rmse", "mae", "overfitting", "auc"]):
            return self._explain_metrics_in_plain_english(q)

        # 7. Model Comparison (e.g. Random Forest vs XGBoost, Ridge vs Lasso)
        if " vs " in q or "compare" in q or "difference between" in q:
            comp_resp = self._explain_comparison(q)
            if comp_resp:
                return comp_resp

        # 8. Specific Model Search (look for specific model keyword)
        matched_model_key = self._find_model_in_query(q)
        if matched_model_key:
            return self._explain_single_model(matched_model_key)

        # 9. Default intelligent guidance tailored to active dataset
        return self._provide_default_model_guidance(df, champ_info)

    def _find_model_in_query(self, query: str) -> Optional[str]:
        """Detects if query is asking about a specific model."""
        lookup_aliases = {
            "linear regression": "linear_regression",
            "ols": "linear_regression",
            "ridge": "ridge_regression",
            "lasso": "lasso_regression",
            "elastic net": "elastic_net",
            "elasticnet": "elastic_net",
            "huber": "huber_regressor",
            "bayesian ridge": "bayesian_ridge",
            "bayes ridge": "bayesian_ridge",
            "knn regressor": "knn_regressor",
            "knn": "knn_classifier",
            "nearest neighbor": "knn_classifier",
            "svr": "svr",
            "support vector regression": "svr",
            "decision tree": "decision_tree_classifier",
            "tree": "decision_tree_classifier",
            "random forest": "random_forest_classifier",
            "extra trees": "extra_trees_classifier",
            "gradient boosting": "gradient_boosting_regressor",
            "hist gradient": "hist_gradient_boosting_regressor",
            "histgradient": "hist_gradient_boosting_regressor",
            "xgboost": "xgboost_classifier",
            "xgb": "xgboost_classifier",
            "lightgbm": "lightgbm_classifier",
            "lgb": "lightgbm_classifier",
            "logistic regression": "logistic_regression",
            "logistic": "logistic_regression",
            "sgd": "sgd_classifier",
            "naive bayes": "naive_bayes",
            "svm": "svm_classifier",
            "svc": "svm_classifier",
            "support vector machine": "svm_classifier",
            "kmeans": "kmeans",
            "k-means": "kmeans",
            "dbscan": "dbscan",
            "isolation forest": "isolation_forest",
            "lof": "local_outlier_factor",
            "local outlier": "local_outlier_factor",
            "pca": "pca",
            "arima": "arima",
            "holt winters": "holt_winters",
            "holt-winters": "holt_winters",
        }
        for alias, key in lookup_aliases.items():
            if alias in query:
                return key
        return None

    def _explain_single_model(self, model_key: str) -> Dict[str, Any]:
        info = self.encyclopedia.get(model_key)
        if not info:
            return {
                "title": "Model Information",
                "markdown": f"Could not find exact documentation for `{model_key}`. Ask me about Linear Regression, Random Forest, XGBoost, PCA, or Anomaly Detection!",
                "voice_text": "I could not find exact documentation for that model. Try asking about Random Forest, XGBoost, or PCA.",
            }

        uses_bulleted = "\n".join([f"- {u}" for u in info["business_uses"]])
        pros_bulleted = "\n".join([f"  ✅ {p}" for p in info["pros"]])
        cons_bulleted = "\n".join([f"  ⚠️ {c}" for c in info["cons"]])

        md = f"""### 🤖 {info['name']} ({info['family']})

**💡 Simple Analogy**:
> *"{info['analogy']}"*

**📖 In Simple English**:
{info['simple_explanation']}

**💼 Real-World Business Use Cases**:
{uses_bulleted}

**⚖️ Strengths & Trade-offs**:
- **Why use it**:
{pros_bulleted}
- **Watch out for**:
{cons_bulleted}
"""
        voice = (
            f"{info['name']}. In simple terms: {info['analogy']} "
            f"It works like this: {info['simple_explanation']} "
            f"A common business use case is {info['business_uses'][0]}. "
            f"Its key advantage is that {info['pros'][0]}."
        )

        return {
            "title": info["name"],
            "markdown": md,
            "voice_text": self._clean_for_voice(voice),
        }

    def _explain_comparison(self, query: str) -> Optional[Dict[str, Any]]:
        # XGBoost vs Random Forest
        if ("xgboost" in query or "xgb" in query) and ("random forest" in query or "rf" in query):
            md = """### ⚔️ Comparison: Random Forest vs. XGBoost

| Factor | 🌲 Random Forest | 🚀 XGBoost |
|---|---|---|
| **Core Idea** | Builds 100 trees **in parallel** (independent voting). | Builds trees **one-by-one** (each tree fixes mistakes of previous ones). |
| **Analogy** | A jury of 100 people voting democrately. | An archery student adjusting their aim after every missed arrow. |
| **Speed & Setup** | Works great 'out of the box' with minimal tuning. | Needs learning rate tuning, but is ultra-fast when tuned. |
| **Overfitting Risk** | Almost impossible to severely overfit. | Can overfit if too many trees are trained without regularizers. |
| **When to Choose** | When you want reliable results with zero tuning effort. | When you want the highest possible accuracy on competitive tabular data. |
"""
            voice = (
                "Here is the difference between Random Forest and XGBoost in simple terms: "
                "Random Forest builds many trees in parallel, like a jury voting democratically. "
                "XGBoost builds trees sequentially, where each new tree specifically fixes the mistakes of the previous trees. "
                "Choose Random Forest for effortless reliability, or choose XGBoost when you want maximum accuracy."
            )
            return {"title": "Random Forest vs XGBoost", "markdown": md, "voice_text": voice}

        # Ridge vs Lasso
        if "ridge" in query and "lasso" in query:
            md = """### ⚔️ Comparison: Ridge vs. Lasso Regression

| Aspect | 🛡️ Ridge Regression (L2) | 🎯 Lasso Regression (L1) |
|---|---|---|
| **Penalty** | Shrinks feature weights close to zero, but **never zero**. | Shrinks feature weights and sets useless ones to **exactly zero**. |
| **Analogy** | Puts a gentle leash on every feature so all work as a team. | A ruthless boss who fires all non-essential workers. |
| **Best Used When** | You have many correlated features (multicollinearity). | You have 100+ features and suspect only 10 actually matter. |
| **Feature Selection** | No (retains all features in formula). | Yes (automatically discards redundant features). |
"""
            voice = (
                "Ridge versus Lasso: Ridge shrinks feature weights so correlated features cooperate, "
                "while Lasso sets useless feature weights to exactly zero, automatically eliminating them. "
                "Use Ridge when features are correlated; use Lasso when you want to discover the top critical features."
            )
            return {"title": "Ridge vs Lasso", "markdown": md, "voice_text": voice}

        return None

    def _explain_current_champion(
        self,
        champ_info: Optional[Dict[str, Any]],
        benchmark_results: Optional[Dict[str, Any]],
        df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        if not champ_info and benchmark_results:
            champ_id = benchmark_results.get("champion_id")
            champ_info = benchmark_results.get("models", {}).get(champ_id)

        if not champ_info:
            return {
                "title": "No Model Trained Yet",
                "markdown": (
                    "No champion model has been trained yet in this session! "
                    "Go to the **🤖 ML & Predictive Analytics → Training & AutoML** tab, click **Launch Automated Benchmark**, "
                    "and I will automatically analyze and explain the winner here."
                ),
                "voice_text": "No model has been trained yet. Please launch automated training in the Machine Learning tab, and I will explain the winner.",
            }

        m_name = champ_info.get("model_name", "Champion Model")
        m_id = champ_info.get("model_id", "")
        task = champ_info.get("task_type", "regression").replace("_", " ").title()
        target = champ_info.get("target_col", "Target")
        metrics = champ_info.get("metrics", {})
        top_feats = champ_info.get("feature_importance", [])

        # Construct layman breakdown
        enc_info = self.encyclopedia.get(m_id, {})
        analogy = enc_info.get("analogy", "A high-performance algorithmic pattern matching engine.")

        metric_highlights = []
        for k, v in list(metrics.items())[:3]:
            metric_highlights.append(f"- **{k.upper()}**: `{v:.4f}`" if isinstance(v, (int, float)) else f"- **{k.upper()}**: `{v}`")

        feat_str = ", ".join([f"`{f['feature']}`" for f in top_feats[:3]]) if top_feats else "numerical patterns"

        md = f"""### 🏆 Active Champion: **{m_name}**

**🎯 What it is predicting**: `{target}` ({task})

**💡 Why this model won**:
> *"{analogy}"*

- **Holdout Validation Scores**:
{chr(10).join(metric_highlights)}
- **Top Deciding Features**: {feat_str}

**💼 What this means for your business**:
This model successfully captured the underlying patterns in your dataset with low validation error. You can test live scenarios in the **Prediction** or **What-If** tabs to simulate decisions.
"""
        voice = (
            f"Your active champion model is {m_name}, predicting {target}. "
            f"It won because it achieved the lowest holdout error while remaining resistant to overfitting. "
            f"The top driving features influencing its decisions are {', '.join([f['feature'] for f in top_feats[:2]]) if top_feats else 'the primary columns'}. "
            f"You can now test What-If scenarios in the predictive cockpit."
        )

        return {"title": f"Champion: {m_name}", "markdown": md, "voice_text": self._clean_for_voice(voice)}

    def _explain_pca_in_plain_english(self) -> Dict[str, Any]:
        md = """### 🌐 Principal Component Analysis (PCA) Explained Simply

**💡 The Camera Analogy**:
> Imagine a 3D statue of a horse in a museum. If you take a photograph from the front, you might see just two legs and a nose. But if you walk around and take a photo from the side at a 45° angle, that single 2D photo shows the horse's head, body, all 4 legs, and tail!
> **That is PCA**: It finds the exact mathematical camera angle that captures the maximum detail in fewer dimensions.

**Why use it in Data Science?**:
1. **Dramatically speeds up training**: Turns 50 correlated columns into 3 principal components.
2. **2D and 3D Visualization**: You can't draw a 50-dimensional graph, but PCA lets you plot all your customers on a 2D map.
3. **Kills Multicollinearity**: Every principal component is 100% uncorrelated with every other component.
"""
        voice = (
            "Principal Component Analysis, or PCA, is like taking a photo of a 3D statue from the angle that captures the most detail. "
            "It squashes dozens of correlated columns down into 2 or 3 components so you can visualize high-dimensional data on a flat screen "
            "and speed up machine learning models while retaining 90 percent of the original variance."
        )
        return {"title": "PCA in Plain English", "markdown": md, "voice_text": voice}

    def _explain_anomaly_in_plain_english(self) -> Dict[str, Any]:
        md = """### 🚨 Anomaly & Outlier Detection Explained Simply

**💡 The Giraffe in a Crowd Analogy**:
> If you have a room full of 1,000 average-height people and 1 giraffe walks in, you don't need a ruler to measure everyone—the giraffe stands out instantly.
> **Isolation Forest** works by making random horizontal and vertical cuts. A normal person surrounded by a crowd takes 15 cuts to isolate. The giraffe is isolated on the very first cut!

**Top Business Applications**:
- **💳 Credit Card Fraud**: Someone buying a $10,000 diamond in Dubai when their card was used at a grocery store in New York 10 minutes earlier.
- **💻 Server Intrusions**: A sudden spike of 1,000 failed password attempts in 2 seconds.
- **🏭 Defective Parts**: A manufactured turbine vibrating at an irregular frequency.
"""
        voice = (
            "Anomaly detection algorithms like Isolation Forest find weird, irregular data points automatically. "
            "Think of it like spotting a giraffe in a crowd of sheep. Normal points are packed tightly together, "
            "while anomalies stand out in lonely regions. It is widely used for credit card fraud, server attacks, and defective product inspection."
        )
        return {"title": "Anomaly Detection Simply Explained", "markdown": md, "voice_text": voice}

    def _explain_forecasting_in_plain_english(self) -> Dict[str, Any]:
        md = """### 📈 Time-Series Forecasting Explained Simply

**💡 The Driving Analogy**:
> Driving forward while glancing in your rearview mirror:
> 1. **Trend**: Are you generally going uphill or downhill? (Overall growth or decline).
> 2. **Seasonality**: Do traffic jams always happen at 5:00 PM on Fridays? (Weekly/yearly cycles).
> 3. **Noise**: A temporary bump in the road (random day-to-day fluctuations).

**Our Platform's Forecasting Models**:
- **Holt-Winters**: Great for sales with holiday seasonality (like retail surges in November/December).
- **ARIMA / SARIMAX**: Rigorous statistical model analyzing historical momentum and errors.
- **ML Autoregression (RF / XGBoost)**: Turns time into tabular lag features (`yesterday`, `last_week`) to discover non-linear trends.
"""
        voice = (
            "Time series forecasting is like driving forward while looking in your rearview mirror. "
            "It breaks historical data down into three components: overall trend, repeating seasonal cycles like weekend surges, "
            "and random noise. Our platform benchmarks ARIMA, Holt-Winters, and machine learning lag models to find the winning trajectory."
        )
        return {"title": "Forecasting in Plain English", "markdown": md, "voice_text": voice}

    def _explain_metrics_in_plain_english(self, query: str) -> Dict[str, Any]:
        md = """### 📊 Machine Learning Metrics Explained in Plain English

#### 1. Classification Metrics (Yes/No predictions):
- **Accuracy**: What percentage of all guesses were correct? (Be careful: if 99% of people don't have cancer, a dumb model that always guesses 'No' has 99% accuracy but is totally useless!).
- **Precision**: When the model yells *"Fraud!"*, how often is it actually right? (High precision = few false alarms).
- **Recall**: Out of all the real fraud cases that happened, what percentage did the model catch? (High recall = catches every crook, but might trigger false alarms).
- **F1-Score**: The balanced harmonic mean between Precision and Recall.

#### 2. Regression Metrics (Number predictions):
- **R² (R-Squared)**: Scores from 0 to 1. A score of 0.85 means the model explains 85% of the swings and patterns in your data.
- **MAE (Mean Absolute Error)**: On average, how many dollars/units off is the model? (Very easy to explain to bosses: *"We are off by ±$12 on average"*).
- **RMSE**: Penalizes giant, disastrous mistakes more severely than small mistakes.
"""
        voice = (
            "Here is how to understand model metrics simply: Accuracy is total correct guesses. "
            "Precision measures how reliable the model is when it raises a red flag. "
            "Recall measures whether any red flags were missed. "
            "For predicting numbers, R-squared tells you what percentage of patterns the model captured, "
            "while MAE tells you how many dollars or units you are off by on average."
        )
        return {"title": "Metrics Explained", "markdown": md, "voice_text": voice}

    def _explain_all_models_summary(self) -> Dict[str, Any]:
        md = """### 📚 Complete Algorithm Catalog at a Glance

Our Quantum Analyst platform equips you with 27+ institutional algorithms across 6 core families:

1. **Linear Family** (`Linear`, `Ridge`, `Lasso`, `ElasticNet`, `Huber`, `Bayesian Ridge`):
   - Fast, transparent baselines providing exact unit multipliers.
2. **Decision Trees & Ensembles** (`Decision Tree`, `Random Forest`, `Extra Trees`):
   - Non-linear rules and democratic voting councils; handles messy data without scaling.
3. **Gradient Boosting** (`HistGradientBoosting`, `XGBoost`, `LightGBM`):
   - Sequential error-correction; industry champion for tabular business data.
4. **Distance & Support Vector** (`KNN`, `SVC`, `SVR`):
   - Geometric neighborhood voting and high-dimensional maximum-margin separation.
5. **Dimensionality & Unsupervised** (`PCA`, `SVD`, `ICA`, `NMF`, `t-SNE`, `K-Means`, `DBSCAN`):
   - Compression, visual clustering, and latent factor discovery.
6. **Anomaly Detection** (`Isolation Forest`, `LOF`, `One-Class SVM`, `Elliptic Envelope`):
   - Automated fraud, defect, and irregular transaction surveillance.
"""
        voice = (
            "Our platform includes 27 algorithms across six families: Linear models for fast transparent baselines, "
            "Random Forests for reliable ensembles, Gradient Boosting and XGBoost for maximum tabular accuracy, "
            "Support Vector Machines and KNN for geometric boundaries, PCA for dimension compression, "
            "and Isolation Forest for automated anomaly detection."
        )
        return {"title": "Algorithm Library Catalog", "markdown": md, "voice_text": voice}

    def _provide_default_model_guidance(self, df: Optional[pd.DataFrame], champ_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        dataset_str = f"your active dataset ({len(df):,} rows × {len(df.columns)} cols)" if df is not None else "your workspace data"
        md = f"""### 👋 Hello! I am your AI Model Copilot

I can explain any machine learning model in simple, everyday English and show you practical business use cases for {dataset_str}.

**Try asking me**:
- *"Explain all models in simple words"*
- *"Why did XGBoost win over Random Forest?"*
- *"Explain PCA like I am 10 years old"*
- *"What is the difference between Precision and Recall?"*
- *"How does Anomaly Detection work in real life?"*
- *"Which model is best for predicting sales?"*

Click any suggestion button below or type/speak your question! 🎙️
"""
        voice = (
            "Hello! I am your AI Model Copilot. You can ask me to explain any model in simple language, "
            "compare algorithms like Random Forest and XGBoost, or explain PCA and anomaly detection."
        )
        return {"title": "AI Model Copilot", "markdown": md, "voice_text": voice}

    def _clean_for_voice(self, text: str) -> str:
        """Removes markdown formatting characters to produce natural speech."""
        t = re.sub(r"\[.*?\]\(.*?\)", "", text)
        t = re.sub(r"[*#>`]", "", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()
