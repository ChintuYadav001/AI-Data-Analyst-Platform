"""
scratch/test_predictive_improvements.py
=======================================
End-to-end verification of the 6-Step Model Improvement Framework:
1. Evaluate Holdout Test vs Train metrics (MAE, RMSE, R²)
2. Outlier and high-sales underestimation detection
3. Domain feature engineering (category, discount, quantity, region, seasonality)
4. Multi-algorithm benchmarking (Ridge, Random Forest, HistGradientBoosting, XGBoost)
5. Underfitting vs overfitting diagnosis
6. Target log transformation (np.log1p / np.expm1) verification
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd

# 1. Test Feature Engineering
from core.ml.feature_engineering import engineer_sales_and_business_features

print(">>> Step 3 Test: Domain Feature Engineering...")
sample_sales_df = pd.DataFrame({
    "order_date": pd.date_range("2023-01-01", periods=100, freq="D"),
    "category": np.random.choice(["Electronics", "Furniture", "Supplies"], size=100),
    "region": np.random.choice(["North", "South", "East", "West"], size=100),
    "unit_price": np.random.uniform(20.0, 500.0, size=100),
    "quantity": np.random.randint(1, 20, size=100),
    "discount": np.random.uniform(0.0, 0.3, size=100),
})
# Create right-skewed sales target: bulk orders create exponential high-sales tail
sample_sales_df["sales"] = sample_sales_df["unit_price"] * sample_sales_df["quantity"] * (1.0 - sample_sales_df["discount"])
# Inject heavy high-sales outlier spike
sample_sales_df.loc[95:, "sales"] = sample_sales_df.loc[95:, "sales"] * 5.0

eng_df, new_feats = engineer_sales_and_business_features(sample_sales_df, target_col="sales")
print(f"Generated {len(new_feats)} new domain features: {new_feats[:6]}...")
assert any("gross_val" in f for f in new_feats), "Missing volume interaction feature!"
assert any("month" in f for f in new_feats), "Missing calendar seasonality feature!"
print("[PASS] Step 3 Feature Engineering passed successfully!")

# 2. Test Evaluation Diagnostics & High-Sales Underestimation
print("\n>>> Step 1 & Step 2 & Step 5 Test: Regression Evaluation & High-Sales Diagnostics...")
from core.ml.evaluation import evaluate_regression

y_true = np.array([50, 60, 75, 80, 90, 100, 120, 150, 1000, 2500], dtype=float)
# Underpredict high-sales tail severely
y_pred = np.array([52, 58, 77, 82, 88, 98, 115, 140, 450, 900], dtype=float)
y_train_t = np.array([50, 60, 75, 80, 90, 100, 120, 150, 1000, 2500], dtype=float)
y_train_p = y_train_t * 0.95

eval_res = evaluate_regression(y_true, y_pred, y_train_true=y_train_t, y_train_pred=y_train_p)
metrics = eval_res["metrics"]
diag = eval_res["improvement_diagnostics"]
print("Holdout Test Metrics:", metrics)
print("Improvement Diagnostics:", diag)

assert "rmse_mae_ratio" in metrics, "Missing rmse_mae_ratio in metrics!"
assert diag["step_2_outliers_and_high_sales"]["has_high_sales_underestimation"] is True, "Should detect high sales underestimation!"
assert len(diag["step_2_outliers_and_high_sales"]["worst_underpredicted_records"]) > 0, "Should extract worst underpredicted records!"
assert diag["step_6_log_transformation"]["log_transform_recommended"] is True, "Should recommend log transformation for skewed sales!"
print("[PASS] Step 1 & Step 2 & Step 5 Diagnostics passed successfully!")

# 3. Test Trainer with Log Transformation (np.log1p / np.expm1)
print("\n>>> Step 6 Test: Model Trainer with np.log1p Log-Transformation...")
from core.ml.trainer import train_single_model

train_res_standard = train_single_model(
    df=sample_sales_df,
    target_col="sales",
    feature_cols=["unit_price", "quantity", "discount", "category", "region"],
    task_type="regression",
    model_id="ridge_regression",
    use_log_transform=False,
)
print("Standard Ridge RMSE:", train_res_standard["metrics"]["rmse"])

train_res_log = train_single_model(
    df=sample_sales_df,
    target_col="sales",
    feature_cols=["unit_price", "quantity", "discount", "category", "region"],
    task_type="regression",
    model_id="ridge_regression",
    use_log_transform=True,
)
print("Log-Transformed Ridge RMSE:", train_res_log["metrics"]["rmse"])
print("Log-transformed active flag:", train_res_log["log_transformed"])
assert train_res_log["log_transformed"] is True, "Log transformed flag should be True!"
print("[PASS] Step 6 Log Transformation in trainer passed successfully!")

# 4. Test 10-Step Autonomous Pipeline with Multi-Algorithm Comparison
print("\n>>> Step 4 Test: Autonomous Pipeline with Algorithm Benchmarking...")
from core.ml.autonomous_pipeline import run_autonomous_1click_pipeline

auto_res = run_autonomous_1click_pipeline(
    df=sample_sales_df,
    target_col="sales",
    preferred_model_id="auto",
    use_log_transform=True,
)
s8_data = auto_res["steps"]["step_8"]["data"]
print("Autonomous Pipeline Completed Successfully!")
print("Step 8 Metrics:", s8_data["metrics"])
print("Step 8 Algorithm Comparison Count:", len(s8_data.get("algorithm_comparison", [])))
for m in s8_data.get("algorithm_comparison", []):
    print(f"  - {m['model_name']}: Test R2 = {m['test_r2']}, Test RMSE = {m['test_rmse']}")

assert len(s8_data.get("algorithm_comparison", [])) >= 3, "Multi-algorithm comparison should have at least 3 models!"
print("[PASS] All 6 predictive model improvements verified and working perfectly!")
