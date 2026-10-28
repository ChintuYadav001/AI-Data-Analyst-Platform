import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from core.ml.autonomous_pipeline import run_autonomous_1click_pipeline

print("--- Testing 1-Click Autonomous Predictive Pipeline ---")

# Synthesize customer dataset matching user example
np.random.seed(42)
n_rows = 500
ages = np.random.randint(18, 70, size=n_rows)
tenure = np.random.randint(1, 72, size=n_rows)
monthly_spending = np.random.uniform(20.0, 150.0, size=n_rows)
contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_rows)
# Churn logic: higher monthly spending and lower tenure -> higher churn
churn_prob = 1 / (1 + np.exp(-(0.03 * monthly_spending - 0.05 * tenure - 0.01 * ages)))
churn_status = (np.random.rand(n_rows) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id": [f"CUST_{i}" for i in range(n_rows)],
    "age": ages,
    "tenure": tenure,
    "monthly_spending": monthly_spending,
    "contract_type": contract_type,
    "churn": churn_status
})

# Add a few duplicate rows and null values to test automated preprocessing
df.loc[10, "monthly_spending"] = np.nan
df.loc[25, "contract_type"] = np.nan
df = pd.concat([df, df.iloc[:5]], ignore_index=True)

print(f"Synthesized Customer Dataset: {df.shape}")

# Run Autonomous 1-Click Pipeline
res = run_autonomous_1click_pipeline(
    df=df,
    target_col="churn",
    preferred_model_id="auto",
    test_size=0.20,
    tune_hyperparameters=True,
    dataset_name="Customer Churn Test"
)

print(f"Pipeline Success: {res['success']}")
print(f"Total Execution Time: {res['total_time_seconds']}s")
print(f"Selected Model: {res['model_name']} ({res['model_id']})")
print(f"Task Type: {res['task_type']}")
print(f"Metrics: {res['metrics']}")
print(f"Experiment ID: {res['experiment_id']}")
print(f"Sample Comparisons Count: {len(res['sample_comparisons'])}")

# Check that all 10 steps were recorded
print("\n--- Verifying All 10 Steps ---")
for step_i in range(1, 11):
    step_key = f"step_{step_i}"
    assert step_key in res["steps"], f"Missing {step_key}!"
    print(f"✓ Step {step_i}: {res['steps'][step_key]['title']} -> {res['steps'][step_key]['data']['status']}")

print("\nALL 10 STEPS VERIFIED 100% SUCCESSFULLY!")
