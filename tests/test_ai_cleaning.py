import pytest
import pandas as pd
import numpy as np

from core.cleaning.orchestrator import AICleaningOrchestrator
from core.models.cleaner_model import DataCleanerModel


@pytest.fixture
def dirty_dataset():
    """Generates a dataset with deliberate defects across all 12 cleaning categories."""
    return pd.DataFrame({
        "Unnamed: 0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Empty_Col": [None] * 10,
        "Customer ID": [f"ID_{i}" for i in range(10)],
        "  User Name  ": ["  Alice ", "BOB", "charlie  ", "Alice", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan"],
        "City Name": ["new york", "New York", "NEW YORK", "chicago", "Chicago", "boston", "Boston", "austin", "Austin", "dallas"],
        "Gender": ["Male", "m", "M", "Female", "f", "female", "Male", "Female", "M", "F"],
        "Income ($)": ["$45,000", "$52,000", "$61,000", "$48,000", "$75,000", "$50,000", "$63,000", "$58,000", "$70,000", "$55,000"],
        "Discount Rate": ["10%", "15.5%", "20%", "5%", "0%", "12%", "25%", "8%", "15%", "10%"],
        "Age": [25, 34, -5, 142, 29, 41, 38, 50, 22, 30],
        "Customer_Rating": [4.5, 5.0, 6.5, 3.8, 4.2, 0.5, 4.0, 3.5, 4.8, 5.0],
        "Order_Quantity": [2, 5, -3, 10, 1, 4, -1, 3, 2, 6],
        "Transaction Date": ["2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09", "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2023-01-14"],
        "Review_HTML": ["<p>Great service!</p>", "Good product", "<b>Fast delivery</b>", "Okay", "Loved it", "Nice", "Decent", "<span>Superb</span>", "Fine", "Recommended"],
        "Category": ["Electronics", "Electronics", "Clothing", "Clothing", "Home", "Electronics", "Clothing", "RareItem1", "RareItem2", "Electronics"],
        "Revenue": [100.0, 120.0, 110.0, 95.0, 105.0, 115.0, 102.0, 108.0, 112.0, 9500.0],  # 9500 is extreme outlier
        "Score_Missing": [85.0, 90.0, np.nan, 88.0, np.nan, 92.0, 89.0, 91.0, 87.0, 93.0]
    })


def test_orchestrator_scans_all_categories(dirty_dataset):
    """Verifies that AICleaningOrchestrator detects issues across all 12 cleaning categories."""
    plan = AICleaningOrchestrator.scan_dataset(dirty_dataset)
    assert "plan_items" in plan
    assert "summary" in plan
    assert plan["summary"]["total_issues"] > 0
    assert plan["summary"]["safe_fixes"] > 0
    assert plan["summary"]["review_required"] > 0

    detected_categories = set(plan["summary"]["categories_detected"])
    # Should detect column cleaning, missing values, inconsistent values, etc.
    assert "Column Cleaning" in detected_categories
    assert "Missing Values" in detected_categories
    assert "Inconsistent Values" in detected_categories
    assert "Invalid / Impossible Values" in detected_categories
    assert "Outlier Detection" in detected_categories
    assert "Numerical Data Cleaning" in detected_categories
    assert "Text Cleaning" in detected_categories


def test_confidence_scoring_and_action_classification(dirty_dataset):
    """Verifies that items receive appropriate confidence scores and action levels."""
    plan = AICleaningOrchestrator.scan_dataset(dirty_dataset)
    items = plan["plan_items"]

    for item in items:
        assert "confidence" in item
        assert 0.0 <= item["confidence"] <= 1.0
        assert item["action"] in ("AUTO_FIX", "REVIEW", "KEEP", "DO_NOT_CHANGE")
        assert "severity" in item
        assert item["severity"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
        assert "reason" in item
        assert len(item["reason"]) > 0

    # Outliers should default to REVIEW or KEEP (never destructive AUTO_FIX)
    outlier_items = [i for i in items if i.get("category") == "Outlier Detection"]
    for oi in outlier_items:
        assert oi["action"] == "REVIEW"
        assert "Flag" in oi["default_option"] or "Keep" in oi["default_option"]


def test_auto_clean_executes_safe_fixes(dirty_dataset):
    """Verifies 1-Click Auto-Clean resolves safe fixes non-destructively."""
    cleaned, log = AICleaningOrchestrator.apply_auto_clean(dirty_dataset)
    
    # 1. Unnamed and empty columns dropped
    assert "Unnamed: 0" not in cleaned.columns
    assert "Empty_Col" not in cleaned.columns

    # 2. Customer ID identifier preserved
    assert any("customer" in c.lower() and "id" in c.lower() for c in cleaned.columns)

    # 3. Currency and percentages converted to numeric
    inc_col = [c for c in cleaned.columns if "income" in c.lower()][0]
    assert pd.api.types.is_numeric_dtype(cleaned[inc_col])

    # 4. HTML tags stripped
    rev_col = [c for c in cleaned.columns if "html" in c.lower() or "review" in c.lower()][0]
    assert not cleaned[rev_col].str.contains("<p>|</p>|<b>|</b>").any()

    # 5. Missing values in Score_Missing imputed
    score_col = [c for c in cleaned.columns if "score_missing" in c.lower()][0]
    assert cleaned[score_col].isna().sum() == 0

    # 6. Audit log populated
    assert len(log) > 0
    assert all("operation" in entry for entry in log)
    assert all("status" in entry for entry in log)


def test_before_after_sample_diff(dirty_dataset):
    """Verifies generation of cell-level diffs between before and after DataFrames."""
    cleaned, log = AICleaningOrchestrator.apply_auto_clean(dirty_dataset)
    diff = AICleaningOrchestrator.generate_sample_diff(dirty_dataset, cleaned, max_rows=5)

    assert "total_changed_rows" in diff
    assert "sample_diffs" in diff
    assert diff["total_changed_rows"] > 0
    assert len(diff["sample_diffs"]) > 0

    first_diff = diff["sample_diffs"][0]
    assert "row_index" in first_diff
    assert "changes" in first_diff
    assert len(first_diff["changes"]) > 0


def test_post_cleaning_validation_report(dirty_dataset):
    """Verifies that post-cleaning validation accurately measures quality uplift."""
    cleaned, log = AICleaningOrchestrator.apply_auto_clean(dirty_dataset)
    val = AICleaningOrchestrator.validate_cleaning(dirty_dataset, cleaned)

    assert "score_before" in val
    assert "score_after" in val
    assert "score_delta" in val
    assert val["score_after"] >= val["score_before"]
    assert "validations" in val
    assert len(val["validations"]) >= 4
    assert any("Missing" in v["label"] for v in val["validations"])
    assert any("Retention" in v["label"] for v in val["validations"])


def test_datacleaner_model_backward_compatibility(dirty_dataset):
    """Verifies that DataCleanerModel exposes both legacy and orchestrator methods."""
    # Legacy recommendation
    recs = DataCleanerModel.recommend_cleaning_plan(dirty_dataset)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert "issue" in recs[0]
    assert "column" in recs[0]
    assert "count" in recs[0]

    # Legacy action execution
    actions = [{"action": "impute_median", "column": "Score_Missing"}]
    c_df, c_log = DataCleanerModel.apply_cleaning_actions(dirty_dataset, actions)
    assert c_df["Score_Missing"].isna().sum() == 0
    assert len(c_log) > 0
    assert "issue" in c_log[0]
    assert "action" in c_log[0]

    # Orchestrator methods on DataCleanerModel
    scan = DataCleanerModel.scan_dataset(dirty_dataset)
    assert "plan_items" in scan
    val = DataCleanerModel.validate_cleaning(dirty_dataset, c_df)
    assert "score_after" in val
    diff = DataCleanerModel.generate_sample_diff(dirty_dataset, c_df)
    assert "total_changed_rows" in diff


def test_categorical_column_safe_imputation():
    """Verifies that pd.Categorical columns with missing values can be safely imputed without TypeError."""
    cat_series = pd.Series(pd.Categorical(["Low", "Medium", np.nan, "High", np.nan]))
    df = pd.DataFrame({"Rating_Tier": cat_series, "Val": [1, 2, 3, 4, 5]})

    cleaned, log = DataCleanerModel.auto_clean(df)
    # The column may be lowercased to rating_tier
    col = [c for c in cleaned.columns if "rating" in c.lower()][0]
    assert cleaned[col].isna().sum() == 0
    assert "Unknown" in list(cleaned[col])


def test_numeric_string_with_nulls_parquet_serialization(tmp_path):
    """Verifies that numeric values stored as strings with nulls are cleanly handled and serialize to parquet without ArrowInvalid."""
    df = pd.DataFrame({
        "Max_BPM": ["174", "166", None, "187", "177", np.nan, "195"],
        "Age": [25, 30, 35, 40, 45, 50, 55],
        "Category": ["Cardio", "Strength", None, "Yoga", "Cardio", "Cardio", "Yoga"]
    })

    cleaned, log = DataCleanerModel.auto_clean(df)
    target_col = [c for c in cleaned.columns if "max_bpm" in c.lower()][0]
    assert cleaned[target_col].isna().sum() == 0
    assert pd.api.types.is_numeric_dtype(cleaned[target_col])
    assert not any(isinstance(x, str) for x in cleaned[target_col])

    # Test parquet export
    pq_path = tmp_path / "test_out.parquet"
    from core.models.project_model import ProjectModel
    clean_ready = ProjectModel._prepare_df_for_parquet(cleaned)
    clean_ready.to_parquet(pq_path, index=False)
    assert pq_path.exists()

    reloaded = pd.read_parquet(pq_path)
    assert len(reloaded) == len(df)
    assert pd.api.types.is_numeric_dtype(reloaded[target_col])


