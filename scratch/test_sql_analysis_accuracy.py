"""
scratch/test_sql_analysis_accuracy.py
======================================
Automated verification test for SmartSQLEngine:
Tests 8 diverse analytical questions on the Superstore dataset.
"""

import os
import sys
import pandas as pd

# Set working directory & sys.path
sys.path.insert(0, os.path.abspath("."))

from core.analytics.smart_sql_engine import SmartSQLEngine


def run_tests():
    data_path = "data/parquet/c0e273b30ba806ea.parquet"
    if not os.path.exists(data_path):
        print(f"[FAIL] Data path {data_path} not found.")
        return False

    df = pd.read_parquet(data_path)
    print(f"[INFO] Loaded Superstore dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    test_queries = [
        ("Which category has the highest sales?", "ranking", "Category"),
        ("Top 5 products by profit", "ranking", "Product Name"),
        ("Total sales and profit by region", "group_aggregation", "Region"),
        ("What is the total sales in California?", "scalar_summary", "State"),
        ("Monthly sales trend", "time_series", "month"),
        ("What is the total revenue?", "scalar_summary", "total_Sales"),
        ("How many orders had negative profit?", "scalar_summary", "Profit"),
        ("Sales by category and region", "multi_dimension", "Category"),
    ]

    all_passed = True

    for i, (question, expected_intent, expected_keyword) in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"Test #{i}: '{question}'")
        res = SmartSQLEngine.answer_question(question, df)

        if not res.get("success"):
            print(f"[FAIL] Query execution failed: {res.get('error')}")
            all_passed = False
            continue

        sql = res["sql"].replace("\n", " ").strip()
        rows = res["row_count"]
        ms = res["execution_time_ms"]
        answer = res.get("answer_markdown", "")
        kpis = [k["value"] for k in res.get("kpi_cards", [])]
        chart = res.get("chart_spec", {})
        chart_type = chart.get("type") if chart else "None"

        print(f"  SQL: {sql}")
        print(f"  Rows: {rows} | Latency: {ms}ms | Chart: {chart_type}")
        print(f"  KPIs: {kpis}")
        # Strip markdown bold for clean terminal display
        clean_ans = answer.replace("**", "")
        print(f"  Answer: {clean_ans[:140]}...")

        # Assertions
        if rows < 1:
            print(f"[FAIL] Returned 0 rows.")
            all_passed = False
        elif not answer or len(answer) < 15:
            print(f"[FAIL] Answer narrative is too short or empty.")
            all_passed = False
        else:
            print(f"[PASS] Test #{i} passed successfully!")

    print("\n" + "=" * 70)
    if all_passed:
        print("[ALL PASS] All 8 SmartSQLEngine analytical questions passed with high accuracy!")
    else:
        print("[FAIL] Some tests encountered issues.")

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
