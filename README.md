# ⚡ AI-Powered End-to-End Data Analyst Platform

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ChintuYadav001/AI-Data-Analyst-Platform)

A production-quality web application and modular analytical engine designed as an **autonomous, explainable, and reproducible AI Data Analyst**.

---

## 🌟 Key Architecture & Modular Models

The project is architected into completely decoupled, dedicated Python models (`core/models/`) adhering strictly to data privacy and security principles:

1. **`DataProfilerModel` (`profiler_model.py`)**: 
   - Statistical profiling, missing value metrics, deep memory inspection.
   - Semantic data type classification (`Numerical`, `Categorical`, `Boolean`, `Date/Time`, `Identifier`, `Text`, `Potential Target Variable`).
   - Detection of suspicious columns (constant zero-variance, identifier leaks, excessive nulls).

2. **`DataQualityModel` (`quality_model.py`)**:
   - Automated health score (0–100) and letter grade (`Excellent`, `Good`, `Fair`, `Needs Attention`).
   - Granular deduction breakdown (missingness, duplicates, constant columns, outliers, category inconsistencies).

3. **`DataCleanerModel` (`cleaner_model.py`)**:
   - Reversible, non-destructive data cleaning engine.
   - Missing value imputation (Mean, Median, Mode, Constant, Forward Fill).
   - Outlier detection via IQR and Z-Score with user actions: `Keep`, `Remove`, `Cap (Winsorize)`, or `Flag`.
   - Category case standardization (Title Case, Lowercase, whitespace trimming).
   - Immutable audit log tracking every operation (`Issue | Column | Count | Action`).

4. **`EDAModel` (`eda_model.py`)**:
   - Numerical summary (Mean, Median, Mode, Variance, Quartiles, Skewness).
   - Categorical distributions, top categories, and rare tail identification (< 2%).
   - Pearson correlation matrix with automated strong positive ($r \ge 0.6$) and strong negative ($r \le -0.6$) linkage extraction.
   - Temporal aggregation (Daily, Monthly trends).

5. **`ChartModel` (`chart_model.py`)**:
   - Intelligent chart recommendation engine mapping data types to best-practice visualizations.
   - Interactive Plotly figures (Line, Bar, Scatter with OLS trendline, Histogram, Donut, Heatmap).

6. **`FeatureEngineeringModel` (`feature_model.py`)**:
   - Dataset-aware feature extraction (Year, Month, Day of Week, Weekend Flag).
   - Business metric derivation (`profit_margin = profit / revenue`).
   - Customer analytics (`customer_total_spend` RFM).
   - Numerical transformations (Log transform `np.log1p`, Quantile binning).
   - User approval workflow with explicit formulas and expected analytical value.

7. **`SQLAnalyticsModel` (`sql_model.py`)**:
   - DuckDB in-memory analytical query engine.
   - **SQL Security Guardrail**: Enforces strict read-only execution. Blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, and multi-statement attacks.
   - Query execution timer and automatic chart generation on SQL results.

8. **`AIInsightsModel` (`ai_insights_model.py`)**:
   - Privacy-preserving analytical synthesis (operates on statistical summaries, never transmitting raw customer rows).
   - Produces Key Insights, Business Risks, Trends, Anomalies, and Actionable Recommendations.
   - Strict anti-hallucination guarantee: every finding is traceable to calculated metrics.

9. **`AIChatModel` (`chat_model.py`)**:
   - Conversational AI agent with controlled backend tool invocation (`execute_sql()`, `profile_dataset()`, `detect_outliers()`, `recommend_chart()`).
   - Natural language to SQL translation, execution, and visual responses.

10. **`DashboardModel` (`dashboard_model.py`)**:
    - Automated KPI calculation (Volume, Revenue, Profit, Margin, Unique Entities).
    - Multi-dimensional dynamic cross-filtering (Region, Category, Segment).

11. **`ReportModel` (`report_model.py`)**:
    - Compiles a 10-section executive analytical report.
    - One-click export to PDF (via ReportLab) and Markdown.

12. **`ProjectModel` (`project_model.py`)**:
    - Multi-project persistence (preserves raw dataset, cleaned dataset, audit history, queries, and reports across sessions).

---

## 🚀 Quick Start & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web UI
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. (Optional) Run the FastAPI Backend Server
```bash
python -m uvicorn main:app --port 8000 --reload
```
Interactive Swagger API documentation available at `http://localhost:8000/docs`.

### 4. Run Automated Test Suite
```bash
python -m pytest tests/
```
All unit tests verify model calculations, cleaning operations, SQL security guardrails, and API endpoints.

---

## 🖥️ Application Navigation Structure

1. **Dashboard**: High-level KPI summary cards and priority charts.
2. **Upload Dataset**: CSV, XLSX, XLS file uploader + built-in sample datasets (E-Commerce Sales, Customer Churn).
3. **Data Profile**: Memory footprint, row/column counts, semantic data typing, and suspicious column detection.
4. **Data Quality**: 0–100 dataset health score with transparent deduction breakdown.
5. **Data Cleaning**: Interactive cleaning workbench, outlier management, audit log, and revert capabilities.
6. **EDA**: Deep exploratory analysis, histograms, correlations, and time trends.
7. **Feature Engineering**: Feature recommendations, formulas, and one-click creation.
8. **SQL Analytics**: DuckDB SQL workbench, execution timing, and auto-visualization.
9. **AI Insights**: Executive summary, business problems, anomalies, and recommendations.
10. **Dashboard Builder**: Customizable KPI cards, chart selector, and dynamic dimension filters.
11. **AI Chat**: Conversational AI Data Analyst assistant with transparent tool calling.
12. **Reports**: 10-section analytical report with one-click PDF generation.
13. **Downloads**: Cleaned CSV/Excel, original dataset, Python scripts, Jupyter notebooks, and logs.
14. **Projects**: Multi-project switcher and dataset version manager.
15. **Settings**: API key management (Gemini / OpenAI), offline mode, and security guarantees.
