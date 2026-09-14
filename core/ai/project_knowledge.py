"""
core.ai.project_knowledge
=========================
Authoritative, verified knowledge base for Quantum Analyst.
Provides factual answers regarding the platform architecture, technologies,
end-to-end analytical workflows, and simple-language explanations with analogies.
Distinguishes strictly between:
- PROJECT_INFO: Architectural facts, features, and technology stack.
- GENERAL_KNOWLEDGE: Machine learning concepts, statistical definitions, and simple analogies.
- DATASET_INFO: Specific metadata and observations from the uploaded data.
- ANALYTICAL_RESULT: Computed outputs from statistical and predictive engines.
"""

from typing import Dict, Any, List, Optional, Tuple
import re

CATEGORY_PROJECT_INFO = "PROJECT_INFO"
CATEGORY_GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
CATEGORY_DATASET_INFO = "DATASET_INFO"
CATEGORY_ANALYTICAL_RESULT = "ANALYTICAL_RESULT"

PROJECT_OVERVIEW = {
    "name": "Quantum Analyst",
    "tagline": "AI-Powered Intelligent Data Analytics and Predictive Analytics Platform",
    "mission": (
        "To provide an automated, end-to-end analytical and machine learning platform that empowers "
        "business analysts, data scientists, and executives to ingest, profile, clean, analyze, "
        "model, and forecast tabular data at enterprise scale with zero mathematical hallucination."
    ),
    "core_principle": "Data engines calculate. ML models predict. AI explains.",
    "supported_file_types": ["CSV (.csv)", "Excel (.xlsx, .xls)", "Apache Parquet (.parquet)"],
    "scale_capability": "Optimized for high-performance handling of datasets from small files up to 250,000+ rows in sub-second time via DuckDB columnar execution.",
}

TECHNOLOGY_STACK = {
    "frontend": "Streamlit (Reactive Enterprise UI with Cyberpunk Dark & Clean Light design modes)",
    "backend_api": "FastAPI (Asynchronous, high-concurrency RESTful API service on port 8000)",
    "sql_engine": "DuckDB (In-process columnar vectorized SQL OLAP engine)",
    "storage_format": "Apache Parquet (Columnar binary storage with snappy compression)",
    "ml_frameworks": ["Scikit-learn", "XGBoost", "LightGBM", "SciPy", "NumPy", "Pandas"],
    "visualization": "Plotly (Hardware-accelerated 2D/3D charts and financial overlays)",
    "reporting": "ReportLab (Vector PDF synthesis) & HTML5/CSV exporters",
    "voice_tech": "Web Speech API (Browser-native SpeechRecognition STT & SpeechSynthesis TTS)",
    "database": "DuckDB for analytics; SQLite / MySQL / PostgreSQL for auth and metadata",
}

WORKFLOW_STAGES = [
    {
        "stage": 1,
        "name": "Data Ingestion & Validation",
        "route": "Upload Dataset",
        "description": "Upload CSV, Excel, or Parquet files. Automatic encoding detection, schema inference, and Parquet caching."
    },
    {
        "stage": 2,
        "name": "Dataset Profiling",
        "route": "Data Profile",
        "description": "Sub-millisecond structural profiling: dimensions, column semantic types, memory footprint, null counts, and cardinality."
    },
    {
        "stage": 3,
        "name": "Data Quality Audit",
        "route": "Data Quality",
        "description": "Rigorous 0-100 quality score with dimension breakdowns (completeness, uniqueness, validity, consistency, accuracy, integrity)."
    },
    {
        "stage": 4,
        "name": "Intelligent Cleaning & Healing",
        "route": "Data Cleaning",
        "description": "12-category non-destructive cleaning plan: missing value imputation, duplicate removal, outlier handling, type casting, and audit logging."
    },
    {
        "stage": 5,
        "name": "Exploratory Data Analysis (EDA)",
        "route": "EDA",
        "description": "14-stage adaptive EDA: univariate distributions, bivariate cross-tabs, correlation matrices, key drivers, and 3-tier chart recommendations."
    },
    {
        "stage": 6,
        "name": "Feature Engineering & Selection",
        "route": "Feature Engineering",
        "description": "Power transforms (Yeo-Johnson), interaction crosses, binning, and 7 statistical feature selection methods (Mutual Info, RFE, Lasso)."
    },
    {
        "stage": 7,
        "name": "Natural Language SQL Analytics",
        "route": "SQL Analytics",
        "description": "Natural language to SQL conversion with AST security validation (read-only SELECT enforcement) executed directly via DuckDB."
    },
    {
        "stage": 8,
        "name": "Machine Learning & AutoML Cockpit",
        "route": "Machine Learning",
        "description": "13-tab ML cockpit: automatic task classification (regression/classification), 27+ candidate algorithms, cross-validated benchmarking, and champion registration."
    },
    {
        "stage": 9,
        "name": "Explainable AI & Diagnostics",
        "route": "Machine Learning",
        "description": "Global feature importance, local prediction attribution, confusion matrices, ROC curves, residual diagnostics, and overfitting alerts."
    },
    {
        "stage": 10,
        "name": "Dimensionality Reduction & PCA",
        "route": "Machine Learning",
        "description": "PCA, Truncated SVD, Kernel PCA, ICA, and t-SNE with scree plots, cumulative variance curves, and 2D/3D component projections."
    },
    {
        "stage": 11,
        "name": "Future Time-Series Forecasting",
        "route": "Machine Learning",
        "description": "Automated stationarity testing (ADF), seasonality checks, and multi-model competition (ARIMA, Holt-Winters, Moving Average) with 95% confidence bands."
    },
    {
        "stage": 12,
        "name": "What-If Scenario Simulation",
        "route": "Machine Learning",
        "description": "Interactive counterfactual analysis: adjust feature values with sliders to simulate real-time impact on predictions and feature elasticities."
    },
    {
        "stage": 13,
        "name": "AI Strategic Insights",
        "route": "AI Insights",
        "description": "Descriptive, diagnostic, predictive, and prescriptive narrative intelligence grounded strictly in verified statistical calculations."
    },
    {
        "stage": 14,
        "name": "Dashboard & Reporting",
        "route": "Dashboard Builder / Reports",
        "description": "Dynamic KPI tracking, executive PDF reports, CSV export packages, and interactive drag-and-drop dashboard builders."
    }
]

PROJECT_QA_REGISTRY: List[Dict[str, Any]] = [
    {
        "keywords": ["what does this project do", "explain quantum analyst", "what is quantum analyst", "overview", "project summary", "about project"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "What does Quantum Analyst do?",
        "simple_answer": (
            "Quantum Analyst is an AI-powered automated data analytics and predictive analytics platform. "
            "It takes your raw data (CSV or Excel) and guides you through the entire data science journey: "
            "checking data quality, cleaning errors, discovering patterns through EDA, building machine learning models, "
            "forecasting future numbers, and generating executive reports—all with natural language and voice assistance."
        ),
        "why_used": [
            "Transforms raw spreadsheets into production-ready predictive models without manual coding",
            "Guarantees zero mathematical hallucination by calculating all numbers via real database engines. Core principle: Data engines calculate. ML models predict. AI explains.",
            "Provides an AI voice copilot that explains complex statistics in plain English"
        ],
        "technologies": ["DuckDB", "Streamlit", "FastAPI", "Scikit-learn", "XGBoost", "Plotly"]
    },
    {
        "keywords": ["simple language", "explain this project in simple language", "layman", "for beginner", "explain simply"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "Can you explain Quantum Analyst in very simple language?",
        "simple_answer": (
            "Imagine having an expert data scientist, a database engineer, and a business analyst sitting next to you. "
            "You drop your spreadsheet on the desk. Quantum Analyst automatically reads it, points out dirty rows or mistakes, "
            "shows you colorful charts that explain what is happening, tests 10+ AI prediction models to find the champion, "
            "and tells you what will happen next month—answering your questions in plain words or out loud with voice!"
        ),
        "why_used": [
            "No coding required: point, click, ask, or speak",
            "Automates hours of tedious data cleaning and model tuning in seconds",
            "Always gives honest, real calculations without guessing"
        ]
    },
    {
        "keywords": ["eda", "what is eda", "exploratory data analysis"],
        "category": CATEGORY_GENERAL_KNOWLEDGE,
        "question": "What is EDA (Exploratory Data Analysis)?",
        "simple_answer": (
            "EDA is like a doctor giving your data a thorough medical check-up before prescribing treatment. "
            "Instead of jumping straight into predictions, EDA looks at charts, averages, correlations, and outliers "
            "to understand what your data looks like and uncover hidden patterns."
        ),
        "analogy": "Like inspecting all ingredients and tasting the soup before serving dinner to guests.",
        "why_used": [
            "Discovers unexpected trends and correlations between variables",
            "Identifies extreme outliers and anomalies that could warp predictions",
            "Validates hypotheses before training machine learning models"
        ]
    },
    {
        "keywords": ["feature engineering", "what is feature engineering", "features"],
        "category": CATEGORY_GENERAL_KNOWLEDGE,
        "question": "What is Feature Engineering?",
        "simple_answer": (
            "Feature engineering is the art of transforming raw columns into smarter, more informative signals "
            "that make it easier for machine learning models to learn and make accurate predictions."
        ),
        "analogy": "Like chopping raw vegetables and adding spices so the cooking process produces a delicious meal instead of raw ingredients.",
        "examples": [
            "Turning a 'Birth Date' column into an 'Age' column",
            "Combining 'Total Spend' and 'Visit Count' to create 'Average Order Value'",
            "Transforming skewed financial numbers using power transforms (Yeo-Johnson) so distributions are balanced"
        ],
        "why_used": [
            "Significantly boosts predictive accuracy without changing algorithms",
            "Helps linear models capture non-linear relationships through interaction terms",
            "Standardizes disparate scales so large numbers don't overpower small ones"
        ]
    },
    {
        "keywords": ["predictive analytics", "what is predictive analytics"],
        "category": CATEGORY_GENERAL_KNOWLEDGE,
        "question": "What is Predictive Analytics?",
        "simple_answer": (
            "Predictive analytics uses historical data to estimate what is likely to happen in the future. "
            "It finds patterns in past behavior—like customer purchases, machine sensor readings, or financial trends—"
            "and uses them to predict outcomes for new, unseen data."
        ),
        "analogy": "Like an experienced weather forecaster who studies atmospheric pressure and cloud formations over years to predict tomorrow's rain with high confidence.",
        "why_used": [
            "Anticipate customer churn before they cancel subscriptions",
            "Forecast inventory demand to prevent stockouts",
            "Score loan applications to minimize default risk"
        ]
    },
    {
        "keywords": ["pca", "what is pca", "principal component analysis", "dimensionality reduction"],
        "category": CATEGORY_GENERAL_KNOWLEDGE,
        "question": "What is PCA (Principal Component Analysis)?",
        "simple_answer": (
            "PCA is a smart mathematical technique that compresses many columns into just a few essential components "
            "while retaining most of the useful information and variance."
        ),
        "analogy": "Taking a 2D photograph of a 3D statue. You lose one dimension of depth, but anyone looking at the photo can still clearly recognize the statue and its features.",
        "why_used": [
            "Reduces 50 or 100 features down to 2 or 3 so you can plot them on a scatter chart",
            "Removes multicollinearity and redundant variables that confuse models",
            "Speeds up model training significantly on large datasets"
        ]
    },
    {
        "keywords": ["random forest", "what is random forest"],
        "category": CATEGORY_GENERAL_KNOWLEDGE,
        "question": "What is Random Forest?",
        "simple_answer": (
            "Random Forest is an ensemble machine learning model that combines the votes of dozens or hundreds "
            "of independent decision trees to make a final, highly accurate prediction."
        ),
        "analogy": "Instead of trusting one doctor's opinion, you ask a committee of 100 diverse medical specialists and take the majority consensus.",
        "why_used": [
            "Rarely overfits because individual tree errors cancel each other out",
            "Handles both numerical numbers and categories effortlessly",
            "Provides clear feature importance scores showing which variables drove decisions"
        ]
    },
    {
        "keywords": ["regression vs classification", "difference between regression and classification", "classification vs regression"],
        "category": CATEGORY_GENERAL_KNOWLEDGE,
        "question": "What is the difference between Regression and Classification?",
        "simple_answer": (
            "The difference comes down to the question you are asking: Regression predicts a continuous NUMBER, "
            "while Classification predicts a discrete CATEGORY or LABEL."
        ),
        "examples": [
            "Regression: 'How many dollars will this customer spend?' or 'What will the house price be?' ($250,000, $412,500)",
            "Classification: 'Will this customer cancel?' (Yes or No) or 'Is this transaction fraudulent?' (Fraud or Legitimate)"
        ],
        "why_used": [
            "Quantum Analyst automatically audits your target column and configures the correct pipeline for you"
        ]
    },
    {
        "keywords": ["how does ai work in this project", "how ai works", "ai integration"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "How does AI work in Quantum Analyst?",
        "simple_answer": (
            "AI in Quantum Analyst operates under a strict rule: 'Data engines calculate. ML models predict. AI explains.' "
            "1. Heavy computations, statistics, and SQL are executed by DuckDB and Scikit-learn on real data. "
            "2. Machine learning models generate genuine predictions and benchmark metrics. "
            "3. The AI reasoning engine ingests verified statistics and explains them "
            "in clear business language, answering questions and providing strategic recommendations with zero hallucination."
        ),
        "why_used": [
            "Eliminates LLM math errors because the database calculates exact values",
            "Keeps private raw data safe by only sending compact statistical context to AI",
            "Provides both visual charts on screen and natural voice audio explanations"
        ]
    },
    {
        "keywords": ["forecasting", "how does forecasting work", "time series", "arima"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "How does the forecasting system work in Quantum Analyst?",
        "simple_answer": (
            "The forecasting engine automatically analyzes your temporal date column and target variable. "
            "It checks for regular date spacing, tests whether the series is stationary (ADF test), "
            "identifies seasonal patterns, and runs an automated competition between multiple forecasting algorithms: "
            "Naive baseline, Seasonal Naive, Moving Average, Holt-Winters Exponential Smoothing, and ARIMA. "
            "It selects the model with the lowest Mean Absolute Error (MAE) and projects future points with 90% and 95% confidence intervals."
        ),
        "why_used": [
            "Provides realistic upper and lower bounds so you know the uncertainty range",
            "Automatic date frequency detection (Daily, Weekly, Monthly, Yearly)",
            "Interactive Plotly charts allow zooming into future trajectory windows"
        ]
    },
    {
        "keywords": ["technologies", "tech stack", "tools used", "what technologies"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "What technologies are used in Quantum Analyst?",
        "simple_answer": (
            "Quantum Analyst is built on a modern, high-performance Python analytics stack: "
            "• DuckDB: In-process columnar SQL database for blazing-fast OLAP queries on 250k+ rows. "
            "• Apache Parquet: Binary columnar cache format with snappy compression. "
            "• FastAPI: Production REST API handling authentication, uploads, and asynchronous jobs. "
            "• Streamlit: Interactive, real-time UI with Cyberpunk Dark & Clean Light design modes. "
            "• Scikit-learn, XGBoost & LightGBM: Institutional machine learning and AutoML engine. "
            "• Plotly: High-resolution interactive charts and 3D PCA visualizers. "
            "• Web Speech API: Native browser voice recognition (STT) and voice synthesis (TTS)."
        )
    },
    {
        "keywords": ["architecture", "explain architecture", "system design"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "How is the Quantum Analyst architecture designed?",
        "simple_answer": (
            "The architecture follows a decoupled, 4-tier design: "
            "1. Storage & Ingestion Tier: Accepts CSV/Excel, converts to compressed Parquet, registers tables in DuckDB. "
            "2. Analytical Engine Tier: High-speed profiler, data quality scorer, 12-category cleaner, and 14-stage EDA engine. "
            "3. ML & Predictive Cockpit Tier: AutoML training, hyperparameter tuning, explainability, forecasting, and What-If scenario simulation. "
            "4. Presentation & Voice Tier: Streamlit UI with unified top navigation, left sidebar, and persistent bottom-right AI Voice Copilot."
        )
    },
    {
        "keywords": ["why duckdb", "duckdb"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "Why is DuckDB used in Quantum Analyst?",
        "simple_answer": (
            "DuckDB is an in-process, columnar vectorized OLAP SQL engine—often called the 'SQLite for Analytics'. "
            "We use DuckDB because standard row-oriented databases (like SQLite) choke when aggregating hundreds of thousands "
            "of rows. DuckDB processes analytical queries (SUM, AVG, GROUP BY, FILTER) up to 50x faster by processing data in columnar chunks "
            "directly in memory, without requiring a separate database server setup."
        ),
        "why_used": [
            "Sub-second queries on 250,000+ rows directly in memory",
            "Direct zero-copy querying of Apache Parquet files",
            "Full ANSI SQL compliance with advanced window functions"
        ]
    },
    {
        "keywords": ["why parquet", "parquet"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "Why does Quantum Analyst use Apache Parquet?",
        "simple_answer": (
            "Apache Parquet is an open-source columnar storage file format. "
            "Unlike raw CSV files (which store numbers as text and must be parsed line-by-line), "
            "Parquet stores data by columns with built-in compression. "
            "A 100 MB CSV file typically shrinks to just 15 MB in Parquet and loads 10x faster because DuckDB only reads "
            "the specific columns requested by your query, ignoring all others."
        ),
        "why_used": [
            "70% to 90% disk storage reduction via snappy compression",
            "Column-pruning: queries only touch columns mentioned in the analysis",
            "Preserves exact data types (integers, floats, dates) with zero parsing ambiguity"
        ]
    },
    {
        "keywords": ["how does ai generate insights", "ai insights", "generate insights"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "How does the AI generate insights in Quantum Analyst?",
        "simple_answer": (
            "AI Insights are generated through a 4-tier structured pipeline: "
            "1. Descriptive (What happened?): Gathers verified distributions, top categories, and volume metrics. "
            "2. Diagnostic (Why did it happen?): Evaluates correlation pairs, segment differences, and anomaly signals. "
            "3. Predictive (What is likely next?): References the champion ML model or time-series forecast. "
            "4. Prescriptive (What should you do?): Formulates prioritized, actionable business recommendations. "
            "Crucially, every single insight references concrete numbers calculated by the engine—never guessed."
        )
    },
    {
        "keywords": ["natural language to sql", "nl to sql", "nlql", "chat with data"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "How does Natural Language to SQL work?",
        "simple_answer": (
            "When you ask 'Show top 5 products by revenue', the system: "
            "1. Scans your dataset schema to match 'products' to a category column and 'revenue' to a numerical column. "
            "2. Synthesizes a safe SQL query: SELECT category, SUM(revenue) FROM dataset GROUP BY category ORDER BY SUM(revenue) DESC LIMIT 5. "
            "3. Validates the query with an AST security gate to guarantee it is read-only (blocking any DROP, DELETE, or UPDATE commands). "
            "4. Executes the query in DuckDB in milliseconds and displays both the clean data table and an automatic Plotly chart."
        )
    },
    {
        "keywords": ["agentic ai", "agentic", "what is agentic ai"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "What is Agentic AI in Quantum Analyst?",
        "simple_answer": (
            "Agentic AI means the AI does not simply answer with generic text; it has agency to select and execute tools. "
            "When you ask a question like 'Is my data ready for machine learning?', the agent decides which specialized tool to call "
            "(such as audit_ml_readiness), examines the computed health score, checks for target leakage and missingness, "
            "and synthesizes a reasoned executive briefing based on the tool's actual empirical output."
        ),
        "why_used": [
            "Autonomously coordinates multi-step data science tasks",
            "Selects from 18+ controlled backend calculation tools",
            "Explains its reasoning and actions step-by-step"
        ]
    },
    {
        "keywords": ["complete workflow", "workflow", "end to end", "steps", "stages"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "What is the complete workflow of Quantum Analyst?",
        "simple_answer": (
            "The complete 14-stage workflow takes you seamlessly from raw data to strategic decisions: "
            "1. Upload Dataset (CSV, Excel, Parquet) -> 2. Data Profile (Dimensions, Types) -> "
            "3. Data Quality (0-100 Score & Deductions) -> 4. Data Cleaning (12-category non-destructive fixes) -> "
            "5. EDA Engine (Distributions, Drivers, Anomalies) -> 6. Feature Engineering (Transforms, Selection) -> "
            "7. SQL Analytics (Natural Language queries) -> 8. Machine Learning (AutoML Benchmark & Champions) -> "
            "9. Diagnostics & Explainability (Attributions, ROC, Residuals) -> 10. PCA & Dimensionality -> "
            "11. Future Forecasting (ARIMA, Holt-Winters) -> 12. What-If Scenario Simulators -> "
            "13. AI Strategic Insights (Descriptive to Prescriptive) -> 14. Executive PDF Reports & Dashboard Builder."
        )
    },
    {
        "keywords": ["how does forecasting work", "forecasting engine", "holt winters", "arima", "time series"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "How does Future Forecasting work in Quantum Analyst?",
        "simple_answer": (
            "Quantum Analyst tests multiple time-series algorithms in parallel (Auto-ARIMA, Holt-Winters Exponential Smoothing, "
            "and Rolling Moving Averages). It runs Augmented Dickey-Fuller stationarity tests, detects seasonality cycles, "
            "and automatically selects the model with the lowest AIC and RMSE, projecting future values with 95% confidence intervals."
        ),
        "analogy": "Like a navigator looking at ocean waves and past currents to forecast exactly where the ship will be in 14 days.",
        "why_used": [
            "Provides reliable 14-day to 90-day future horizon projections",
            "Calculates statistical confidence intervals so you know worst-case and best-case bounds",
            "Guarantees that past seasonal patterns are factored in automatically"
        ]
    },
    {
        "keywords": ["what if", "scenario analysis", "counterfactual", "simulate"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "What is What-If Scenario Analysis?",
        "simple_answer": (
            "What-If Scenario Analysis lets you test counterfactual business questions. You take a real customer or product record, "
            "adjust sliders for key variables (like raising price by 10% or changing marketing spend), and see how your champion "
            "machine learning model's prediction changes in real time."
        ),
        "analogy": "Like a flight simulator for business decisions—test extreme turbulence safely before flying the real plane.",
        "why_used": [
            "Simulates the impact of strategic pricing or operational changes before committing capital",
            "Identifies high-elasticity leverage points that produce the biggest ROI changes",
            "Helps non-technical executives interact directly with machine learning predictions"
        ]
    },
    {
        "keywords": ["data quality score", "quality score", "how is quality calculated", "health score"],
        "category": CATEGORY_PROJECT_INFO,
        "question": "How is the Data Quality Score calculated?",
        "simple_answer": (
            "The Data Quality Score is a rigorous 0-to-100 metric calculated across 6 objective dimensions: "
            "1. Completeness (missing value penalties), 2. Uniqueness (duplicate row penalties), "
            "3. Validity (out-of-range types and format errors), 4. Consistency (cross-column logical contradictions), "
            "5. Accuracy (outlier severity), and 6. Integrity (schema conformance). A score above 85 is considered ML-Ready."
        ),
        "analogy": "Like a credit score for your spreadsheet—penalizing errors and rewarding clean, consistent data.",
        "why_used": [
            "Gives instant visibility into whether data is safe for decision making",
            "Pinpoints the exact column and reason behind every lost point",
            "Guides the 12-category automated cleaning engine"
        ]
    }
]


STOP_WORDS = {
    "what", "does", "this", "is", "a", "an", "the", "in", "to", "for", "of",
    "and", "or", "me", "my", "i", "can", "you", "how", "do", "it", "on", "with",
    "about", "mean", "help", "tell", "explain", "why", "are", "we", "at", "there", "any"
}


def search_project_knowledge(query: str) -> Optional[Dict[str, Any]]:
    """
    Searches the verified project knowledge base for an exact or semantic match.
    Returns the structured entry if found, else None.
    """
    q_clean = query.lower().strip()

    # 1. Exact phrase match
    for item in PROJECT_QA_REGISTRY:
        for kw in item["keywords"]:
            if kw in q_clean:
                return item

    # 2. Token overlap match (excluding stop words)
    q_words = {w for w in re.findall(r"\w+", q_clean) if w not in STOP_WORDS}
    if not q_words:
        return None

    best_match = None
    best_score = 0

    for item in PROJECT_QA_REGISTRY:
        for kw in item["keywords"]:
            kw_words = {w for w in re.findall(r"\w+", kw) if w not in STOP_WORDS}
            if not kw_words:
                continue
            overlap = len(q_words & kw_words)
            if overlap >= 2 and overlap > best_score:
                best_score = overlap
                best_match = item

    if best_score >= 2:
        return best_match

    return None


def format_project_response(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Formats a verified knowledge base entry into structured Markdown and clean voice text.
    """
    lines = [
        f"### 💡 {item['question']}",
        "",
        item["simple_answer"],
        ""
    ]

    if item.get("category") == CATEGORY_PROJECT_INFO:
        lines.extend([
            f"**Core Principle:** *{PROJECT_OVERVIEW['core_principle']}*",
            ""
        ])

    if item.get("analogy"):
        lines.extend([
            f"> **Intuitive Analogy:** {item['analogy']}",
            ""
        ])

    if item.get("examples"):
        lines.append("**Real-World Examples:**")
        for ex in item["examples"]:
            lines.append(f"- {ex}")
        lines.append("")

    if item.get("why_used"):
        lines.append("**Why Quantum Analyst Uses It:**")
        for w in item["why_used"]:
            lines.append(f"• {w}")
        lines.append("")

    if item.get("technologies"):
        lines.append(f"**Key Technologies:** `{'`, `'.join(item['technologies'])}`")

    md_text = "\n".join(lines)
    voice_text = item["simple_answer"].replace("•", ".").replace("**", "").replace("💡", "")
    if item.get("analogy"):
        voice_text += f" Think of it this way: {item['analogy']}"

    return md_text, voice_text
