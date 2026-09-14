"""
core.ml.preprocessing
=====================
Reproducible scikit-learn preprocessing pipelines for Quantum Analyst ML.
Constructs ColumnTransformers for numeric, categorical, and datetime data with
zero data leakage.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts informative numeric features from datetime columns:
    year, month, day, dayofweek, hour, is_weekend, elapsed_days.
    """

    def __init__(self, datetime_cols: Optional[List[str]] = None):
        self.datetime_cols = datetime_cols or []
        self.ref_date_ = None
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.datetime_cols if self.datetime_cols else None)

        cols = self.datetime_cols if self.datetime_cols else X.columns.tolist()
        # Find global minimum date as baseline for elapsed_days
        min_dates = []
        for col in cols:
            dt_series = pd.to_datetime(X[col], errors="coerce")
            valid = dt_series.dropna()
            if len(valid) > 0:
                min_dates.append(valid.min())

        self.ref_date_ = min(min_dates) if min_dates else pd.Timestamp("2000-01-01")
        
        # Build output feature names
        self.feature_names_out_ = []
        for col in cols:
            for suffix in ["_year", "_month", "_day", "_dayofweek", "_hour", "_is_weekend", "_elapsed_days"]:
                self.feature_names_out_.append(f"{col}{suffix}")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.datetime_cols if self.datetime_cols else None)

        cols = self.datetime_cols if self.datetime_cols else X.columns.tolist()
        extracted = []

        for col in cols:
            dt_series = pd.to_datetime(X[col], errors="coerce")
            year = dt_series.dt.year.fillna(2020).values
            month = dt_series.dt.month.fillna(1).values
            day = dt_series.dt.day.fillna(1).values
            dayofweek = dt_series.dt.dayofweek.fillna(0).values
            hour = dt_series.dt.hour.fillna(0).values
            is_weekend = (dayofweek >= 5).astype(float)
            
            # Elapsed days from ref_date_
            ref = self.ref_date_ if self.ref_date_ is not None else pd.Timestamp("2000-01-01")
            elapsed = ((dt_series - ref).dt.total_seconds() / 86400.0).fillna(0.0).values

            col_features = np.column_stack([year, month, day, dayofweek, hour, is_weekend, elapsed])
            extracted.append(col_features)

        if not extracted:
            return np.empty((len(X), 0))

        return np.hstack(extracted)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


def build_preprocessor(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = "standard",  # standard, robust, minmax, none
    impute_strategy: str = "median",  # median, mean
    cat_encoding: str = "onehot",  # onehot, ordinal
    max_categories_onehot: int = 25,
) -> Tuple[ColumnTransformer, List[str], Dict[str, List[str]]]:
    """
    Builds a complete sklearn ColumnTransformer tailored to the provided features.
    
    Returns:
        preprocessor: ColumnTransformer (unfitted)
        feature_types: Dict categorizing columns into numeric, categorical, datetime
        feature_cols: Cleaned list of feature columns
    """
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            # Check if strings could be datetime
            col_lower = str(col).lower()
            is_date_named = any(k in col_lower for k in ["date", "time", "year", "day", "month", "timestamp"])
            sample = series.dropna().head(20)
            is_dt = False
            if is_date_named and len(sample) > 0:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = pd.to_datetime(sample, errors="coerce")
                        if res.notna().sum() >= len(sample) * 0.8:
                            is_dt = True
                except Exception:
                    is_dt = False
            if is_dt:
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)

    transformers = []

    # 1. Numeric pipeline
    if numeric_cols:
        num_steps = [
            ("imputer", SimpleImputer(strategy=impute_strategy)),
        ]
        if scaler_type == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaler_type == "robust":
            num_steps.append(("scaler", RobustScaler()))
        elif scaler_type == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        # If 'none', no scaler is added

        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("numeric", num_pipeline, numeric_cols))

    # 2. Categorical pipeline
    if categorical_cols:
        # Separate low cardinality vs high cardinality if onehot selected
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
        if cat_encoding == "ordinal":
            cat_steps.append((
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ))
        else:
            cat_steps.append((
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=max_categories_onehot,
                ),
            ))
        cat_pipeline = Pipeline(steps=cat_steps)
        transformers.append(("categorical", cat_pipeline, categorical_cols))

    # 3. Datetime pipeline
    if datetime_cols:
        dt_pipeline = Pipeline(steps=[
            ("extractor", DateTimeFeatureExtractor(datetime_cols=datetime_cols))
        ])
        transformers.append(("datetime", dt_pipeline, datetime_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    column_types = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }

    return preprocessor, column_types


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve output feature names from a fitted ColumnTransformer."""
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        # Fallback names
        output_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    output_names.extend([str(x) for x in transformer.get_feature_names_out(cols)])
                    continue
                except Exception:
                    pass
            output_names.extend([str(c) for c in cols])
        return output_names
