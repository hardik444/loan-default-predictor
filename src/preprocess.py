import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .buckets import apply_buckets

CATEGORICAL = ["purpose", "home_ownership"]
NUMERIC_BASE = ["fico", "annual_income", "loan_amount", "term_months", "emp_years",
                "open_accounts", "credit_history_years", "dti",
                "loan_to_income", "fico_bucket"]


def add_features(df: pd.DataFrame, edges) -> pd.DataFrame:
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"].replace(0, np.nan)
    if "fico" in df and edges is not None:
        df["fico_bucket"] = apply_buckets(df["fico"], edges)
    return df


def build_preprocessor(
    use_fico_bucket: bool = True,
    scale: bool = True,
    feature_columns: list[str] | None = None,
) -> ColumnTransformer:
    available = set(feature_columns) if feature_columns is not None else set(NUMERIC_BASE + CATEGORICAL)
    numeric = [
        c for c in NUMERIC_BASE
        if c in available and (use_fico_bucket or c != "fico_bucket")
    ]
    categorical = [c for c in CATEGORICAL if c in available]
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
