"""Score a single borrower. Used by the Streamlit app and runnable from the CLI."""
from pathlib import Path
import joblib
import pandas as pd

from .preprocess import add_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "loan_model.joblib"
TIERS = [(0.05, "Very low"), (0.10, "Low"), (0.20, "Medium"), (0.35, "High"), (1.01, "Very high")]


def load_artifact():
    return joblib.load(MODEL_PATH)


def score(artifact: dict, borrower: dict) -> dict:
    df = add_features(pd.DataFrame([borrower]), artifact["edges"])
    proba = float(artifact["model"].predict_proba(df[artifact["features"]])[:, 1][0])
    tier = next(label for cut, label in TIERS if proba < cut)
    return {"probability": proba, "risk_tier": tier,
            "decision": "REVIEW / DECLINE" if proba >= artifact["threshold"] else "APPROVE",
            "fico_bucket": (
                int(df["fico_bucket"].iloc[0]) if "fico_bucket" in df else None
            )}


if __name__ == "__main__":
    art = load_artifact()
    example = dict(fico=640, annual_income=45000, loan_amount=20000, term_months=60,
                   emp_years=2.0, open_accounts=8, credit_history_years=6.0,
                   purpose="small_business", home_ownership="RENT", dti=28.0)
    print(score(art, example))
