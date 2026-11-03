"""Data loading.

Real data: put a CSV at data/loans.csv (see README for Kaggle "Give Me Some Credit"
or Lending Club column mapping). If it is missing, we generate a realistic
synthetic dataset so the whole pipeline still runs end to end.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "loans.csv"
LENDING_CLUB_PATH = DATA_DIR / "loan_lenders.csv"
TARGET = "default"
LENDING_CLUB_COLUMNS = [
    "loan_amnt",
    "term",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "issue_d",
    "loan_status",
    "purpose",
    "dti",
    "earliest_cr_line",
    "open_acc",
]


def _normalize_lending_club_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    status = chunk["loan_status"].astype("string").str.strip()
    labels = {
        "Fully Paid": 0,
        "Does not meet the credit policy. Status:Fully Paid": 0,
        "Charged Off": 1,
        "Does not meet the credit policy. Status:Charged Off": 1,
        "Default": 1,
    }
    resolved = status.isin(labels)
    if not resolved.any():
        return pd.DataFrame(columns=[
            "annual_income", "loan_amount", "term_months", "emp_years",
            "open_accounts", "credit_history_years", "purpose",
            "home_ownership", "dti", TARGET,
        ])

    data = chunk.loc[resolved]
    outcome = status.loc[resolved].map(labels).astype("int8")

    issue_date = pd.to_datetime(data["issue_d"], format="%b-%Y", errors="coerce")
    credit_start = pd.to_datetime(
        data["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    credit_history_years = (
        (issue_date.dt.year - credit_start.dt.year)
        + (issue_date.dt.month - credit_start.dt.month) / 12
    ).clip(lower=0)

    employment = data["emp_length"].astype("string").str.strip()
    emp_years = pd.to_numeric(
        employment.str.extract(r"(\d+)", expand=False), errors="coerce"
    ).astype(float)
    emp_years = emp_years.mask(employment.str.startswith("<", na=False), 0.5)

    normalized = pd.DataFrame({
        "annual_income": pd.to_numeric(data["annual_inc"], errors="coerce"),
        "loan_amount": pd.to_numeric(data["loan_amnt"], errors="coerce"),
        "term_months": pd.to_numeric(
            data["term"].astype("string").str.extract(r"(\d+)", expand=False),
            errors="coerce",
        ),
        "emp_years": emp_years,
        "open_accounts": pd.to_numeric(data["open_acc"], errors="coerce"),
        "credit_history_years": credit_history_years,
        "purpose": data["purpose"].astype("string").str.strip().str.lower(),
        "home_ownership": data["home_ownership"].astype("string").str.strip().str.upper(),
        "dti": pd.to_numeric(data["dti"], errors="coerce"),
        TARGET: outcome,
    }, index=data.index)
    return normalized.reset_index(drop=True)


def load_lending_club(path: Path = LENDING_CLUB_PATH, chunksize: int = 200_000) -> pd.DataFrame:
    """Load resolved Lending Club loans and map them to the training schema."""
    normalized_chunks = []
    for chunk in pd.read_csv(
        path, usecols=LENDING_CLUB_COLUMNS, chunksize=chunksize, low_memory=False
    ):
        normalized = _normalize_lending_club_chunk(chunk)
        if not normalized.empty:
            normalized_chunks.append(normalized)
    if not normalized_chunks:
        raise ValueError(f"No resolved loans with supported outcomes found in {path}.")

    data = pd.concat(normalized_chunks, ignore_index=True)
    for column in ("purpose", "home_ownership"):
        data[column] = data[column].astype("category")
    return data


def make_synthetic(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fico = np.clip(rng.normal(690, 55, n), 480, 850).round()
    income = np.exp(rng.normal(11.0, 0.55, n)).round(-2)          # annual income
    loan_amount = np.clip(rng.normal(14000, 8000, n), 1000, 45000).round(-2)
    term = rng.choice([36, 60], n, p=[0.7, 0.3])
    emp_years = np.clip(rng.exponential(5, n), 0, 30).round(1)
    open_accounts = np.clip(rng.poisson(10, n), 1, 40)
    credit_history_years = np.clip(rng.normal(14, 7, n), 1, 50).round(1)
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "car", "small_business", "other"],
        n, p=[0.40, 0.20, 0.12, 0.10, 0.06, 0.12])
    home = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.42, 0.13])
    dti = np.clip(rng.normal(18, 8, n), 0, 45).round(1)            # debt-to-income %
    loan_to_income = loan_amount / income

    # true (hidden) default risk
    z = (-1.2
         - 0.022 * (fico - 690)
         + 0.035 * (dti - 18)
         + 3.2 * (loan_to_income - 0.15)
         + 0.45 * (term == 60)
         + 0.60 * (purpose == "small_business")
         + 0.15 * (home == "RENT")
         - 0.03 * emp_years
         - 0.015 * credit_history_years
         # non-linear effects, as in real credit data
         + 0.06 * np.clip(dti - 30, 0, None)                      # very high DTI is extra risky
         + 0.9 * ((term == 60) & (fico < 650))                    # long loan + weak credit
         + 0.7 * ((purpose == "small_business") & (loan_to_income > 0.3))
         + 0.5 * (income < 30000))
    z -= 2.0  # makes defaults rare (class imbalance, ~10%)
    p = 1 / (1 + np.exp(-z))
    default = rng.binomial(1, p)

    df = pd.DataFrame(dict(
        fico=fico, annual_income=income, loan_amount=loan_amount, term_months=term,
        emp_years=emp_years, open_accounts=open_accounts,
        credit_history_years=credit_history_years, purpose=purpose,
        home_ownership=home, dti=dti, default=default))
    # realistic missing values
    for col, frac in [("emp_years", 0.06), ("dti", 0.03), ("credit_history_years", 0.04)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


def load_data() -> pd.DataFrame:
    if LENDING_CLUB_PATH.exists():
        print(f"Loading resolved Lending Club data from {LENDING_CLUB_PATH}")
        data = load_lending_club()
        defaults = int(data[TARGET].sum())
        print(
            f"Resolved rows: {len(data):,} | defaults: {defaults:,} "
            f"({data[TARGET].mean():.2%})"
        )
        return data
    if DATA_PATH.exists():
        print(f"Loading real data from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    print("data/loans.csv not found -> using synthetic data")
    return make_synthetic()
