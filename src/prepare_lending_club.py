"""Convert the Lending Club 'accepted' loans CSV into data/loans.csv for this project.

Usage:
    python -m src.prepare_lending_club path/to/accepted_2007_to_2018Q4.csv --rows 100000

Only FINISHED loans are kept (Fully Paid vs Charged Off/Default). We use only
information known when the loan was issued -- never columns like last_fico_range_*
or total_pymnt, which would leak the answer.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "loans.csv"
NEED = ["loan_status", "loan_amnt", "term", "annual_inc", "emp_length", "home_ownership",
        "purpose", "dti", "fico_range_low", "fico_range_high", "open_acc",
        "earliest_cr_line", "issue_d"]
MAIN_PURPOSES = {"debt_consolidation", "credit_card", "home_improvement", "car", "small_business"}


def emp_years(s):
    s = str(s)
    if s.startswith("<"): return 0.5
    digits = "".join(ch for ch in s if ch.isdigit())
    return float(digits) if digits else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--rows", type=int, default=100000)
    a = ap.parse_args()

    cols = pd.read_csv(a.csv, nrows=0).columns
    missing = [c for c in NEED if c not in cols]
    if missing:
        raise SystemExit(f"This file is missing columns {missing}. Use the Lending Club "
                         f"'accepted_2007_to_2018Q4.csv' file (it has fico_range_low/high).")
    df = pd.read_csv(a.csv, usecols=NEED, low_memory=False)

    status = df["loan_status"].astype(str)
    paid = status.str.contains("Fully Paid"); bad = status.str.contains("Charged Off|Default")
    df = df[paid | bad].copy()
    df["default"] = bad[df.index].astype(int)

    out = pd.DataFrame({
        "fico": (df["fico_range_low"] + df["fico_range_high"]) / 2,
        "annual_income": df["annual_inc"], "loan_amount": df["loan_amnt"],
        "term_months": df["term"].astype(str).str.extract(r"(\d+)")[0].astype(float),
        "emp_years": df["emp_length"].map(emp_years),
        "open_accounts": df["open_acc"],
        "credit_history_years": (pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce") -
                                 pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")).dt.days / 365.25,
        "purpose": df["purpose"].where(df["purpose"].isin(MAIN_PURPOSES), "other"),
        "home_ownership": df["home_ownership"], "dti": df["dti"], "default": df["default"],
    })
    out = out[out["home_ownership"].isin(["RENT", "MORTGAGE", "OWN"]) & out["fico"].notna()
              & (out["annual_income"] > 0)]
    if len(out) > a.rows:
        out = out.sample(a.rows, random_state=42)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out):,} loans to {OUT} | defaults: {int(out['default'].sum()):,} "
          f"({out['default'].mean():.1%})")


if __name__ == "__main__":
    main()
