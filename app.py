from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_artifact, score

ROOT = Path(__file__).parent
st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Loan Default Risk Predictor")
st.caption("Enter loan details to estimate the probability of default. Educational project - not financial advice.")


@st.cache_resource
def get_artifact():
    return load_artifact()


try:
    art = get_artifact()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Score a borrower", "Model performance", "Why? (SHAP)"])

with tab1:
    c1, c2, c3 = st.columns(3)
    if "fico" in art["features"]:
        fico = c1.slider("FICO score", 480, 850, 690)
    else:
        st.info("FICO was not available in the training data; this model does not use it.")
        fico = None
    income = c1.number_input("Annual income", 10000, 500000, 60000, step=1000)
    loan = c1.number_input("Loan amount", 1000, 45000, 15000, step=500)
    term = c2.selectbox("Term (months)", [36, 60])
    dti = c2.slider("Debt-to-income (%)", 0.0, 45.0, 18.0)
    emp = c2.slider("Years employed", 0.0, 30.0, 5.0)
    accts = c3.slider("Open accounts", 1, 40, 10)
    hist = c3.slider("Credit history (years)", 1.0, 50.0, 14.0)
    purpose = c3.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement",
                                       "car", "small_business", "other"])
    home = c2.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])

    if st.button("Predict risk", type="primary"):
        borrower = dict(annual_income=income, loan_amount=loan, term_months=term,
                        emp_years=emp, open_accounts=accts, credit_history_years=hist,
                        purpose=purpose, home_ownership=home, dti=dti)
        if fico is not None:
            borrower["fico"] = fico
        r = score(art, borrower)
        m1, m2, m3 = st.columns(3)
        m1.metric("Default probability", f"{r['probability']:.1%}")
        m2.metric("Risk tier", r["risk_tier"])
        m3.metric("Suggested action", r["decision"])
        st.progress(min(r["probability"], 1.0))
        st.caption(f"Decision threshold: {art['threshold']:.0%} (chosen so missing a default costs 10x "
                   f"more than rejecting a good borrower). Model: {art['model_name']}.")

with tab2:
    st.subheader("Cross-validation and held-out test performance")
    st.caption(
        "The model is selected by repeated-CV PR-AUC; test ROC-AUC and PR-AUC "
        "include stratified-bootstrap 95% confidence intervals. CV standard "
        "deviations summarize validation-fold scores. Compare total cost and "
        "defaults caught with approve-everyone and reject-everyone baselines. "
        "Results are uncertain when the test set has few defaults."
    )
    st.dataframe(pd.read_csv(ROOT / "reports" / "model_comparison.csv"), width="stretch")
    st.subheader("Cost sensitivity by false-negative:false-positive ratio")
    st.caption(
        "Each model's threshold is chosen using development-set out-of-fold "
        "scores, then evaluated on the held-out test set."
    )
    st.dataframe(
        pd.read_csv(ROOT / "reports" / "cost_ratio_comparison.csv"),
        width="stretch",
    )
    st.image(str(ROOT / "reports" / "evaluation.png"))
    st.subheader("Default rate by risk tier")
    st.image(str(ROOT / "reports" / "risk_tiers.png"))
    st.image(str(ROOT / "reports" / "threshold_cost.png"))

with tab3:
    shap_img = ROOT / "reports" / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="Which features push default risk up (right) or down (left)")
