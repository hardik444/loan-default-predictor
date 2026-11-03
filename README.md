# Loan Default Risk Predictor

An end-to-end credit-risk ML pipeline: it predicts the probability that a borrower
defaults, converts it into a risk tier and a cost-based approve/review decision,
and explains *why* with SHAP. Built as a follow-up to the J.P. Morgan Quantitative
Research job simulation (probability of default + FICO bucketing).

## What's inside
| Piece | File | Idea |
|---|---|---|
| FICO bucketing | `src/buckets.py` | Dynamic programming finds bucket edges that maximise default log-likelihood (the JPM idea, now a model feature) |
| Preprocessing | `src/preprocess.py` | Median imputation, scaling, one-hot encoding, `loan_to_income`, `fico_bucket` |
| Imbalance | `src/train.py` | Repeated stratified cross-validation + balanced class weights / `scale_pos_weight`, judged primarily by PR-AUC and cost (not accuracy) |
| Models | `src/train.py` | Logistic Regression (baseline), Random Forest, XGBoost, + raw-FICO ablation |
| Calibration | `src/train.py` | Sigmoid calibration so "20%" really means ~20% of such borrowers default |
| Threshold | `src/train.py` | Chosen from repeated-CV out-of-fold predictions by minimising cost (missing a default = 10x a wrongly flagged good borrower) |
| Evaluation | `src/train.py` | Model selection by repeated-CV PR-AUC; held-out test ROC-AUC and PR-AUC include stratified-bootstrap 95% confidence intervals and an approve-everyone / no-skill PR baseline |
| Explainability | `src/train.py` | SHAP summary plot + `reports/feature_importance_shap.csv` |
| Dashboard / app | `app.py` | Streamlit: score a borrower, view metrics, view SHAP |

## Run it
```bash
pip install -r requirements.txt
python -m src.train          # trains, evaluates, saves model + charts in reports/
python -m src.predict        # scores an example borrower
streamlit run app.py         # opens the web app
```

### Data availability

The raw datasets are intentionally excluded from GitHub. To train on Lending
Club data, place the source CSV at `data/loan_lenders.csv`; alternatively, place
a CSV in the canonical format at `data/loans.csv`. If neither file is present,
the training pipeline uses its built-in synthetic dataset. The trained model
and evaluation reports are included in the repository.

## Data
If `data/loan_lenders.csv` exists, the pipeline loads it in chunks, maps
Lending Club fields, and trains only on fully resolved outcomes: Fully Paid as
non-default and Charged Off/Default as default. Current and late/in-grace-period
loans are excluded. It derives term, employment length, and credit-history years
from the source columns. The supplied Lending Club file does not have FICO
columns, so FICO and FICO-bucket features are omitted rather than inferred from
grade. Otherwise, if `data/loans.csv` exists, it is used; when neither file is
present, the project generates a **synthetic** dataset (20,000 loans,
approximately 12% defaults). The canonical CSV format is:
`fico, annual_income, loan_amount, term_months, emp_years, open_accounts,
credit_history_years, purpose, home_ownership, dti, default` (default: 1 = defaulted).

## Results
The training report selects a model by mean PR-AUC across repeated stratified
cross-validation, then reports its performance on a held-out test set. Training
uses class weighting; the decision threshold is chosen from out-of-fold scores
to minimize the configured default-miss / false-alarm cost. The CSV also
includes 95% stratified-bootstrap confidence intervals for test ROC-AUC and
PR-AUC, a no-skill PR baseline, and the test cost of approving every borrower. A warning is emitted
when the test set contains fewer than 100 defaults; treat metrics from such a
small test set as highly uncertain and do not present them as strong evidence.

## Design decisions to talk about in interviews
1. Why accuracy is misleading for imbalanced default data, and why a small number of test defaults makes metric estimates uncertain.
2. Why bucket edges are learned on the training split only (avoids leakage).
3. Why probabilities are calibrated and how the cost ratio sets the threshold.
4. Bucketed FICO vs raw FICO: compare the two logistic regression rows.
5. Limits: synthetic data, no fairness audit, no time-based split.
