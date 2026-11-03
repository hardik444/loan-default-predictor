"""Train, compare, tune the threshold, explain and save the loan default model.

Run from the project root:  python -m src.train
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .buckets import optimal_fico_buckets
from .data import TARGET, load_data
from .preprocess import add_features, build_preprocessor

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
COST_FN = 10.0
COST_FP = 1.0
N_SPLITS = 5
N_REPEATS = 3
N_BOOTSTRAPS = 1000
LOW_DEFAULT_WARNING = 100


def best_threshold(y_true, proba, cost_fn=COST_FN, cost_fp=COST_FP):
    """Find the exact score cutoff with minimum cost, including both extremes."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if len(y_true) != len(proba):
        raise ValueError("y_true and proba must contain the same number of rows.")
    if not np.isfinite(proba).all():
        raise ValueError("Threshold selection requires finite probabilities.")

    scores, inverse = np.unique(proba, return_inverse=True)
    positives = np.bincount(inverse, weights=(y_true == 1).astype(int))
    negatives = np.bincount(inverse, weights=(y_true == 0).astype(int))
    positives_below = np.concatenate(([0], np.cumsum(positives)[:-1]))
    negatives_below = np.concatenate(([0], np.cumsum(negatives)[:-1]))
    thresholds = np.concatenate(([0.0], scores, [np.nextafter(scores[-1], np.inf)]))
    missed_defaults = np.concatenate(([0], positives_below, [positives.sum()]))
    flagged_good = np.concatenate(([
        negatives.sum()
    ], negatives.sum() - negatives_below, [0]))
    costs = cost_fn * missed_defaults + cost_fp * flagged_good
    best = int(np.argmin(costs))
    return float(thresholds[best]), thresholds, costs


def _fit_calibrated(pipe, X, y, calibration_folds=3):
    estimator = clone(pipe)
    if isinstance(estimator.named_steps["clf"], XGBClassifier):
        negatives = int((y == 0).sum())
        positives = int((y == 1).sum())
        estimator.set_params(clf__scale_pos_weight=negatives / positives)
    return CalibratedClassifierCV(
        estimator, method="sigmoid", cv=calibration_folds
    ).fit(X, y)


def _bootstrap_ci(y_true, scores, metric, seed, n_bootstraps=N_BOOTSTRAPS):
    """Stratified percentile bootstrap CI, keeping each class count fixed."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    if not len(positive) or not len(negative):
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        sample = np.concatenate((
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ))
        estimates[i] = metric(y_true[sample], scores[sample])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _make_candidates(features, large_dataset=False):
    forest_trees = 100 if large_dataset else 300
    boosting_rounds = 200 if large_dataset else 400
    candidates = {
        "Logistic Regression (baseline)": Pipeline([
            ("prep", build_preprocessor(True, True, features)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(True, False, features)),
            ("clf", RandomForestClassifier(n_estimators=forest_trees, min_samples_leaf=20,
                                           class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=SEED))]),
        "XGBoost": Pipeline([
            ("prep", build_preprocessor(True, False, features)),
            ("clf", XGBClassifier(n_estimators=boosting_rounds, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=1.0, eval_metric="auc",
                                  random_state=SEED, n_jobs=-1))]),
    }
    if "fico" in features:
        candidates["Logistic Regression (raw FICO only)"] = Pipeline([
            ("prep", build_preprocessor(False, True, features)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    return candidates


def main():
    df = load_data()
    if TARGET not in df:
        raise ValueError(f"Input data must contain the target column {TARGET!r}.")
    class_counts = df[TARGET].value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < N_SPLITS:
        raise ValueError(
            f"Training requires both target classes and at least {N_SPLITS} rows "
            f"of each; found {class_counts.to_dict()}."
        )
    print(f"Rows: {len(df):,} | default rate: {df[TARGET].mean():.2%}")
    large_dataset = len(df) >= 100_000
    n_splits = 3 if large_dataset else N_SPLITS
    n_repeats = 2 if large_dataset else N_REPEATS
    calibration_folds = 2 if large_dataset else 3
    bootstrap_iterations = 300 if large_dataset else N_BOOTSTRAPS
    print(
        f"Evaluation: {n_repeats} repeats of {n_splits}-fold stratified CV; "
        f"{bootstrap_iterations} bootstrap samples."
    )

    development_raw, test_raw = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=SEED
    )
    y_dev = development_raw[TARGET].to_numpy()
    y_test = test_raw[TARGET].to_numpy()
    test_default_count = int(y_test.sum())
    if test_default_count < LOW_DEFAULT_WARNING:
        warnings.warn(
            f"The held-out test set contains only {test_default_count} defaults "
            f"({len(test_raw)} rows). AUC confidence intervals and threshold "
            f"comparisons will be uncertain; fewer than {LOW_DEFAULT_WARNING} "
            "test defaults is below the project's reporting threshold.",
            RuntimeWarning,
            stacklevel=1,
        )

    edges = None
    if "fico" in development_raw:
        edges = [int(e) for e in optimal_fico_buckets(
            development_raw["fico"], y_dev, n_buckets=6
        )]
        print("FICO bucket upper edges:", edges)
    else:
        print("FICO unavailable in source data; training without FICO buckets.")
    development = add_features(development_raw, edges)
    test = add_features(test_raw, edges)
    features = [column for column in development.columns if column != TARGET]
    X_dev, X_test = development[features], test[features]

    candidates = _make_candidates(features, large_dataset=large_dataset)
    fold_splits = []
    for repeat in range(n_repeats):
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=SEED + repeat
        )
        fold_splits.extend(
            (repeat, train_idx, valid_idx)
            for train_idx, valid_idx in splitter.split(development_raw, y_dev)
        )

    rows = []
    fitted = {}
    thresholds_by_model = {}
    cost_curves = {}
    for name, pipe in candidates.items():
        print(f"\nRepeated stratified CV: {name}")
        repeated_oof = np.full((n_repeats, len(development_raw)), np.nan)
        for repeat, train_idx, valid_idx in fold_splits:
            fold_train = development_raw.iloc[train_idx]
            fold_valid = development_raw.iloc[valid_idx]
            fold_y = y_dev[train_idx]
            fold_edges = None
            if "fico" in fold_train:
                fold_edges = optimal_fico_buckets(
                    fold_train["fico"], fold_y, n_buckets=6
                )
            train_features = add_features(fold_train, fold_edges)
            valid_features = add_features(fold_valid, fold_edges)
            model = _fit_calibrated(
                pipe, train_features[features], fold_y,
                calibration_folds=calibration_folds,
            )
            repeated_oof[repeat, valid_idx] = model.predict_proba(
                valid_features[features]
            )[:, 1]

        if not np.isfinite(repeated_oof).all():
            raise RuntimeError(f"Repeated CV did not produce all OOF scores for {name}.")
        oof_scores = repeated_oof.mean(axis=0)
        cv_auc = np.array([
            roc_auc_score(y_dev, repeat_scores) for repeat_scores in repeated_oof
        ])
        cv_pr_auc = np.array([
            average_precision_score(y_dev, repeat_scores)
            for repeat_scores in repeated_oof
        ])
        threshold, curve_thresholds, curve_costs = best_threshold(y_dev, oof_scores)

        final_model = _fit_calibrated(
            pipe, X_dev, y_dev, calibration_folds=calibration_folds
        )
        fitted[name] = final_model
        test_scores = final_model.predict_proba(X_test)[:, 1]
        predicted_default = test_scores >= threshold
        roc_low, roc_high = _bootstrap_ci(
            y_test, test_scores, roc_auc_score, SEED,
            n_bootstraps=bootstrap_iterations,
        )
        pr_low, pr_high = _bootstrap_ci(
            y_test, test_scores, average_precision_score, SEED + 1,
            n_bootstraps=bootstrap_iterations,
        )
        rows.append({
            "model": name,
            "cv_roc_auc_mean": cv_auc.mean(),
            "cv_roc_auc_std": cv_auc.std(ddof=1),
            "cv_pr_auc_mean": cv_pr_auc.mean(),
            "cv_pr_auc_std": cv_pr_auc.std(ddof=1),
            "roc_auc": roc_auc_score(y_test, test_scores),
            "roc_auc_ci_lower": roc_low,
            "roc_auc_ci_upper": roc_high,
            "pr_auc": average_precision_score(y_test, test_scores),
            "pr_auc_ci_lower": pr_low,
            "pr_auc_ci_upper": pr_high,
            "threshold": threshold,
            "precision": precision_score(y_test, predicted_default, zero_division=0),
            "recall": recall_score(y_test, predicted_default, zero_division=0),
            "defaults_caught": int((predicted_default & (y_test == 1)).sum()),
            "false_positives": int((predicted_default & (y_test == 0)).sum()),
            "total_cost": (
                COST_FN * ((~predicted_default) & (y_test == 1)).sum()
                + COST_FP * (predicted_default & (y_test == 0)).sum()
            ),
        })
        thresholds_by_model[name] = threshold
        cost_curves[name] = (curve_thresholds, curve_costs)

    approve_all_cost = COST_FN * test_default_count
    rows.append({
        "model": "Approve everyone / no-skill PR baseline",
        "cv_roc_auc_mean": np.nan,
        "cv_roc_auc_std": np.nan,
        "cv_pr_auc_mean": float(y_dev.mean()),
        "cv_pr_auc_std": np.nan,
        "roc_auc": np.nan,
        "roc_auc_ci_lower": np.nan,
        "roc_auc_ci_upper": np.nan,
        "pr_auc": float(y_test.mean()),
        "pr_auc_ci_lower": np.nan,
        "pr_auc_ci_upper": np.nan,
        "threshold": np.nan,
        "precision": test_default_count / len(y_test),
        "recall": 1.0,
        "defaults_caught": test_default_count,
        "false_positives": int((y_test == 0).sum()),
        "total_cost": approve_all_cost,
    })

    results = pd.DataFrame(rows)
    eligible = results[
        results["model"].isin(candidates)
        & ~results["model"].str.contains("raw FICO")
    ]
    best_name = eligible.loc[
        eligible["cv_pr_auc_mean"].notna(),
        ["model", "cv_pr_auc_mean"],
    ].sort_values("cv_pr_auc_mean", ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    threshold = thresholds_by_model[best_name]
    p_test = best.predict_proba(X_test)[:, 1]
    print("\n=== Repeated CV and held-out test comparison ===")
    print(results.round(4).to_string(index=False))
    print(
        f"\nSelected by repeated-CV PR-AUC: {best_name} | "
        f"OOF cost threshold: {threshold:.4f}"
    )
    print(
        f"Approve-everyone baseline test cost: {approve_all_cost:.0f} "
        f"({test_default_count} missed defaults × {COST_FN:.0f})"
    )
    results.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    joblib.dump({
        "model": best,
        "edges": edges,
        "threshold": threshold,
        "features": features,
        "model_name": best_name,
    }, ROOT / "models" / "loan_model.joblib")
    with (ROOT / "models" / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump({
            "model": best_name,
            "threshold": threshold,
            "fico_edges": edges,
        }, config_file, indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, model in fitted.items():
        RocCurveDisplay.from_predictions(
            y_test, model.predict_proba(X_test)[:, 1], name=name, ax=ax[0]
        )
    ax[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="No-skill baseline")
    ax[0].legend(loc="lower right", fontsize="small")
    ax[0].set_title("ROC curves (held-out test set)")
    PrecisionRecallDisplay.from_predictions(y_test, p_test, ax=ax[1], name=best_name)
    ax[1].axhline(
        y_test.mean(), linestyle="--", color="gray",
        label=f"No-skill baseline ({y_test.mean():.3%})",
    )
    ax[1].legend(loc="upper right", fontsize="small")
    ax[1].set_title("Precision-Recall (held-out test set)")
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, p_test >= threshold),
        display_labels=["Repaid", "Default"],
    ).plot(ax=ax[2], colorbar=False)
    ax[2].set_title(f"Confusion matrix @ threshold {threshold:.3f}")
    plt.tight_layout()
    fig.savefig(ROOT / "reports" / "evaluation.png", dpi=130)
    plt.close(fig)

    thresholds, costs = cost_curves[best_name]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, costs)
    ax.axhline(
        COST_FN * int(y_dev.sum()), color="gray", linestyle=":",
        label="Approve everyone (OOF)",
    )
    ax.axvline(threshold, color="r", linestyle="--")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total cost (repeated-CV OOF)")
    ax.set_title(f"OOF threshold by cost (FN costs {COST_FN:.0f}x FP)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "threshold_cost.png", dpi=130)
    plt.close(fig)

    scored = test.copy()
    scored["p_default"] = p_test
    scored["risk_tier"] = pd.qcut(
        scored["p_default"],
        5,
        labels=["Very low", "Low", "Medium", "High", "Very high"],
    )
    tier = scored.groupby("risk_tier", observed=True).agg(
        borrowers=(TARGET, "size"),
        actual_default_rate=(TARGET, "mean"),
        avg_predicted=("p_default", "mean"),
    ).round(4)
    tier.to_csv(ROOT / "reports" / "risk_tiers.csv")
    print("\n=== Default rate by risk tier ===\n", tier)
    ax = tier[["actual_default_rate", "avg_predicted"]].plot.bar(
        figsize=(7, 4), rot=0
    )
    ax.set_title("Default rate by risk tier (held-out test set)")
    ax.set_ylabel("Default rate")
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "risk_tiers.png", dpi=130)
    plt.close()

    fico_report = ROOT / "reports" / "default_rate_by_fico_bucket.csv"
    if "fico_bucket" in test:
        test.groupby("fico_bucket")[TARGET].mean().to_csv(fico_report)
    else:
        pd.DataFrame(columns=["fico_bucket", TARGET]).to_csv(
            fico_report, index=False
        )

    raw = clone(candidates[best_name])
    if isinstance(raw.named_steps["clf"], XGBClassifier):
        raw.set_params(
            clf__scale_pos_weight=int((y_dev == 0).sum()) / int((y_dev == 1).sum())
        )
    raw.fit(X_dev, y_dev)
    prep, clf = raw.named_steps["prep"], raw.named_steps["clf"]
    shap_sample = X_test.sample(min(1500, len(X_test)), random_state=SEED)
    Xt = prep.transform(shap_sample)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    names = prep.get_feature_names_out()
    if isinstance(clf, LogisticRegression):
        sv = shap.LinearExplainer(clf, Xt).shap_values(Xt)
    else:
        sv = shap.TreeExplainer(clf).shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        if getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
    shap.summary_plot(sv, Xt, feature_names=names, show=False)
    plt.tight_layout()
    plt.savefig(
        ROOT / "reports" / "shap_summary.png", dpi=130, bbox_inches="tight"
    )
    plt.close()
    mean_abs = pd.Series(np.abs(sv).mean(0), index=names).sort_values(
        ascending=False
    )
    mean_abs.to_csv(
        ROOT / "reports" / "feature_importance_shap.csv",
        header=["mean_abs_shap"],
    )
    print("\nTop drivers of default (SHAP):\n", mean_abs.head(6).round(4))

    print("\nSaved model to models/loan_model.joblib and charts to reports/")


if __name__ == "__main__":
    main()
