"""Paired significance test: is the hybrid ensemble's AUROC edge over the
lexical-only baseline real, or noise?

Repeats stratified 10-fold CV across multiple random seeds. Both models
are trained/evaluated on the IDENTICAL train/test split within each fold
(paired design), and every fold's result is kept -- none are discarded
or re-run looking for a particular outcome. Reports a paired t-test, a
Wilcoxon signed-rank test (non-parametric, since per-fold differences
need not be normal), and a bootstrap confidence interval on the mean
AUROC difference.

Hyperparameters are fixed to the values already selected by
RandomizedSearchCV in train_ensemble.py, rather than re-tuned per fold --
re-tuning inside every fold of every repeat would be both extremely slow
and would add a second source of variance on top of the one being tested.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.models.features import to_xy
from src.models.train_ensemble import LEXICAL_COLUMNS

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
N_REPEATS = 10
N_SPLITS = 10
N_BOOTSTRAP = 10000


def build_fixed_hybrid_ensemble() -> VotingClassifier:
    xgb = XGBClassifier(
        eval_metric="logloss", random_state=42, n_jobs=-1,
        n_estimators=200, max_depth=6, learning_rate=0.01,
        subsample=1.0, colsample_bytree=0.9,
    )
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    return VotingClassifier(estimators=[("xgboost", xgb), ("random_forest", rf)], voting="soft", weights=[2, 1])


def _fpr(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def _paired_report(name: str, a: np.ndarray, b: np.ndarray) -> None:
    diffs = b - a
    if np.allclose(diffs, 0):
        print(f"{name}: identical in every fold (both {a.mean():.4f}) -- no variation to test, ceiling effect")
        return
    t_stat, t_p = stats.ttest_1samp(diffs, 0.0)
    try:
        w_stat, w_p = stats.wilcoxon(diffs)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    print(f"{name}: lexical={a.mean():.4f} hybrid={b.mean():.4f} diff={diffs.mean():+.4f}  "
          f"t-test p={t_p:.4f}  wilcoxon p={w_p:.4f}")


def run() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "hybrid_features.csv"))
    X, y, _ = to_xy(df)
    X_lexical = X[LEXICAL_COLUMNS].to_numpy()
    X_hybrid = X.to_numpy()
    y = y.to_numpy()

    lexical_template = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    hybrid_template = build_fixed_hybrid_ensemble()

    lex_auroc, hyb_auroc = [], []
    lex_prec, hyb_prec = [], []
    lex_fpr, hyb_fpr = [], []
    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1000 + repeat)
        for train_idx, test_idx in skf.split(X_hybrid, y):
            lex_fold = clone(lexical_template)
            hyb_fold = clone(hybrid_template)

            lex_fold.fit(X_lexical[train_idx], y[train_idx])
            hyb_fold.fit(X_hybrid[train_idx], y[train_idx])

            y_test = y[test_idx]
            lex_proba = lex_fold.predict_proba(X_lexical[test_idx])[:, 1]
            hyb_proba = hyb_fold.predict_proba(X_hybrid[test_idx])[:, 1]
            lex_pred = lex_fold.predict(X_lexical[test_idx])
            hyb_pred = hyb_fold.predict(X_hybrid[test_idx])

            lex_auroc.append(roc_auc_score(y_test, lex_proba))
            hyb_auroc.append(roc_auc_score(y_test, hyb_proba))
            lex_prec.append(precision_score(y_test, lex_pred, zero_division=0))
            hyb_prec.append(precision_score(y_test, hyb_pred, zero_division=0))
            lex_fpr.append(_fpr(y_test, lex_pred))
            hyb_fpr.append(_fpr(y_test, hyb_pred))

    lex_auroc, hyb_auroc = np.array(lex_auroc), np.array(hyb_auroc)
    lex_prec, hyb_prec = np.array(lex_prec), np.array(hyb_prec)
    lex_fpr, hyb_fpr = np.array(lex_fpr), np.array(hyb_fpr)
    diffs = hyb_auroc - lex_auroc

    t_stat, t_p = stats.ttest_1samp(diffs, 0.0)
    w_stat, w_p = stats.wilcoxon(diffs)
    rng = np.random.default_rng(42)
    boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(N_BOOTSTRAP)])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    n = len(diffs)
    print(f"folds evaluated: {n}  ({N_REPEATS} repeats x {N_SPLITS} folds)\n")
    print("--- AUROC (what was tested before) ---")
    print(f"lexical-only AUROC: mean={lex_auroc.mean():.4f}  std={lex_auroc.std():.4f}")
    print(f"hybrid AUROC:       mean={hyb_auroc.mean():.4f}  std={hyb_auroc.std():.4f}")
    print(f"mean diff (hybrid - lexical): {diffs.mean():.4f}  std={diffs.std():.4f}")
    print(f"paired t-test:        t={t_stat:.3f}  p={t_p:.4f}")
    print(f"wilcoxon signed-rank: W={w_stat:.3f}  p={w_p:.4f}")
    print(f"95% bootstrap CI on mean diff: [{ci_low:.4f}, {ci_high:.4f}]")

    print("\n--- Precision and FPR (what the original proposal actually claims) ---")
    _paired_report("precision", lex_prec, hyb_prec)
    _paired_report("false positive rate", lex_fpr, hyb_fpr)

    out = pd.DataFrame({
        "lexical_auroc": lex_auroc, "hybrid_auroc": hyb_auroc,
        "lexical_precision": lex_prec, "hybrid_precision": hyb_prec,
        "lexical_fpr": lex_fpr, "hybrid_fpr": hyb_fpr,
    })
    out_path = os.path.join(PROCESSED_DIR, "significance_test_results.csv")
    out.to_csv(out_path, index=False)
    print(f"\n[ok] wrote all {n} per-fold results -> {out_path}")
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    run()
