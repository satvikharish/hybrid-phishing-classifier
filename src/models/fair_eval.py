"""Stronger evaluation of lexical vs. host-telemetry vs. hybrid phishing models.

Fixes over the first evaluation:
  * both classes are compared in the same hostname-only form (see
    build_fair_dataset.py),
  * cross-validation never puts the same registered domain in both train and
    test (StratifiedGroupKFold),
  * paired comparisons use the Nadeau-Bengio corrected resampled t-test,
    a per-repeat test, and a domain-level bootstrap, because folds from
    repeated cross-validation overlap and are not independent,
  * reports threshold-free and operating-point metrics that do not saturate
    (average precision, FPR at 99% recall, precision at 99% recall),
  * compares the ensemble against XGBoost alone and Random Forest alone,
  * ablates the host feature groups,
  * evaluates a two-stage cascade (lexical first, server checks only when
    the lexical model is unsure).

Usage:
    python -m src.models.fair_eval lexical       # dataset-construction inflation + lexical baselines
    python -m src.models.fair_eval hybrid        # main evaluation, ablations, cascade
    python -m src.models.fair_eval shap          # SHAP importance of the hybrid model
    python -m src.models.fair_eval fast-subset   # URL text + DNS + SSL only
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

from src.data_collection.build_fair_dataset import hostname_of, registered_domain
from src.features.lexical import extract_features

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

LEX_FAIR = ["url_length", "hostname_length", "num_dots", "num_hyphens", "num_underscores", "num_digits",
            "num_special_chars", "digit_ratio", "shannon_entropy", "num_subdomains", "has_ip_address",
            "suspicious_keyword_count"]
HOST_GROUPS = {
    "dns": ["has_a_record", "has_mx_record", "has_txt_record"],
    "whois": ["domain_age_days", "has_whois_record"],
    "ssl": ["has_valid_ssl", "ssl_days_until_expiry"],
    "http": ["http_status", "server_header_present", "has_hsts", "has_csp", "has_x_frame_options"],
}
HOST_ALL = [c for g in HOST_GROUPS.values() for c in g]
ORIG_LEX = ["url_length", "hostname_length", "path_length", "query_length", "num_dots", "num_hyphens",
            "num_underscores", "num_slashes", "num_digits", "num_special_chars", "digit_ratio",
            "shannon_entropy", "num_subdomains", "has_ip_address", "has_https", "suspicious_keyword_count"]

XGB_BASE = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist")
XGB_SEARCH = {"n_estimators": [100, 200, 300, 400], "max_depth": [3, 4, 5, 6, 8],
              "learning_rate": [0.01, 0.05, 0.1, 0.2], "subsample": [0.7, 0.8, 0.9, 1.0],
              "colsample_bytree": [0.7, 0.8, 0.9, 1.0]}


def xgb_factory(params=None, spw=1.0):
    p = {**XGB_BASE, **(params or {})}
    return lambda: XGBClassifier(scale_pos_weight=spw, **p)


def rf_factory():
    return lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)


class SoftVote:
    """Weighted average of XGBoost and Random Forest probabilities (weights 2:1)."""

    def __init__(self, xgb_make, rf_make, w_xgb=2.0, w_rf=1.0):
        self.xgb, self.rf, self.wx, self.wr = xgb_make(), rf_make(), w_xgb, w_rf

    def fit(self, X, y):
        self.xgb.fit(X, y)
        self.rf.fit(X, y)
        return self

    def predict_proba(self, X):
        return (self.wx * self.xgb.predict_proba(X) + self.wr * self.rf.predict_proba(X)) / (self.wx + self.wr)


def soft_vote_factory(params=None, spw=1.0):
    return lambda: SoftVote(xgb_factory(params, spw), rf_factory())


def metrics(y, p, thr=0.5, extra=True):
    pred = (p >= thr).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    out = {"accuracy": float((pred == y).mean()), "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)), "fpr": fp / max(fp + tn, 1),
           "auroc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p))}
    if extra:
        fpr, tpr, _ = roc_curve(y, p)
        out["fpr_at_99tpr"] = float(fpr[np.argmax(tpr >= 0.99)])
        prec, rec, _ = precision_recall_curve(y, p)
        out["precision_at_99recall"] = float(prec[rec >= 0.99].max()) if (rec >= 0.99).any() else 0.0
        out["brier"] = float(brier_score_loss(y, p))
    return out


def repeated_cv(X, y, groups, models, n_splits=5, n_repeats=3, seed=42, grouped=True):
    """models: name -> (factory, columns). Returns per-fold metrics and out-of-fold probabilities."""
    n = len(y)
    oof = {m: np.full((n_repeats, n), np.nan) for m in models}
    rows = []
    for r in range(n_repeats):
        if grouped:
            splits = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r).split(X, y, groups)
        else:
            splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r).split(X, y)
        for k, (tr, te) in enumerate(splits):
            for name, (make, cols) in models.items():
                m = make()
                m.fit(X.iloc[tr][cols], y[tr])
                p = m.predict_proba(X.iloc[te][cols])[:, 1]
                oof[name][r, te] = p
                rows.append({"repeat": r, "fold": k, "model": name, "n_train": len(tr), "n_test": len(te),
                             **metrics(y[te], p)})
    return pd.DataFrame(rows), oof


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in folds.columns if c not in ("repeat", "fold", "model", "n_train", "n_test")]
    g = folds.groupby("model")[cols]
    mean, sd = g.mean(), g.std()
    out = mean.copy()
    for c in cols:
        out[c] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean[c], sd[c])]
    return out


def corrected_t(diffs, n_splits):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    d = np.asarray(diffs, float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"t": float("nan"), "p": 1.0 if d.mean() == 0 else 0.0}
    t = d.mean() / np.sqrt((1.0 / J + 1.0 / (n_splits - 1)) * var)
    return {"t": float(t), "p": float(2 * stats.t.sf(abs(t), J - 1))}


def compare(folds, a, b, metric, n_splits):
    """Paired comparison b - a on one metric."""
    pa = folds[folds.model == a].set_index(["repeat", "fold"])[metric]
    pb = folds[folds.model == b].set_index(["repeat", "fold"])[metric]
    d = (pb - pa).dropna()
    per_rep = d.groupby(level="repeat").mean()
    res = {"a": a, "b": b, "metric": metric, "mean_a": float(pa.mean()), "mean_b": float(pb.mean()),
           "mean_diff": float(d.mean()), "folds_b_better": int((d > 1e-12).sum()),
           "folds_tied": int((d.abs() <= 1e-12).sum()), "folds_b_worse": int((d < -1e-12).sum()),
           "naive_paired_t_p": float(stats.ttest_1samp(d, 0).pvalue) if d.var() > 0 else 1.0}
    c = corrected_t(d, n_splits)
    res["corrected_t"], res["corrected_p"] = c["t"], c["p"]
    try:
        res["per_repeat_wilcoxon_p"] = float(stats.wilcoxon(per_rep).pvalue)
    except ValueError:
        res["per_repeat_wilcoxon_p"] = float("nan")
    return res


def domain_bootstrap(y, p_a, p_b, groups, metric="auroc", B=500, seed=0):
    """Bootstrap domains (not rows) to get a CI on metric(b) - metric(a)."""
    fn = roc_auc_score if metric == "auroc" else average_precision_score
    rng = np.random.default_rng(seed)
    idx_by_g = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(np.array)
    gs = idx_by_g.index.to_numpy()
    diffs = []
    for _ in range(B):
        pick = rng.choice(gs, size=len(gs), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(fn(y[idx], p_b[idx]) - fn(y[idx], p_a[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"metric": metric, "mean_diff": float(np.mean(diffs)), "ci_low": float(lo), "ci_high": float(hi)}


def cascade(y, p_lex, p_hyb, deltas, lex_ms, host_ms):
    """Use the hybrid score only when the lexical score is within `delta` of 0.5."""
    rows = []
    for d in deltas:
        unsure = np.abs(p_lex - 0.5) < d
        p = np.where(unsure, p_hyb, p_lex)
        frac = float(unsure.mean())
        rows.append({"delta": d, "share_needing_server_checks": frac,
                     "avg_latency_ms": lex_ms + frac * host_ms, **metrics(y, p)})
    return pd.DataFrame(rows)


def load_latency():
    path = os.path.join(PROCESSED, "latency_results.csv")
    lat = pd.read_csv(path)
    return float(lat["lexical_ms"].mean()), float(lat["host_telemetry_ms"].mean())


def run_lexical(seed=42):
    """Shows how much of the first result came from the dataset construction."""
    fair = pd.read_csv(os.path.join(PROCESSED, "fair_dataset.csv"))
    Xf = pd.DataFrame([extract_features(f"http://{h}") for h in fair.hostname])[LEX_FAIR].astype(float)
    yf, gf = fair.label.to_numpy(), fair.reg_domain.to_numpy()

    orig = pd.read_csv(os.path.join(PROCESSED, "lexical_features.csv"))
    Xo = orig[ORIG_LEX].astype(float).reset_index(drop=True)
    yo = orig.label.astype(int).to_numpy()
    go = orig.url.map(hostname_of).map(registered_domain).to_numpy()

    out = []
    for data_name, X, y, g, cols in [("original construction (bare-domain benign URLs)", Xo, yo, go, ORIG_LEX),
                                     ("fair construction (hostname only, both classes)", Xf, yf, gf, LEX_FAIR)]:
        spw = float((y == 0).sum() / (y == 1).sum())
        for grouped in (False, True):
            folds, _ = repeated_cv(X, y, g, {"xgb": (xgb_factory(spw=spw), cols)}, n_splits=5, n_repeats=1,
                                   seed=seed, grouped=grouped)
            row = {"data": data_name, "cv": "domain-grouped" if grouped else "random rows", "n_rows": len(y)}
            row.update({k: float(folds[k].mean()) for k in ["accuracy", "precision", "recall", "fpr", "auroc", "ap",
                                                           "fpr_at_99tpr", "precision_at_99recall"]})
            out.append(row)
            print(row)
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(PROCESSED, "fair_lexical_inflation.csv"), index=False)

    spw = float((yf == 0).sum() / (yf == 1).sum())
    folds, oof = repeated_cv(Xf, yf, gf, {"xgb_lexical": (xgb_factory(spw=spw), LEX_FAIR),
                                          "rf_lexical": (rf_factory(), LEX_FAIR)}, n_splits=5, n_repeats=3, seed=seed)
    folds.to_csv(os.path.join(PROCESSED, "fair_lexical_folds.csv"), index=False)
    summ = summarize(folds)
    summ.to_csv(os.path.join(PROCESSED, "fair_lexical_summary.csv"))
    print(summ.T.to_string())
    return res, summ


def prep_hybrid(path):
    df = pd.read_csv(path)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    X = df[LEX_FAIR + HOST_ALL].apply(pd.to_numeric, errors="coerce").astype(float)
    return df, X, df.label.to_numpy().astype(int), df.reg_domain.to_numpy()


def split_eval_tune(X, y, g, seed=42):
    """80% of domains for evaluation, a separate 20% for tuning (no domain is in both)."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(cv.split(X, y, g))
    return train_idx, test_idx  # (evaluation, tuning)


def tune(X, y, g, cols, seed=42):
    """Tune XGBoost on a held-out 20% of domains that is NOT used for evaluation."""
    eval_idx, tune_idx = split_eval_tune(X, y, g, seed)
    search = RandomizedSearchCV(XGBClassifier(**XGB_BASE), XGB_SEARCH, n_iter=15, scoring="average_precision",
                                cv=GroupKFold(n_splits=3), random_state=seed, n_jobs=1)
    search.fit(X.iloc[tune_idx][cols], y[tune_idx], groups=g[tune_idx])
    return search.best_params_, eval_idx, tune_idx


def run_hybrid(path=None, tag="fair", seed=42, n_repeats=10):
    path = path or os.path.join(PROCESSED, "fair_hybrid_features.csv")
    df, X, y, g = prep_hybrid(path)
    lex_ms, host_ms = load_latency()
    report = {"n_rows": int(len(y)), "n_phishing": int(y.sum()), "n_domains": int(len(set(g)))}

    params, eval_idx, tune_idx = tune(X, y, g, LEX_FAIR + HOST_ALL, seed)
    report["tuned_xgb_params"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
    report["n_tuning_rows"], report["n_eval_rows"] = int(len(tune_idx)), int(len(eval_idx))
    Xe, ye, ge = X.iloc[eval_idx].reset_index(drop=True), y[eval_idx], g[eval_idx]
    dfe = df.iloc[eval_idx].reset_index(drop=True)

    HYB = LEX_FAIR + HOST_ALL
    models = {
        "xgb_lexical": (xgb_factory(params), LEX_FAIR),
        "xgb_host_only": (xgb_factory(params), HOST_ALL),
        "xgb_hybrid": (xgb_factory(params), HYB),
        "rf_hybrid": (rf_factory(), HYB),
        "ensemble_hybrid": (soft_vote_factory(params), HYB),
    }
    for gname, gcols in HOST_GROUPS.items():
        models[f"xgb_lexical+{gname}"] = (xgb_factory(params), LEX_FAIR + gcols)
        models[f"xgb_{gname}_only"] = (xgb_factory(params), gcols)

    folds, oof = repeated_cv(Xe, ye, ge, models, n_splits=5, n_repeats=n_repeats, seed=seed)
    folds.to_csv(os.path.join(PROCESSED, f"{tag}_hybrid_folds.csv"), index=False)
    summ = summarize(folds)
    summ.to_csv(os.path.join(PROCESSED, f"{tag}_hybrid_summary.csv"))
    print(summ[["accuracy", "precision", "recall", "fpr", "auroc", "ap", "fpr_at_99tpr", "precision_at_99recall"]].to_string())

    comps = []
    for a, b in [("xgb_lexical", "xgb_hybrid"), ("xgb_lexical", "ensemble_hybrid"), ("xgb_hybrid", "ensemble_hybrid"),
                 ("rf_hybrid", "ensemble_hybrid"), ("xgb_lexical", "xgb_host_only")]:
        for met in ["auroc", "ap", "precision", "fpr", "fpr_at_99tpr"]:
            comps.append(compare(folds, a, b, met, 5))
    comp = pd.DataFrame(comps)
    comp.to_csv(os.path.join(PROCESSED, f"{tag}_hybrid_comparisons.csv"), index=False)
    print(comp[comp.metric.isin(["auroc", "ap"])][["a", "b", "metric", "mean_a", "mean_b", "mean_diff", "folds_b_better",
                                                  "folds_tied", "folds_b_worse", "naive_paired_t_p", "corrected_p",
                                                  "per_repeat_wilcoxon_p"]].to_string())

    boots = [domain_bootstrap(ye, oof["xgb_lexical"][0], oof["xgb_hybrid"][0], ge, m) for m in ("auroc", "ap")]
    report["domain_bootstrap_hybrid_minus_lexical"] = boots
    print(boots)

    # Subset where the host resolves (removes the 'dead phishing domain' shortcut)
    alive = (dfe.has_a_record == 1).to_numpy() if dfe.has_a_record.dtype != bool else dfe.has_a_record.to_numpy()
    if alive.sum() > 200 and len(np.unique(ye[alive])) == 2:
        Xa, ya, ga = Xe[alive].reset_index(drop=True), ye[alive], ge[alive]
        mods = {"xgb_lexical": models["xgb_lexical"], "xgb_hybrid": models["xgb_hybrid"], "xgb_host_only": models["xgb_host_only"]}
        f_a, _ = repeated_cv(Xa, ya, ga, mods, n_splits=5, n_repeats=n_repeats, seed=seed)
        f_a.to_csv(os.path.join(PROCESSED, f"{tag}_alive_folds.csv"), index=False)
        s_a = summarize(f_a)
        s_a.to_csv(os.path.join(PROCESSED, f"{tag}_alive_summary.csv"))
        print("\nAlive-only subset:", len(ya), "rows,", int(ya.sum()), "phishing\n", s_a[["accuracy", "precision", "fpr", "auroc", "ap"]].to_string())
        c_a = pd.DataFrame([compare(f_a, "xgb_lexical", "xgb_hybrid", m, 5) for m in ["auroc", "ap", "precision", "fpr"]])
        c_a.to_csv(os.path.join(PROCESSED, f"{tag}_alive_comparisons.csv"), index=False)
        print(c_a[["metric", "mean_a", "mean_b", "mean_diff", "naive_paired_t_p", "corrected_p"]].to_string())
        report["alive_subset_rows"], report["alive_subset_phishing"] = int(len(ya)), int(ya.sum())

    # Cascade using repeat-averaged out-of-fold scores
    deltas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    parts = [cascade(ye, oof["xgb_lexical"][r], oof["xgb_hybrid"][r], deltas, lex_ms, host_ms) for r in range(n_repeats)]
    casc = pd.concat(parts).groupby("delta").mean().reset_index()
    casc.to_csv(os.path.join(PROCESSED, f"{tag}_cascade.csv"), index=False)
    print(casc[["delta", "share_needing_server_checks", "avg_latency_ms", "accuracy", "precision", "fpr", "auroc", "ap"]].to_string())

    pd.DataFrame({"hostname": dfe.hostname, "label": ye, "reg_domain": ge,
                  "p_lexical": oof["xgb_lexical"][0], "p_hybrid": oof["xgb_hybrid"][0],
                  "p_host_only": oof["xgb_host_only"][0]}).to_csv(os.path.join(PROCESSED, f"{tag}_oof_predictions.csv"), index=False)
    report["latency_ms_used"] = {"lexical": lex_ms, "host_telemetry": host_ms}
    with open(os.path.join(PROCESSED, f"{tag}_hybrid_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return folds, comp, casc


def run_shap(seed=42):
    """SHAP importance of the hybrid XGBoost model fit on the full telemetry sample."""
    import shap

    df, X, y, g = prep_hybrid(os.path.join(PROCESSED, "fair_hybrid_features.csv"))
    params = json.load(open(os.path.join(PROCESSED, "fair_hybrid_report.json")))["tuned_xgb_params"]
    cols = LEX_FAIR + HOST_ALL
    model = xgb_factory(params)()
    model.fit(X[cols], y)
    sv = shap.TreeExplainer(model).shap_values(X[cols])
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp["source"] = ["Server check" if c in HOST_ALL else "URL text" for c in cols]
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(PROCESSED, "fair_shap_importance.csv"), index=False)
    print(imp.head(12).round(4).to_string())
    return imp


def run_fast_subset(seed=42, n_repeats=10):
    """URL text + DNS + SSL only (drops the slow WHOIS and the uninformative HTTP checks)."""
    df, X, y, g = prep_hybrid(os.path.join(PROCESSED, "fair_hybrid_features.csv"))
    params = json.load(open(os.path.join(PROCESSED, "fair_hybrid_report.json")))["tuned_xgb_params"]
    eval_idx, _ = split_eval_tune(X, y, g, seed)
    Xe, ye, ge = X.iloc[eval_idx].reset_index(drop=True), y[eval_idx], g[eval_idx]
    models = {"xgb_lexical": (xgb_factory(params), LEX_FAIR),
              "xgb_lexical+dns+ssl": (xgb_factory(params), LEX_FAIR + HOST_GROUPS["dns"] + HOST_GROUPS["ssl"]),
              "xgb_hybrid": (xgb_factory(params), LEX_FAIR + HOST_ALL)}
    folds, _ = repeated_cv(Xe, ye, ge, models, n_splits=5, n_repeats=n_repeats, seed=seed)
    folds.to_csv(os.path.join(PROCESSED, "fair_fast_subset_folds.csv"), index=False)
    summ = summarize(folds)
    summ.to_csv(os.path.join(PROCESSED, "fair_fast_subset_summary.csv"))
    comp = pd.DataFrame([compare(folds, "xgb_lexical+dns+ssl", "xgb_hybrid", m, 5) for m in ["auroc", "ap", "precision", "fpr"]])
    comp.to_csv(os.path.join(PROCESSED, "fair_fast_subset_comparison.csv"), index=False)
    print(summ[["accuracy", "precision", "recall", "fpr", "auroc", "ap", "fpr_at_99tpr"]].to_string())
    print(comp[["metric", "mean_a", "mean_b", "mean_diff", "corrected_p"]].to_string())
    return folds, comp


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    mode = sys.argv[1] if len(sys.argv) > 1 else "lexical"
    if mode == "lexical":
        run_lexical()
    elif mode == "shap":
        run_shap()
    elif mode == "fast-subset":
        run_fast_subset()
    else:
        run_hybrid()
