"""Test on newer data: train on the September 16 snapshot, test on phishing
domains that first appeared in a fresh download and on benign domains the
model never saw.

Strict novelty: a test domain is used only if its registered domain was not
in the September data at all (either class). Phishing hosts on popular
legitimate domains are excluded, as in the main dataset. Benign test hosts
are Majestic Million ranks 50,001-60,000 from the fresh download.

Usage:
    python -m src.models.temporal_validation build [new_date]   # downloads feeds, writes temporal_test.csv
    python -m src.models.temporal_validation lexical             # lexical model on the full snapshot
    python -m src.models.temporal_validation hybrid              # hybrid vs lexical on a telemetry sample
"""
import json
import os
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd

from src.data_collection.build_fair_dataset import MAX_PER_DOMAIN, hostname_of, registered_domain
from src.data_collection.fetch_feeds import fetch_all
from src.features.lexical import extract_features
from src.models.fair_eval import (HOST_ALL, LEX_FAIR, domain_bootstrap, metrics, prep_hybrid, xgb_factory)

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OLD = "2026-09-16"


def _read_phish(day):
    urls = []
    with open(os.path.join(RAW, f"openphish_{day}.txt"), errors="ignore") as f:
        urls += [l.strip() for l in f if l.strip()]
    urls += pd.read_csv(os.path.join(RAW, f"phishtank_{day}.csv"), usecols=["url"])["url"].tolist()
    return urls


def build(new_day=None, seed=42):
    new_day = new_day or date.today().isoformat()
    fetch_all()
    old_ph = pd.DataFrame({"url": _read_phish(OLD)})
    old_ph["host"] = old_ph.url.map(hostname_of)
    old_be = pd.read_csv(os.path.join(RAW, f"majestic_million_{OLD}.csv"), usecols=["Domain"], nrows=50000)
    old_domains = set(old_ph.host.dropna().map(registered_domain)) | set(old_be.Domain.map(lambda d: registered_domain(d.lower())))

    new_ph = pd.DataFrame({"url": _read_phish(new_day)})
    new_ph["host"] = new_ph.url.map(hostname_of)
    new_ph = new_ph.dropna(subset=["host"]).drop_duplicates("host")
    new_ph["reg_domain"] = new_ph.host.map(registered_domain)

    new_maj = pd.read_csv(os.path.join(RAW, f"majestic_million_{new_day}.csv"), usecols=["Domain"], nrows=60000)
    new_be = new_maj.iloc[50000:60000].copy()
    new_be["host"] = new_be.Domain.str.lower()
    new_be["reg_domain"] = new_be.host.map(registered_domain)
    popular = set(new_maj.iloc[:60000].Domain.map(lambda d: registered_domain(d.lower())))

    report = {"new_day": new_day, "new_phishing_hosts_in_feed": int(len(new_ph))}
    new_ph = new_ph[~new_ph.reg_domain.isin(old_domains)]
    report["new_phishing_hosts_on_unseen_domains"] = int(len(new_ph))
    on_popular = new_ph.reg_domain.isin(popular)
    report["of_which_on_popular_legit_domains_dropped"] = int(on_popular.sum())
    new_ph = new_ph[~on_popular].sample(frac=1.0, random_state=seed).groupby("reg_domain", group_keys=False).head(MAX_PER_DOMAIN)
    new_be = new_be[~new_be.reg_domain.isin(old_domains)].drop_duplicates("host")

    test = pd.concat([pd.DataFrame({"hostname": new_ph.host, "label": 1, "reg_domain": new_ph.reg_domain, "source": "fresh_feed"}),
                      pd.DataFrame({"hostname": new_be.host, "label": 0, "reg_domain": new_be.reg_domain, "source": "majestic_50k_60k"})],
                     ignore_index=True)
    test.to_csv(os.path.join(PROCESSED, "temporal_test.csv"), index=False)
    report["final_phishing"], report["final_benign"] = int((test.label == 1).sum()), int((test.label == 0).sum())
    with open(os.path.join(PROCESSED, "temporal_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def _lex(hosts):
    return pd.DataFrame([extract_features(f"http://{h}") for h in hosts])[LEX_FAIR].astype(float)


def lexical():
    train = pd.read_csv(os.path.join(PROCESSED, "fair_dataset.csv"))
    test = pd.read_csv(os.path.join(PROCESSED, "temporal_test.csv"))
    Xtr, ytr = _lex(train.hostname), train.label.to_numpy()
    Xte, yte = _lex(test.hostname), test.label.to_numpy()
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    m = xgb_factory(spw=spw)()
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    res = {"model": "xgb_lexical (trained on Sept snapshot)", "n_test": int(len(yte)), "n_phishing": int(yte.sum()), **metrics(yte, p)}
    pd.DataFrame([res]).to_csv(os.path.join(PROCESSED, "temporal_lexical_results.csv"), index=False)
    print(json.dumps(res, indent=2))
    return res


def hybrid():
    tr_df, Xtr, ytr, gtr = prep_hybrid(os.path.join(PROCESSED, "fair_hybrid_features.csv"))
    te_df, Xte, yte, gte = prep_hybrid(os.path.join(PROCESSED, "temporal_hybrid_features.csv"))
    rp = os.path.join(PROCESSED, "fair_hybrid_report.json")
    params = json.load(open(rp)).get("tuned_xgb_params", {}) if os.path.exists(rp) else {}
    sets = {"xgb_lexical": LEX_FAIR, "xgb_host_only": HOST_ALL, "xgb_hybrid": LEX_FAIR + HOST_ALL}
    rows, probs = [], {}
    for name, cols in sets.items():
        m = xgb_factory(params)()
        m.fit(Xtr[cols], ytr)
        probs[name] = m.predict_proba(Xte[cols])[:, 1]
        rows.append({"model": name, "n_test": int(len(yte)), "n_phishing": int(yte.sum()), **metrics(yte, probs[name])})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROCESSED, "temporal_hybrid_results.csv"), index=False)
    print(res[["model", "accuracy", "precision", "recall", "fpr", "auroc", "ap", "fpr_at_99tpr", "precision_at_99recall"]].to_string())
    boots = [domain_bootstrap(yte, probs["xgb_lexical"], probs["xgb_hybrid"], gte, m, B=1000) for m in ("auroc", "ap")]
    print(boots)
    with open(os.path.join(PROCESSED, "temporal_hybrid_bootstrap.json"), "w") as f:
        json.dump(boots, f, indent=2)
    return res


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": lambda: build(sys.argv[2] if len(sys.argv) > 2 else None), "lexical": lexical, "hybrid": hybrid}[mode]()
