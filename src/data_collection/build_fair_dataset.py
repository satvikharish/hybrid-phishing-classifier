"""Rebuild the dataset so benign and phishing URLs are compared fairly.

The original dataset.csv built benign rows as "http://" + a bare domain, so
every benign URL had no path and no HTTPS, while most phishing URLs had a
path and HTTPS. A model could score highly on those shortcuts alone.

This script puts both classes in the same form (lowercase hostname, "www."
removed, nothing else), then:
  - drops duplicate hostnames,
  - drops hostnames whose registered domain appears in BOTH classes
    (phishing hosted on popular legitimate platforms such as google.com;
    host-level features cannot separate those, so they are counted and
    reported rather than kept as conflicting labels),
  - caps phishing at MAX_PER_DOMAIN hostnames per registered domain so a
    few big hosts do not dominate.

Output: data/processed/fair_dataset.csv [hostname, label, reg_domain, source]
Also writes data/processed/fair_dataset_report.json with the counts.
"""
import json
import os
from urllib.parse import urlparse

import pandas as pd
import tldextract

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MAX_PER_DOMAIN = 3
SEED = 42


def hostname_of(url: str):
    host = urlparse(url if "://" in url else f"http://{url}").hostname
    if not host:
        return None
    host = host.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def registered_domain(host: str) -> str:
    ext = tldextract.extract(host)
    return ext.registered_domain or host


def build(df: pd.DataFrame, max_per_domain: int = MAX_PER_DOMAIN):
    report = {"input_rows": int(len(df))}
    df = df.copy()
    df["hostname"] = df["url"].map(hostname_of)
    df = df.dropna(subset=["hostname"])
    df["reg_domain"] = df["hostname"].map(registered_domain)
    report["phishing_rows_before"] = int((df.label == 1).sum())
    report["benign_rows_before"] = int((df.label == 0).sum())

    df = df.drop_duplicates(subset=["hostname", "label"])
    report["distinct_phishing_hostnames"] = int((df.label == 1).sum())
    report["distinct_phishing_registered_domains"] = int(df[df.label == 1].reg_domain.nunique())

    benign_domains = set(df[df.label == 0].reg_domain)
    conflict = (df.label == 1) & df.reg_domain.isin(benign_domains)
    report["phishing_hostnames_on_popular_legit_domains_dropped"] = int(conflict.sum())
    report["share_of_distinct_phishing_hostnames_on_popular_legit_domains"] = round(float(conflict.sum() / max((df.label == 1).sum(), 1)), 4)
    df = df[~conflict]

    ph = df[df.label == 1].sample(frac=1.0, random_state=SEED)
    ph = ph.groupby("reg_domain", group_keys=False).head(max_per_domain)
    be = df[df.label == 0]
    out = pd.concat([ph, be], ignore_index=True)
    out = out.drop_duplicates(subset=["hostname"], keep=False) if out.hostname.duplicated().any() else out
    report["final_phishing"] = int((out.label == 1).sum())
    report["final_benign"] = int((out.label == 0).sum())
    report["final_phishing_registered_domains"] = int(out[out.label == 1].reg_domain.nunique())
    return out[["hostname", "label", "reg_domain", "source"]], report


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PROCESSED, "dataset.csv"))
    out, report = build(df)
    out.to_csv(os.path.join(PROCESSED, "fair_dataset.csv"), index=False)
    with open(os.path.join(PROCESSED, "fair_dataset_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
