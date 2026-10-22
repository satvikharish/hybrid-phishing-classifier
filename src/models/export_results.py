"""Copy the aggregate results from data/processed/ (git-ignored) into results/ (committed).

Only aggregate numbers are exported. Files that list individual hostnames
(the datasets, the telemetry samples, per-host predictions) stay out of the
repository: the phishing feeds are third-party data with their own terms,
and they change daily. The scripts in this repository rebuild them.

The out-of-fold predictions are exported with the hostname and domain columns
removed (label and scores only), which is all the ROC figure needs.
"""
import json
import os
import shutil

import pandas as pd

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")

COPY = [
    "fair_dataset_report.json",
    "fair_lexical_inflation.csv", "fair_lexical_summary.csv", "fair_lexical_folds.csv",
    "fair_hybrid_summary.csv", "fair_hybrid_folds.csv", "fair_hybrid_comparisons.csv", "fair_hybrid_report.json",
    "fair_alive_summary.csv", "fair_alive_folds.csv", "fair_alive_comparisons.csv",
    "fair_fast_subset_summary.csv", "fair_fast_subset_folds.csv", "fair_fast_subset_comparison.csv",
    "fair_cascade.csv", "fair_shap_importance.csv",
    "temporal_report.json", "temporal_lexical_results.csv", "temporal_hybrid_results.csv", "temporal_hybrid_bootstrap.json",
    "latency_extended.json", "latency_fast_path.json",
]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for name in COPY:
        shutil.copy(os.path.join(PROCESSED, name), os.path.join(RESULTS, name))
    oof = pd.read_csv(os.path.join(PROCESSED, "fair_oof_predictions.csv"))
    oof[["label", "p_lexical", "p_host_only", "p_hybrid"]].to_csv(os.path.join(RESULTS, "fair_oof_scores.csv"), index=False)
    print(f"[ok] exported {len(COPY) + 1} files -> results/")


if __name__ == "__main__":
    main()
