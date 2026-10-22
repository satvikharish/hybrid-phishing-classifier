# Hybrid Phishing Classifier

Does adding live server checks (DNS, SSL, WHOIS, HTTP) to features read from the URL text improve phishing detection, and what does it cost in time? This repository holds the data pipeline, the models, the evaluation, and the figures for that study.

## Main findings

All numbers are from a fair, hostname-only comparison with cross-validation split by registered domain (see "The dataset correction" below).

| | URL text only | URL text + server checks |
|---|---|---|
| AUROC | 0.875 | 0.966 |
| Recall (threshold 0.5) | 72% | 92% |
| False positive rate (threshold 0.5) | 9.6% | 9.7% |
| False positive rate at 99% recall | 0.89 | 0.51 |
| AUROC on newer, unseen phishing domains | 0.823 | 0.945 |

- SSL and DNS checks carry most of the gain. HTTP headers add nothing, and WHOIS adds little (the WHOIS lookups were unreliable).
- A weighted XGBoost and Random Forest ensemble is no better than XGBoost alone.
- Server checks are slow: median 845 ms per host, slowest 10% about 7 s (the WHOIS timeout). DNS and SSL alone have a median of 222 ms. A cascade that runs the checks only when the URL-text model is unsure cuts the checks by 44% for a 0.010 loss in AUROC.

Aggregate results for every table and figure are in [`results/`](results/). Figures are in [`reports/figures/`](reports/figures/).

## The dataset correction

The first version of this project stored benign URLs as bare domains, while phishing URLs kept their paths and HTTPS. A model could score about 99.6% accuracy (AUROC 0.999) from URL shape alone. With both classes in the same form (hostname only), the same model scores 86.5% accuracy (AUROC 0.878). The first-draft figures are kept in [`reports/archive_first_draft/`](reports/archive_first_draft/) for the record.

## Repository layout

```
src/data_collection/   fetch_feeds, build_dataset (original), build_fair_dataset, build_fair_hybrid_features
src/features/          lexical.py (URL-text features), host_telemetry.py (DNS, WHOIS, SSL, HTTP lookups)
src/models/            fair_eval.py (main evaluation), temporal_validation.py, latency_analysis.py,
                       export_results.py, paper_figures.py
src/utils/cache.py     disk cache for server lookups
results/               aggregate results (no hostnames)
reports/figures/       the five three-panel figures
tests/                 unit tests
```

## Reproducing the results

Python 3.9. On macOS, XGBoost needs `brew install libomp`. Install the packages with `pip install -r requirements.txt` (exact versions used are in `requirements-lock.txt`). Run from the repository root, in this order:

```bash
python -m src.data_collection.fetch_feeds                      # download the public feeds
python -m src.data_collection.build_dataset                    # original dataset (kept for the Claim 1 comparison)
python -m src.data_collection.build_lexical_features
python -m src.data_collection.build_fair_dataset               # fair hostname-level dataset
python -m src.data_collection.build_fair_hybrid_features 1500  # server checks for 3,000 hosts (about 10 minutes)
python -m src.models.fair_eval lexical                         # dataset-construction comparison
python -m src.models.fair_eval hybrid                          # main evaluation, ablations, cascade
python -m src.models.fair_eval shap
python -m src.models.fair_eval fast-subset                     # URL text + DNS + SSL only
python -m src.models.temporal_validation build                 # newer-data test set
python -m src.data_collection.build_fair_hybrid_features 1000 temporal_test.csv temporal_hybrid_features.csv 42
python -m src.models.temporal_validation lexical
python -m src.models.temporal_validation hybrid
python -m src.models.latency_analysis extended 100             # 200 uncached hosts, then cached
python -m src.models.latency_analysis fast 60                  # DNS + SSL only
python -m src.models.export_results                            # copy aggregate results to results/
python -m src.models.paper_figures                             # redraw the figures from results/
python -m pytest tests
```

The study used feeds downloaded on September 16, 2026 and October 22, 2026. Those snapshots are not redistributed and the feeds change daily, so a rerun downloads different URLs and gives similar but not identical numbers. `temporal_validation.py` expects the earlier snapshot's files to be dated 2026-09-16 (the `OLD` constant), so change it if you rerun on a new date.

## First-draft pipeline

`build_dataset.py`, `build_lexical_features.py`, `build_host_features.py`, `build_hybrid_features.py`, `train_baseline.py`, `train_ensemble.py`, `shap_analysis.py`, `evaluate.py`, `features.py`, and the first part of `latency_analysis.py` belong to the original analysis. They are kept because the Claim 1 comparison needs the original dataset. Do not use their results as findings.

## Data and ethics

- Phishing hostnames come from PhishTank and OpenPhish. Benign hostnames come from the Majestic Million. These are third-party data with their own terms, so lists of individual hostnames are not committed; the scripts rebuild them.
- The server-check code requests only the status code and headers of each host and never renders or runs page content.
