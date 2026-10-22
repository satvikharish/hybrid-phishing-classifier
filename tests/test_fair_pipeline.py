import numpy as np
import pandas as pd
from scipy import stats

from src.data_collection.build_fair_dataset import build, hostname_of, registered_domain
from src.models.fair_eval import cascade, corrected_t, metrics


def test_hostname_of_strips_scheme_www_port_and_case():
    assert hostname_of("HTTP://WWW.Example.com/path?x=1") == "example.com"
    assert hostname_of("https://a.b.example.co.uk:8080/x") == "a.b.example.co.uk"
    assert hostname_of("example.org") == "example.org"


def test_registered_domain():
    assert registered_domain("a.b.example.co.uk") == "example.co.uk"
    assert registered_domain("login.weebly.com") == "weebly.com"


def test_fair_dataset_rules():
    rows = [("http://example.com", 0, "majestic"), ("http://good.org", 0, "majestic")]
    rows += [("http://example.com/login", 1, "phishtank")]  # phishing on a popular legit domain: dropped
    rows += [("http://evil1.xyz/a", 1, "phishtank"), ("http://evil1.xyz/b", 1, "openphish")]  # same hostname: kept once
    rows += [(f"http://{c}.evil2.tk/x", 1, "phishtank") for c in "abcde"]  # five hosts, one domain: capped at 3
    out, report = build(pd.DataFrame(rows, columns=["url", "label", "source"]), max_per_domain=3)
    phish = out[out.label == 1]
    assert report["phishing_hostnames_on_popular_legit_domains_dropped"] == 1
    assert "example.com" not in set(phish.hostname)
    assert (phish.hostname == "evil1.xyz").sum() == 1
    assert (phish.reg_domain == "evil2.tk").sum() == 3
    assert set(out[out.label == 0].hostname) == {"example.com", "good.org"}


def test_corrected_t_is_more_conservative_than_naive_t():
    rng = np.random.default_rng(0)
    d = 0.02 + rng.normal(0, 0.01, 50)
    naive = stats.ttest_1samp(d, 0).pvalue
    corrected = corrected_t(d, n_splits=5)["p"]
    assert corrected > naive


def test_corrected_t_all_zero_differences():
    assert corrected_t(np.zeros(20), n_splits=5)["p"] == 1.0


def test_cascade_extremes_match_single_models():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p_lex = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 400), 0.01, 0.99)
    p_hyb = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 400), 0.01, 0.99)
    res = cascade(y, p_lex, p_hyb, [0.0, 0.5], lex_ms=1.0, host_ms=1000.0).set_index("delta")
    assert res.loc[0.0, "share_needing_server_checks"] == 0.0
    assert abs(res.loc[0.0, "auroc"] - metrics(y, p_lex)["auroc"]) < 1e-12
    assert res.loc[0.5, "share_needing_server_checks"] == 1.0
    assert abs(res.loc[0.5, "auroc"] - metrics(y, p_hyb)["auroc"]) < 1e-12
    assert res.loc[0.5, "avg_latency_ms"] > res.loc[0.0, "avg_latency_ms"]


def test_metrics_fpr_at_99_recall_perfect_separation():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.9] * 50)
    m = metrics(y, p)
    assert m["auroc"] == 1.0 and m["fpr"] == 0.0 and m["fpr_at_99tpr"] == 0.0
