"""Week 6: feature-extraction latency trade-offs (lexical vs. host telemetry).

Measures wall-clock time per URL for each feature source, to quantify the
deployment cost of adding host-based telemetry (the proposal's stated
"extraction latency trade-offs" discussion point).
"""
import asyncio
import json
import os
import time

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.host_telemetry import extract_host_features
from src.features.lexical import extract_features

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")


def measure_lexical_latency(urls: list) -> list:
    latencies = []
    for url in urls:
        start = time.perf_counter()
        extract_features(url)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    return latencies


async def _measure_host_latency_async(urls: list) -> list:
    latencies = []
    connector = aiohttp.TCPConnector(limit=50, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for url in urls:
            start = time.perf_counter()
            await extract_host_features(url, session)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
    return latencies


def measure_host_latency(urls: list) -> list:
    return asyncio.run(_measure_host_latency_async(urls))


def run(sample_size: int = 50) -> pd.DataFrame:
    dataset_path = os.path.join(PROCESSED_DIR, "dataset.csv")
    df = pd.read_csv(dataset_path).sample(sample_size, random_state=7)
    urls = df["url"].tolist()

    lexical_ms = measure_lexical_latency(urls)
    host_ms = measure_host_latency(urls)

    results = pd.DataFrame({
        "lexical_ms": lexical_ms,
        "host_telemetry_ms": host_ms,
    })

    os.makedirs(FIGURES_DIR, exist_ok=True)
    means = results.mean()
    plt.figure()
    plt.bar(["Lexical\n(no network)", "Host telemetry\n(DNS/WHOIS/SSL/HTTP)"], means.values)
    plt.yscale("log")
    plt.ylabel("Mean extraction latency per URL, log scale (ms)")
    plt.title("Feature extraction latency: lexical vs. host telemetry")
    for i, v in enumerate(means.values):
        plt.text(i, v, f"{v:.1f} ms", ha="center", va="bottom")
    fig_path = os.path.join(FIGURES_DIR, "latency_comparison.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(results.describe())
    print(f"[ok] wrote {fig_path}")
    return results


def _pct(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def run_fast_path(n_per_class: int = 60, out_path: str = None) -> dict:
    """Time the DNS + SSL checks only, on hosts with no cached DNS or SSL result."""
    from src.features.host_telemetry import get_dns_flags, get_ssl_info
    from src.utils.cache import CACHE_DIR

    fair = pd.read_csv(os.path.join(PROCESSED_DIR, "fair_dataset.csv")).sample(frac=1.0, random_state=77)
    fair = fair[[not os.path.exists(os.path.join(CACHE_DIR, "dns", f"{h}.json")) for h in fair.hostname]]
    fair = fair[[not os.path.exists(os.path.join(CACHE_DIR, "ssl", f"{h}.json")) for h in fair.hostname]]
    picked = pd.concat([fair[fair.label == 1].head(n_per_class), fair[fair.label == 0].head(n_per_class)])

    async def one(host):
        t = time.perf_counter()
        await asyncio.gather(get_dns_flags(host), get_ssl_info(host))
        return (time.perf_counter() - t) * 1000

    async def main():
        return [await one(h) for h in picked.hostname]

    ms = np.array(asyncio.run(main()))
    out = {"dns_plus_ssl_uncached_ms": _pct(ms), "phishing": _pct(ms[:n_per_class]), "benign": _pct(ms[n_per_class:])}
    out_path = out_path or os.path.join(PROCESSED_DIR, "latency_fast_path.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))
    return out


def run_extended(n_per_class: int = 100) -> dict:
    """Larger latency study on hosts that are NOT in the lookup cache.

    Times each host once with an empty cache entry (first pass) and once
    again straight away (cached pass), sequentially, one host at a time.
    """
    from src.utils.cache import CACHE_DIR

    fair = pd.read_csv(os.path.join(PROCESSED_DIR, "fair_dataset.csv"))
    fair = fair.sample(frac=1.0, random_state=11)
    fair = fair[[not os.path.exists(os.path.join(CACHE_DIR, "dns", f"{h}.json")) for h in fair.hostname]]
    picked = pd.concat([fair[fair.label == 1].head(n_per_class), fair[fair.label == 0].head(n_per_class)])
    hosts = [f"http://{h}" for h in picked.hostname]

    lex_ms = []
    for _ in range(20):
        lex_ms += measure_lexical_latency(hosts)
    cold_ms = measure_host_latency(hosts)
    warm_ms = measure_host_latency(hosts)

    df = pd.DataFrame({"hostname": picked.hostname.values, "label": picked.label.values,
                       "host_uncached_ms": cold_ms, "host_cached_ms": warm_ms})
    df.to_csv(os.path.join(PROCESSED_DIR, "latency_extended.csv"), index=False)
    summary = {"lexical_ms": _pct(lex_ms), "host_uncached_ms": _pct(cold_ms), "host_cached_ms": _pct(warm_ms),
               "host_uncached_phishing_ms": _pct(df[df.label == 1].host_uncached_ms),
               "host_uncached_benign_ms": _pct(df[df.label == 0].host_uncached_ms)}
    with open(os.path.join(PROCESSED_DIR, "latency_extended.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "extended":
        run_extended(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    elif len(sys.argv) > 1 and sys.argv[1] == "fast":
        run_fast_path(int(sys.argv[2]) if len(sys.argv) > 2 else 60, sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        sample = int(sys.argv[1]) if len(sys.argv) > 1 else 50
        out = run(sample_size=sample)
        out_path = os.path.join(PROCESSED_DIR, "latency_results.csv")
        out.to_csv(out_path, index=False)
        print(f"[ok] wrote {out_path}")
