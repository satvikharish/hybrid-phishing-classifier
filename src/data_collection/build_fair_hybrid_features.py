"""Lexical + host-telemetry features for a balanced, domain-diverse sample of
the fair (hostname-level) dataset.

Usage:
    python -m src.data_collection.build_fair_hybrid_features [per_class] [dataset_csv] [out_csv] [seed]

Defaults: 1500 per class from fair_dataset.csv -> fair_hybrid_features.csv.
Phishing hosts are sampled at most one per registered domain, and the run
order is shuffled so rate-limit failures are not tied to either class.
"""
import asyncio
import os
import sys

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from src.features.host_telemetry import extract_host_features
from src.features.lexical import extract_features

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
CONCURRENCY = 50


def sample_hosts(fair: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    ph = fair[fair.label == 1].sample(frac=1.0, random_state=seed).drop_duplicates("reg_domain")
    ph = ph.head(per_class)
    be = fair[fair.label == 0].sample(n=per_class, random_state=seed)
    return pd.concat([ph, be]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


async def collect(hosts: list) -> pd.DataFrame:
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)

    async def one(host, session):
        async with sem:
            return await extract_host_features(f"http://{host}", session)

    async with aiohttp.ClientSession(connector=connector) as session:
        rows = await tqdm_asyncio.gather(*[one(h, session) for h in hosts])
    return pd.DataFrame(rows)


def run(per_class=1500, dataset_csv="fair_dataset.csv", out_csv="fair_hybrid_features.csv", seed=42) -> pd.DataFrame:
    fair = pd.read_csv(os.path.join(PROCESSED, dataset_csv))
    sample = sample_hosts(fair, per_class, seed)
    lex = pd.DataFrame([extract_features(f"http://{h}") for h in sample.hostname])
    host = asyncio.run(collect(sample.hostname.tolist()))
    host["hostname"] = sample.hostname.values
    host = host.drop(columns=["url"])
    out = pd.concat([sample.reset_index(drop=True), lex], axis=1).merge(host, on="hostname", how="left")
    out.to_csv(os.path.join(PROCESSED, out_csv), index=False)
    print(f"[ok] wrote {len(out)} rows -> {out_csv}")
    return out


if __name__ == "__main__":
    a = sys.argv
    run(int(a[1]) if len(a) > 1 else 1500, a[2] if len(a) > 2 else "fair_dataset.csv",
        a[3] if len(a) > 3 else "fair_hybrid_features.csv", int(a[4]) if len(a) > 4 else 42)
