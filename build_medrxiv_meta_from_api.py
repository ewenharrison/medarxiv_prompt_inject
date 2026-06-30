#!/usr/bin/env python3
"""
build_medrxiv_meta_from_api.py

Create a meta.csv with posting dates and submission metadata for DOIs found in per_version.csv
by querying the medRxiv API DOI-detail endpoint.

API docs: https://api.medrxiv.org/  (details/[server]/[DOI]/na/json)  :contentReference[oaicite:1]{index=1}

Input:
  - per_version.csv from your local scan (must include a 'doi' column)

Output:
  - meta.csv with columns:
      doi, version, date, title, authors, author_corresponding, author_corresponding_institution,
      type, license, category, jatsxml, abstract, published, server

Usage:
  python build_medrxiv_meta_from_api.py \
    --per_version /home/eharrison/medarxiv_hidden_text/out/per_version.csv \
    --out /home/eharrison/medarxiv_hidden_text/out/meta.csv \
    --sleep 0.2 \
    --server medrxiv

Notes:
- Resumable: if --out exists, it will skip DOIs already present in it.
- Rate limiting: respects 429 with Retry-After (if provided) and exponential backoff.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from typing import Dict, Iterable, List, Set

import pandas as pd
import requests
from tqdm import tqdm
from urllib.parse import quote

DEFAULT_UA = "medrxiv-genai-meta/0.1 (research; contact: you@example.org)"


def existing_dois(meta_csv: str) -> Set[str]:
    if not os.path.exists(meta_csv) or os.path.getsize(meta_csv) == 0:
        return set()
    try:
        df = pd.read_csv(meta_csv, usecols=["doi"])
        return set(df["doi"].astype(str).tolist())
    except Exception:
        return set()


def doi_detail_url(doi: str, server: str) -> str:
    # Keep slashes safe; API examples show DOIs with slashes in the path. :contentReference[oaicite:2]{index=2}
    safe_doi = quote(doi, safe="/")
    return f"https://api.medrxiv.org/details/{server}/{safe_doi}/na/json"


def get_with_backoff(session: requests.Session, url: str, timeout: int, max_retries: int) -> Dict:
    backoff = 1.0
    for attempt in range(max_retries + 1):
        r = session.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    sleep_s = float(ra)
                except Exception:
                    sleep_s = backoff
            else:
                sleep_s = backoff
            time.sleep(sleep_s + random.random() * 0.25)
            backoff = min(backoff * 2, 60.0)
            continue

        # transient server errors
        if 500 <= r.status_code < 600:
            time.sleep(backoff + random.random() * 0.25)
            backoff = min(backoff * 2, 60.0)
            continue

        # non-retryable
        r.raise_for_status()

    raise RuntimeError(f"Failed after retries: {url}")


def append_rows(meta_csv: str, rows: List[Dict]) -> None:
    # Create file with header if missing
    file_exists = os.path.exists(meta_csv) and os.path.getsize(meta_csv) > 0

    fieldnames = [
        "doi", "version", "date", "title", "authors",
        "author_corresponding", "author_corresponding_institution",
        "type", "license", "category", "jatsxml", "abstract",
        "published", "server",
    ]

    with open(meta_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        for row in rows:
            # ensure only expected keys
            out = {k: row.get(k, "") for k in fieldnames}
            w.writerow(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_version", required=True, help="Path to per_version.csv")
    ap.add_argument("--out", required=True, help="Output meta.csv")
    ap.add_argument("--server", default="medrxiv", help="medrxiv or biorxiv")
    ap.add_argument("--sleep", type=float, default=0.2, help="Polite delay between DOIs")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--max_retries", type=int, default=8)
    ap.add_argument("--user_agent", default=DEFAULT_UA)
    ap.add_argument("--max_dois", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    df = pd.read_csv(args.per_version, usecols=["doi"])
    dois = sorted({d for d in df["doi"].astype(str).tolist() if d and d != "nan"})
    if args.max_dois and args.max_dois > 0:
        dois = dois[: args.max_dois]

    done = existing_dois(args.out)
    todo = [d for d in dois if d not in done]

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent, "Accept": "application/json"})

    for doi in tqdm(todo, desc="Fetching medRxiv metadata"):
        url = doi_detail_url(doi, args.server)
        try:
            payload = get_with_backoff(session, url, timeout=args.timeout, max_retries=args.max_retries)
            coll = payload.get("collection", []) or []
            # Each item in collection corresponds to a version. :contentReference[oaicite:3]{index=3}
            rows = []
            for item in coll:
                rows.append({
                    "doi": item.get("doi", doi),
                    "version": item.get("version", ""),
                    "date": item.get("date", ""),
                    "title": item.get("title", ""),
                    "authors": item.get("authors", ""),
                    "author_corresponding": item.get("author_corresponding", ""),
                    "author_corresponding_institution": item.get("author_corresponding_institution", ""),
                    "type": item.get("type", ""),
                    "license": item.get("license", ""),
                    "category": item.get("category", ""),
                    "jatsxml": item.get("jatsxml", ""),
                    "abstract": item.get("abstract", ""),
                    "published": item.get("published", ""),
                    "server": item.get("server", args.server),
                })
            if rows:
                append_rows(args.out, rows)
        except Exception as e:
            # record minimal failure (still resumable)
            append_rows(args.out, [{
                "doi": doi, "version": "", "date": "", "title": "",
                "authors": "", "author_corresponding": "", "author_corresponding_institution": "",
                "type": "", "license": "", "category": "",
                "jatsxml": "", "abstract": "", "published": "", "server": args.server
            }])
        finally:
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Done. Wrote: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
