#!/usr/bin/env python3
"""
medrxiv_invisible_text_scan.py

Heuristic scan of medRxiv PDFs for potentially invisible prompt-injection text intended
to influence LLM-assisted peer review (white-on-white, tiny fonts, transparent text,
hidden render mode, optional content layers, etc.).

Key upgrades vs the earlier version:
- Robust retries + exponential backoff for API + PDF download (handles 429/5xx/network issues).
- Streaming CSV writes (findings + paper_summary) so progress is not lost.
- Resume support: skips DOI+version already present in paper_summary output.

Dependencies:
  pip install PyMuPDF requests tqdm pandas

CLI usage:
  python medrxiv_invisible_text_scan.py --start 2025-01-01 --end 2025-01-31

Resume (default on):
  python medrxiv_invisible_text_scan.py --start 2025-01-01 --end 2025-01-31 --out findings.csv

Interactive usage:
  import medrxiv_invisible_text_scan as m
  m.run("2025-01-01", "2025-01-31", outdir="./pdf_cache", out="findings.csv")
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import pandas as pd
import requests
from requests import Response
from tqdm import tqdm

# Prefer modern import name; fallback for older environments.
try:
    import pymupdf  # PyMuPDF
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore


# ---------------------------- Heuristics (tune) -----------------------------

PROMPTY_RE = re.compile(
    r"(?is)\b("
    r"ignore\s+all\s+previous\s+instructions|"
    r"as\s+a\s+language\s+model|"
    r"llm\s+reviewer|"
    r"give\s+a\s+positive\s+review|"
    r"recommend\s+accept(ing)?|"
    r"do\s+not\s+highlight\s+any\s+negatives|"
    r"system\s+prompt|"
    r"follow\s+these\s+instructions"
    r")\b"
)

NEAR_WHITE_MIN = 0.97  # [0,1] close-to-white
TINY_FONT_PT = 3.0
MICRO_FONT_PT = 1.0

LOW_OPACITY = 0.10
ZEROISH_OPACITY = 0.02

MIN_TEXT_LEN = 8

DEFAULT_SLEEP_S = 0.25  # a bit kinder by default
DEFAULT_UA = "medrxiv-invisible-text-scan/0.4 (research)"

# Retry behaviour
DEFAULT_MAX_RETRIES = 8
DEFAULT_BACKOFF_BASE_S = 1.0
DEFAULT_BACKOFF_MAX_S = 90.0
RETRY_STATUS = {429, 500, 502, 503, 504}


# ------------------------------- Data model ---------------------------------

@dataclass(frozen=True)
class Preprint:
    doi: str
    version: int
    date: str  # YYYY-MM-DD


# ------------------------------ Utilities -----------------------------------

def _validate_date(s: str) -> None:
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Date '{s}' must be in YYYY-MM-DD format (e.g., 2025-01-03).") from e


def _key(doi: str, version: int) -> str:
    return f"{doi}v{version}"


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _csv_has_header(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _read_processed_keys_from_paper_summary(summary_csv: str) -> Set[str]:
    """
    Resume mechanism: read existing paper_summary CSV and skip those DOI+version keys.
    """
    if not os.path.exists(summary_csv) or os.path.getsize(summary_csv) == 0:
        return set()
    try:
        df = pd.read_csv(summary_csv, usecols=["doi", "version"])
        out = set(_key(str(d), int(v)) for d, v in zip(df["doi"], df["version"]))
        return out
    except Exception:
        # If summary file is malformed, don't risk skipping.
        return set()


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    stream: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
    backoff_max_s: float = DEFAULT_BACKOFF_MAX_S,
    **kwargs: Any,
) -> Response:
    """
    Robust request wrapper:
    - retries on 429 and 5xx, plus connection/timeouts
    - respects Retry-After header when present
    - exponential backoff with jitter
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.request(method, url, timeout=timeout, stream=stream, **kwargs)
            if resp.status_code in RETRY_STATUS:
                # Determine wait time
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_s = min(float(retry_after), backoff_max_s)
                    except Exception:
                        wait_s = None
                else:
                    wait_s = None

                if wait_s is None:
                    # exponential backoff + jitter
                    wait_s = min(backoff_base_s * (2 ** attempt), backoff_max_s)
                    wait_s *= (0.75 + 0.5 * random.random())

                # Drain body to reuse connection if not streaming
                try:
                    if not stream:
                        _ = resp.text
                except Exception:
                    pass

                if attempt >= max_retries:
                    resp.raise_for_status()
                time.sleep(wait_s)
                continue

            return resp

        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as e:
            last_err = e
            if attempt >= max_retries:
                raise
            wait_s = min(backoff_base_s * (2 ** attempt), backoff_max_s)
            wait_s *= (0.75 + 0.5 * random.random())
            time.sleep(wait_s)

    # Should never reach here
    if last_err:
        raise last_err
    raise RuntimeError("request_with_retry failed unexpectedly")


# ------------------------------ medRxiv API ---------------------------------

def iter_medrxiv_preprints(
    start: str,
    end: str,
    session: requests.Session,
    *,
    max_papers: int = 0,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Iterator[Preprint]:
    """
    Official API:
      https://api.medrxiv.org/details/medrxiv/{start}/{end}/{cursor}/json

    cursor increments by 100. Stops when fewer than 100 items returned.
    max_papers=0 => no cap.

    Uses request_with_retry to survive rate limits / transient errors.
    """
    cursor = 0
    seen = 0

    while True:
        url = f"https://api.medrxiv.org/details/medrxiv/{start}/{end}/{cursor}/json"
        resp = request_with_retry(
            session, "GET", url, timeout=60, stream=False, max_retries=max_retries
        )
        resp.raise_for_status()

        try:
            payload = resp.json()
        except Exception:
            # Treat as transient: sleep a bit and retry via outer loop by decrementing cursor.
            time.sleep(2.0)
            continue

        collection = payload.get("collection", []) or []
        if not collection:
            break

        for item in collection:
            doi = item.get("doi")
            version = item.get("version")
            date = item.get("date")
            if not doi or version is None or not date:
                continue
            yield Preprint(doi=str(doi), version=int(version), date=str(date))
            seen += 1
            if max_papers and seen >= max_papers:
                return

        if len(collection) < 100:
            break

        cursor += 100


def pdf_url_for(doi: str, version: int) -> str:
    return f"https://www.medrxiv.org/content/{doi}v{version}.full.pdf"


def safe_filename(doi: str, version: int) -> str:
    return f"{doi.replace('/', '_')}v{version}.pdf"


def download_pdf(
    pre: Preprint,
    outdir: str,
    session: requests.Session,
    *,
    sleep_s: float = DEFAULT_SLEEP_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[str]:
    """
    Download PDF if not cached. Returns local path, or None if missing/404.
    Robust to transient download failures.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, safe_filename(pre.doi, pre.version))

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    url = pdf_url_for(pre.doi, pre.version)

    try:
        resp = request_with_retry(
            session, "GET", url, timeout=120, stream=True, max_retries=max_retries
        )
    except Exception:
        return None

    if resp.status_code == 404:
        return None
    try:
        resp.raise_for_status()
    except Exception:
        return None

    tmp = path + ".part"
    try:
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return None
    finally:
        try:
            resp.close()
        except Exception:
            pass

    if sleep_s > 0:
        time.sleep(sleep_s)

    return path


# ------------------------------ PDF scanning --------------------------------

def _color_is_near_white(color: Tuple[float, ...]) -> bool:
    if not color:
        return False
    if len(color) == 1:
        return color[0] >= NEAR_WHITE_MIN
    if len(color) >= 3:
        return (
            color[0] >= NEAR_WHITE_MIN
            and color[1] >= NEAR_WHITE_MIN
            and color[2] >= NEAR_WHITE_MIN
        )
    return False


def _span_text_from_trace(span: Dict[str, Any]) -> str:
    chars = span.get("chars", ()) or ()
    out_chars: List[str] = []
    for c in chars:
        try:
            cp = c[0]
            if isinstance(cp, int):
                out_chars.append(chr(cp))
        except Exception:
            continue
    s = "".join(out_chars)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def scan_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    doc = pymupdf.open(pdf_path)

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        try:
            spans = page.get_texttrace()
        except Exception:
            continue

        for sp in spans:
            text = _span_text_from_trace(sp)
            if not text:
                continue

            size = _safe_float(sp.get("size", 0.0), 0.0)
            opacity = _safe_float(sp.get("opacity", 1.0), 1.0)
            stype = int(sp.get("type", -1) if sp.get("type", -1) is not None else -1)  # 3 == hidden
            layer = sp.get("layer")
            color = tuple(sp.get("color", ()) or ())
            bbox = sp.get("bbox", None)

            reasons: List[str] = []

            if stype == 3:
                reasons.append("hidden_render_mode(type=3)")

            if opacity <= ZEROISH_OPACITY:
                reasons.append(f"near_zero_opacity({opacity:.3f})")
            elif opacity <= LOW_OPACITY:
                reasons.append(f"low_opacity({opacity:.3f})")

            if 0 < size <= MICRO_FONT_PT:
                reasons.append(f"micro_font({size:.2f}pt)")
            elif 0 < size <= TINY_FONT_PT:
                reasons.append(f"tiny_font({size:.2f}pt)")

            if _color_is_near_white(color):
                reasons.append("near_white_color")

            if layer:
                reasons.append(f"optional_content_layer({layer})")

            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x0, y0, x1, y1 = bbox
                area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
                if area > 0:
                    density = len(text) / area
                    if size <= MICRO_FONT_PT and len(text) >= MIN_TEXT_LEN:
                        reasons.append(f"microtext_density({density:.2f}/pt^2)")

            prompty = bool(PROMPTY_RE.search(text))
            visibility_suspicious = bool(reasons) and (
                len(text) >= MIN_TEXT_LEN or any("micro_font" in r for r in reasons)
            )

            if visibility_suspicious or prompty:
                findings.append(
                    dict(
                        pdf_path=pdf_path,
                        page=page_index + 1,
                        text=text[:400],
                        text_len=len(text),
                        size_pt=size,
                        opacity=opacity,
                        type=stype,
                        layer=layer,
                        color=color,
                        bbox=bbox,
                        prompty=prompty,
                        reasons=";".join(reasons) if reasons else "",
                    )
                )

    doc.close()
    return findings


# ------------------------------ Streaming output ----------------------------

FINDINGS_FIELDS = [
    "doi", "version", "date",
    "pdf_path", "page", "text", "text_len",
    "size_pt", "opacity", "type", "layer",
    "color", "bbox", "prompty", "reasons",
]

PAPER_FIELDS = [
    "doi", "version", "date",
    "pdf_downloaded", "pdf_path",
    "n_findings", "any_prompty", "any_hidden_or_transparent",
    "status", "error",
]


def _open_csv_writer(path: str, fieldnames: List[str]) -> Tuple[Any, csv.DictWriter]:
    _ensure_parent_dir(path)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if not file_exists:
        w.writeheader()
        f.flush()
    return f, w


# ------------------------------ Public API ----------------------------------

def run(
    start: str,
    end: str,
    *,
    outdir: str = "./medrxiv_pdfs",
    out: str = "findings.csv",
    sleep_s: float = DEFAULT_SLEEP_S,
    max_papers: int = 0,
    user_agent: str = DEFAULT_UA,
    resume: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Run the scan end-to-end with retries + streaming persistence.

    Returns:
      - out_csv, summary_csv (absolute paths)
      - processed_n, skipped_n
    """
    _validate_date(start)
    _validate_date(end)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/pdf,application/json;q=0.9,*/*;q=0.8",
        }
    )

    out_csv = out
    summary_csv = os.path.splitext(out)[0] + "_paper_summary.csv"

    # Resume: load already processed keys from summary CSV
    processed_keys: Set[str] = set()
    if resume:
        processed_keys = _read_processed_keys_from_paper_summary(summary_csv)

    findings_fh, findings_writer = _open_csv_writer(out_csv, FINDINGS_FIELDS)
    paper_fh, paper_writer = _open_csv_writer(summary_csv, PAPER_FIELDS)

    processed_n = 0
    skipped_n = 0

    try:
        preprints_iter = iter_medrxiv_preprints(
            start, end, session, max_papers=max_papers, max_retries=max_retries
        )

        for pre in tqdm(preprints_iter, desc="Scanning medRxiv"):
            k = _key(pre.doi, pre.version)
            if resume and k in processed_keys:
                skipped_n += 1
                continue

            # Default paper row
            paper_row: Dict[str, Any] = dict(
                doi=pre.doi,
                version=pre.version,
                date=pre.date,
                pdf_downloaded=False,
                pdf_path=None,
                n_findings=0,
                any_prompty=False,
                any_hidden_or_transparent=False,
                status="started",
                error="",
            )

            try:
                pdf_path = download_pdf(
                    pre, outdir, session, sleep_s=sleep_s, max_retries=max_retries
                )
                paper_row["pdf_downloaded"] = bool(pdf_path)
                paper_row["pdf_path"] = pdf_path

                if not pdf_path:
                    paper_row["status"] = "pdf_missing_or_download_failed"
                    paper_writer.writerow(paper_row)
                    paper_fh.flush()
                    processed_keys.add(k)
                    processed_n += 1
                    continue

                findings = scan_pdf(pdf_path)

                # Stream findings rows immediately
                any_hidden_or_transparent = False
                any_prompty = False
                for f in findings:
                    f_out = dict(f)
                    f_out["doi"] = pre.doi
                    f_out["version"] = pre.version
                    f_out["date"] = pre.date
                    findings_writer.writerow(f_out)

                    reasons = (f_out.get("reasons") or "")
                    if ("hidden_render_mode" in reasons) or ("opacity" in reasons):
                        any_hidden_or_transparent = True
                    if bool(f_out.get("prompty")):
                        any_prompty = True

                findings_fh.flush()

                paper_row.update(
                    n_findings=len(findings),
                    any_prompty=any_prompty,
                    any_hidden_or_transparent=any_hidden_or_transparent,
                    status="ok",
                )

            except Exception as e:
                # Save paper row even on error, so resume can skip / you can filter later.
                paper_row["status"] = "error"
                paper_row["error"] = f"{type(e).__name__}: {e}"

            # Always write paper summary row
            paper_writer.writerow(paper_row)
            paper_fh.flush()

            processed_keys.add(k)
            processed_n += 1

    finally:
        try:
            findings_fh.close()
        except Exception:
            pass
        try:
            paper_fh.close()
        except Exception:
            pass

    return dict(
        out_csv=os.path.abspath(out_csv),
        summary_csv=os.path.abspath(summary_csv),
        processed_n=processed_n,
        skipped_n=skipped_n,
    )


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--outdir", default="./medrxiv_pdfs")
    ap.add_argument("--out", default="findings.csv")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_S)
    ap.add_argument("--max_papers", type=int, default=0)
    ap.add_argument("--user_agent", type=str, default=DEFAULT_UA)
    ap.add_argument("--no_resume", action="store_true", help="Disable resume/skip behaviour")
    ap.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)

    args = ap.parse_args(argv)

    res = run(
        start=args.start,
        end=args.end,
        outdir=args.outdir,
        out=args.out,
        sleep_s=args.sleep,
        max_papers=args.max_papers,
        user_agent=args.user_agent,
        resume=not args.no_resume,
        max_retries=args.max_retries,
    )

    # Quick summary (robust even if empty)
    try:
        ps = pd.read_csv(res["summary_csv"])
        downloaded = ps[ps.get("pdf_downloaded", False) == True]  # noqa: E712
        ok = ps[ps.get("status", "") == "ok"]
        print(f"Processed (this run): {res['processed_n']}  | Skipped (resume): {res['skipped_n']}")
        print(f"Total rows in paper_summary: {len(ps)}")
        if len(downloaded) > 0:
            prev_any = (downloaded.get("n_findings", 0) > 0).mean()
            prev_prompty = downloaded.get("any_prompty", False).fillna(False).mean()
            print(f"Downloaded PDFs: {len(downloaded)}")
            print(f"Papers with any flagged span: {prev_any:.4f}")
            print(f"Papers with prompt-like text (regex): {prev_prompty:.4f}")
        print(f"Outputs: {res['out_csv']} and {res['summary_csv']}")
    except Exception:
        print(f"Outputs: {res['out_csv']} and {res['summary_csv']}")


if __name__ == "__main__":
    main()
