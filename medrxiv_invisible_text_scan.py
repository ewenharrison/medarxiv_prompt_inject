#!/usr/bin/env python3
"""
medrxiv_invisible_text_scan.py

Heuristic scan of medRxiv PDFs for potentially invisible prompt-injection text intended
to influence LLM-assisted peer review (white-on-white, tiny fonts, transparent text,
PDF hidden render mode, optional content layers, etc.).

Dependencies:
  pip install PyMuPDF requests pandas tqdm

CLI usage:
  python medrxiv_invisible_text_scan.py --start 2025-01-01 --end 2025-01-31

Interactive usage:
  import medrxiv_invisible_text_scan as m
  res = m.run("2025-01-01", "2025-01-31", outdir="./pdf_cache", out="findings.csv")
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import requests
from tqdm import tqdm

# Prefer modern import name; fallback for older environments.
try:
    import pymupdf  # PyMuPDF
except ImportError:
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

DEFAULT_SLEEP_S = 0.2
DEFAULT_UA = "medrxiv-invisible-text-scan/0.3 (research)"


# ------------------------------- Data model ---------------------------------

@dataclass(frozen=True)
class Preprint:
    doi: str
    version: int
    date: str  # YYYY-MM-DD


# ------------------------------ medRxiv API ---------------------------------

def iter_medrxiv_preprints(
    start: str,
    end: str,
    session: requests.Session,
    max_papers: int = 0,
) -> Iterable[Preprint]:
    """
    Official API:
      https://api.medrxiv.org/details/medrxiv/{start}/{end}/{cursor}/json
    cursor increments by 100. Stops when fewer than 100 items returned.
    max_papers=0 => no cap.
    """
    cursor = 0
    seen = 0

    while True:
        url = f"https://api.medrxiv.org/details/medrxiv/{start}/{end}/{cursor}/json"
        r = session.get(url, timeout=60)
        r.raise_for_status()
        payload = r.json()

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
    sleep_s: float = DEFAULT_SLEEP_S,
) -> Optional[str]:
    """
    Download PDF if not cached. Returns local path, or None if missing/404.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, safe_filename(pre.doi, pre.version))

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    url = pdf_url_for(pre.doi, pre.version)
    r = session.get(url, stream=True, timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()

    tmp = path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    os.replace(tmp, path)

    if sleep_s > 0:
        time.sleep(sleep_s)

    return path


# ------------------------------ PDF scanning --------------------------------

def _color_is_near_white(color: Tuple[float, ...]) -> bool:
    # color can be (g,) or (r,g,b) floats in [0,1]
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
    """
    get_texttrace returns spans with "chars" entries like:
      (codepoint, glyph_id, (x,y), (x0,y0,x1,y1))
    """
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
    """
    Span-level scan for suspicious/invisible text properties.
    """
    findings: List[Dict[str, Any]] = []

    doc = pymupdf.open(pdf_path)
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)

        try:
            spans = page.get_texttrace()
        except Exception:
            # get_texttrace not available or failed on this page
            continue

        for sp in spans:
            text = _span_text_from_trace(sp)
            if not text:
                continue

            size = float(sp.get("size", 0.0) or 0.0)
            opacity = float(sp.get("opacity", 1.0) or 1.0)
            stype = int(sp.get("type", -1) if sp.get("type", -1) is not None else -1)  # 3 == hidden
            layer = sp.get("layer")  # Optional content layer name or None
            color = tuple(sp.get("color", ()) or ())
            bbox = sp.get("bbox", None)

            reasons: List[str] = []

            # Hidden render mode
            if stype == 3:
                reasons.append("hidden_render_mode(type=3)")

            # Transparency
            if opacity <= ZEROISH_OPACITY:
                reasons.append(f"near_zero_opacity({opacity:.3f})")
            elif opacity <= LOW_OPACITY:
                reasons.append(f"low_opacity({opacity:.3f})")

            # Tiny fonts (microtext / dot-like)
            if 0 < size <= MICRO_FONT_PT:
                reasons.append(f"micro_font({size:.2f}pt)")
            elif 0 < size <= TINY_FONT_PT:
                reasons.append(f"tiny_font({size:.2f}pt)")

            # Near-white text
            if _color_is_near_white(color):
                reasons.append("near_white_color")

            # Optional content layer
            if layer:
                reasons.append(f"optional_content_layer({layer})")

            # Microtext density (chars packed into tiny bbox)
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


# ------------------------------ Public API ----------------------------------

def run(
    start: str,
    end: str,
    outdir: str = "./medrxiv_pdfs",
    out: str = "findings.csv",
    sleep_s: float = DEFAULT_SLEEP_S,
    max_papers: int = 0,
    user_agent: str = DEFAULT_UA,
) -> Dict[str, Any]:
    """
    Run the scan end-to-end.

    Returns a dict containing:
      - findings_df: span-level findings (pandas.DataFrame)
      - paper_summary_df: paper-level summary (pandas.DataFrame)
      - out_csv: findings CSV path
      - summary_csv: paper summary CSV path
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/pdf,application/json;q=0.9,*/*;q=0.8",
        }
    )

    preprints = list(iter_medrxiv_preprints(start, end, session, max_papers=max_papers))

    all_rows: List[Dict[str, Any]] = []
    paper_rows: List[Dict[str, Any]] = []

    for pre in tqdm(preprints, desc="Scanning medRxiv"):
        pdf_path = download_pdf(pre, outdir, session, sleep_s=sleep_s)

        row = dict(
            doi=pre.doi,
            version=pre.version,
            date=pre.date,
            pdf_downloaded=bool(pdf_path),
            pdf_path=pdf_path,
        )

        if not pdf_path:
            row.update(
                n_findings=0,
                any_prompty=False,
                any_hidden_or_transparent=False,
            )
            paper_rows.append(row)
            continue

        try:
            findings = scan_pdf(pdf_path)
        except Exception:
            findings = []

        for f in findings:
            f["doi"] = pre.doi
            f["version"] = pre.version
            f["date"] = pre.date
            all_rows.append(f)

        row.update(
            n_findings=len(findings),
            any_prompty=any(x.get("prompty") for x in findings),
            any_hidden_or_transparent=any(
                ("hidden_render_mode" in (x.get("reasons") or ""))
                or ("opacity" in (x.get("reasons") or ""))
                for x in findings
            ),
        )
        paper_rows.append(row)

    findings_df = pd.DataFrame(all_rows)
    paper_summary_df = pd.DataFrame(paper_rows)

    findings_df.to_csv(out, index=False)
    summary_csv = os.path.splitext(out)[0] + "_paper_summary.csv"
    paper_summary_df.to_csv(summary_csv, index=False)

    return dict(
        findings_df=findings_df,
        paper_summary_df=paper_summary_df,
        out_csv=os.path.abspath(out),
        summary_csv=os.path.abspath(summary_csv),
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

    args = ap.parse_args(argv)

    res = run(
        start=args.start,
        end=args.end,
        outdir=args.outdir,
        out=args.out,
        sleep_s=args.sleep,
        max_papers=args.max_papers,
        user_agent=args.user_agent,
    )

    downloaded = res["paper_summary_df"]
    downloaded = downloaded[downloaded["pdf_downloaded"] == True]  # noqa: E712

    if len(downloaded) > 0:
        prev_any = (downloaded["n_findings"] > 0).mean()
        prev_prompty = downloaded["any_prompty"].fillna(False).mean()
        print(f"Downloaded PDFs: {len(downloaded)}")
        print(f"Papers with any flagged span: {prev_any:.4f}")
        print(f"Papers with prompt-like text (regex): {prev_prompty:.4f}")
        print(f"Outputs: {res['out_csv']} and {res['summary_csv']}")
    else:
        print("No PDFs downloaded in this interval.")


if __name__ == "__main__":
    main()
