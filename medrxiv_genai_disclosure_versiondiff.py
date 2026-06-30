#!/usr/bin/env python3
"""
medrxiv_genai_disclosure_versiondiff.py

Folder-based scan of medRxiv PDFs (already on disk) for:
  1) Explicit GenAI/LLM disclosure language (per PDF version)
  2) Version-to-version change metrics (per DOI, consecutive pairs v1->v2->v3...)
     using ONLY per-version extracted features + compact similarity sketches
     (so we do NOT re-read PDFs again during pairing).

This fixes the bug you hit ("ValueError: document closed") by never touching doc.page_count
after doc.close().

Dependencies:
  pip install PyMuPDF pandas tqdm

Recommended:
  - If you have posting dates for each (doi, version), provide --meta_csv with columns:
      doi,version,date   (date = YYYY-MM-DD)
    That lets RSRS include the "rapid" condition.

Usage:
  python medrxiv_genai_disclosure_versiondiff.py \
    --input_dir /home/eharrison/medarxiv_hidden_text/medrxiv_pdfs \
    --outdir /home/eharrison/medarxiv_hidden_text/out \
    --workers 12 \
    --resume

Outputs in --outdir:
  - per_version.csv
  - per_pair.csv
  - run_summary.txt

Notes:
- Filename pattern like 10.1101_2025.07.23.25331971v1.pdf is supported.
- This is NOT "LLM detection". RSRS is framed as "rapid stylistic revision signature".
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import os
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    import pymupdf  # PyMuPDF
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore


# ----------------------------- Disclosure lexicon ----------------------------

DISCLOSURE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("chatgpt", re.compile(r"(?i)\bchatgpt\b")),
    ("gpt4", re.compile(r"(?i)\bgpt[\-\s]?4\b")),
    ("gpt", re.compile(r"(?i)\bgpt[\-\s]?(3\.5|4|4o|5|5\.?\d*)\b")),
    ("openai", re.compile(r"(?i)\bopenai\b")),
    ("llm", re.compile(r"(?i)\b(large\s+language\s+model|llm(s)?)\b")),
    ("genai", re.compile(r"(?i)\b(generative\s+ai|genai)\b")),
    ("ai_assisted_writing", re.compile(r"(?i)\b(ai[-\s]*assisted|ai[-\s]*aided)\s+(writing|drafting|editing)\b")),
    ("ai_tool_writing", re.compile(r"(?i)\b(used|utili[sz]ed)\s+(an?\s+)?(ai|llm)\s+(tool|system|model)\b")),
    ("grammarly", re.compile(r"(?i)\bgrammarly\b")),
    ("deepl", re.compile(r"(?i)\bdeepl\b")),
    ("quillbot", re.compile(r"(?i)\bquillbot\b")),
    ("writefull", re.compile(r"(?i)\bwritefull\b")),
    ("claude", re.compile(r"(?i)\bclaude\b")),
    ("gemini", re.compile(r"(?i)\bgemini\b")),
    ("copilot", re.compile(r"(?i)\bcopilot\b")),
]

DISCLOSURE_CONTEXT_RE = re.compile(
    r"(?i)\b(acknowledg(e)?ments?|we\s+thank|assisted|edited|drafted|proofread|"
    r"language\s+editing|writing\s+assistance|generat(ed|ive)|llm|chatgpt|gpt)\b"
)


# ----------------------------- Change metrics --------------------------------

STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","from","has","have","if","in","into",
    "is","it","its","of","on","or","that","the","their","then","there","these","they","this",
    "to","was","were","will","with","we","our","you","your","not","can","may","might","should"
}

SENT_SPLIT_RE = re.compile(r"[.!?]+")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s


def casefold_text(s: str) -> str:
    return normalize_text(s).casefold()


def tokenize_words(s: str) -> List[str]:
    return [m.group(0).casefold() for m in WORD_RE.finditer(s or "")]


def content_words(s: str) -> List[str]:
    toks = tokenize_words(s)
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]


def sentence_lengths(s: str) -> Tuple[float, float, int]:
    if not s:
        return (0.0, 0.0, 0)
    parts = [p.strip() for p in SENT_SPLIT_RE.split(s) if p.strip()]
    n = len(parts)
    if n == 0:
        return (0.0, 0.0, 0)
    lengths = [len(content_words(p)) for p in parts]
    mean = sum(lengths) / n
    var = sum((x - mean) ** 2 for x in lengths) / n
    return (mean, var, n)


def punctuation_density(s: str) -> float:
    if not s:
        return 0.0
    punct = sum(1 for c in s if c in ",;:()")
    n = max(1, len(s))
    return punct * 1000.0 / n


def syllable_count_word(w: str) -> int:
    w = w.casefold()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease(s: str) -> float:
    if not s:
        return 0.0
    sentences = max(1, len([p for p in SENT_SPLIT_RE.split(s) if p.strip()]))
    words = [m.group(0) for m in WORD_RE.finditer(s)]
    n_words = len(words)
    if n_words == 0:
        return 0.0
    syllables = sum(syllable_count_word(w) for w in words)
    return 206.835 - 1.015 * (n_words / sentences) - 84.6 * (syllables / n_words)


def extract_headings(text: str) -> List[str]:
    if not text:
        return []
    common = {
        "abstract","introduction","methods","materials and methods","results","discussion",
        "conclusion","conclusions","references","acknowledgements","acknowledgments",
        "conflicts of interest","competing interests","data availability","ethics",
    }
    out: List[str] = []
    for line in text.splitlines():
        ln = line.strip()
        if not ln or len(ln) > 70:
            continue
        lcf = ln.casefold()
        if lcf in common:
            out.append(lcf)
            continue
        if ln.endswith(":") and re.fullmatch(r"[A-Za-z0-9 ,\-:/()]+", ln):
            out.append(lcf.rstrip(":"))
            continue
        letters = [c for c in ln if c.isalpha()]
        if letters and (sum(1 for c in letters if c.isupper()) / len(letters) > 0.85) and len(letters) >= 6:
            out.append(lcf)
    # dedupe preserving order
    seen = set()
    dedup: List[str] = []
    for h in out:
        if h not in seen:
            dedup.append(h)
            seen.add(h)
    return dedup


def jaccard_set(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ----------------------------- KMV sketch (compact similarity) ----------------

def hash64(s: str) -> int:
    # stable 64-bit hash from sha1
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
    return int(h, 16)


def kmv_signature(text: str, k: int = 64, max_tokens: int = 200_000) -> List[int]:
    """
    K-minimum values (KMV) sketch of the set of content words.
    Approx similarity via Jaccard on the sketches (good enough for a proxy).

    k=64 keeps per-row storage ~ (64 * 16hex + commas) ~ ~1.1KB.
    """
    toks = content_words(text)
    if not toks:
        return []

    # cap to avoid pathological memory/time on huge texts
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]

    # get unique tokens without building an enormous set first:
    # we do a simple seen set but it will still be large; acceptable in practice.
    seen: set = set()

    # keep a max-heap of size k for smallest hashes
    heap: List[int] = []

    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        hv = hash64(t)
        if len(heap) < k:
            heapq.heappush(heap, -hv)  # max-heap via negation
        else:
            if hv < -heap[0]:
                heapq.heapreplace(heap, -hv)

    sig = sorted([-x for x in heap])
    return sig


def sig_to_str(sig: List[int]) -> str:
    return ",".join(f"{x:016x}" for x in sig)


def str_to_sig(s: str) -> List[int]:
    if not s or not isinstance(s, str):
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p, 16))
        except Exception:
            continue
    return out


def kmv_jaccard(sig1: List[int], sig2: List[int]) -> float:
    if not sig1 or not sig2:
        return 0.0
    s1 = set(sig1)
    s2 = set(sig2)
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


# ----------------------------- DOI/version parsing ---------------------------

# Works for: 10.1101_2025.07.23.25331971v1.pdf
DOI_V_RE_US = re.compile(r"(10\.\d{4,9}_[^ ]+?)v(\d+)\.pdf$", re.I)
# Also handle embedded "10.1101/....v2" if present
DOI_V_RE_SLASH = re.compile(r"(10\.\d{4,9}/[^\s/]+?)v(\d+)\b", re.I)


def infer_doi_version_from_filename(path: str) -> Tuple[Optional[str], Optional[int]]:
    base = os.path.basename(path)

    m = DOI_V_RE_US.search(base)
    if m:
        doi_us = m.group(1)
        v = int(m.group(2))
        doi = doi_us.replace("_", "/", 1) if doi_us.startswith("10.") else doi_us.replace("_", "/")
        return doi, v

    m = DOI_V_RE_SLASH.search(base)
    if m:
        return m.group(1), int(m.group(2))

    return None, None


# ----------------------------- PDF extraction --------------------------------

def extract_pdf_text(pdf_path: str, max_chars: int = 2_000_000) -> Tuple[str, int, int]:
    """
    Returns (text, page_count, extracted_chars).
    IMPORTANT: page_count is captured BEFORE closing doc.
    """
    doc = pymupdf.open(pdf_path)
    page_count = int(doc.page_count)

    parts: List[str] = []
    total = 0
    for i in range(page_count):
        try:
            t = doc.load_page(i).get_text("text") or ""
        except Exception:
            t = ""
        if t:
            parts.append(t)
            total += len(t)
        if total >= max_chars:
            break

    doc.close()
    text = normalize_text("\n".join(parts))
    return text, page_count, len(text)


def find_disclosures(text: str) -> Dict[str, Any]:
    t = text or ""
    lcf = casefold_text(t)
    hits: List[str] = []
    for name, rx in DISCLOSURE_PATTERNS:
        if rx.search(t):
            hits.append(name)

    context = ""
    if hits:
        idx = None
        for _, rx in DISCLOSURE_PATTERNS:
            m = rx.search(t)
            if m:
                idx = m.start()
                break
        if idx is not None:
            left = max(0, idx - 250)
            right = min(len(t), idx + 250)
            context = re.sub(r"\s+", " ", t[left:right]).strip()

    return {
        "disclosure_any": bool(hits),
        "disclosure_terms": ";".join(hits),
        "disclosure_context": context[:400],
        "disclosureish_language": bool(hits) or bool(DISCLOSURE_CONTEXT_RE.search(lcf)),
    }


def compute_version_features(text: str) -> Dict[str, Any]:
    mean_sl, var_sl, n_sent = sentence_lengths(text)
    headings = extract_headings(text)
    return {
        "chars": len(text),
        "words": len(tokenize_words(text)),
        "content_words": len(content_words(text)),
        "sentences": n_sent,
        "mean_sentence_len": mean_sl,
        "var_sentence_len": var_sl,
        "punct_density_per_1k_chars": punctuation_density(text),
        "flesch_reading_ease": flesch_reading_ease(text),
        "headings": ";".join(headings),
        "headings_n": len(headings),
    }


# ----------------------------- Resume / sharding -----------------------------

def iter_pdfs(input_dir: str) -> Iterable[str]:
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(".pdf"):
            yield os.path.join(input_dir, fn)


def sha1_mod(s: str, mod: int) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def read_done_paths_ok(per_version_csv: str) -> set:
    """
    Resume semantics:
      - Skip PDFs already scanned successfully (status == ok)
      - Re-scan PDFs previously in error (so bugfixes or transient issues can recover)
    """
    if not os.path.exists(per_version_csv) or os.path.getsize(per_version_csv) == 0:
        return set()
    try:
        df = pd.read_csv(per_version_csv, usecols=["pdf_path", "status"])
        df = df[df["status"].astype(str) == "ok"]
        return set(df["pdf_path"].astype(str).tolist())
    except Exception:
        return set()


# ----------------------------- Worker ----------------------------------------

def scan_one_pdf(args: Tuple[str, int, int]) -> Dict[str, Any]:
    pdf_path, max_chars, kmv_k = args

    doi, ver = infer_doi_version_from_filename(pdf_path)

    row: Dict[str, Any] = {
        "pdf_path": pdf_path,
        "filename": os.path.basename(pdf_path),
        "doi": doi or "",
        "version": int(ver) if ver is not None else "",
        "status": "ok",
        "error": "",
        "page_count": "",
        "extracted_chars": "",
    }

    try:
        text, page_count, extracted_chars = extract_pdf_text(pdf_path, max_chars=max_chars)
        row["page_count"] = page_count
        row["extracted_chars"] = extracted_chars

        row.update(find_disclosures(text))

        feats = compute_version_features(text)
        row.update({f"feat_{k}": v for k, v in feats.items()})

        # compact similarity sketch + fingerprint
        sig = kmv_signature(text, k=kmv_k)
        row["kmv_k"] = kmv_k
        row["kmv_sig"] = sig_to_str(sig)
        row["text_sha1_prefix"] = hashlib.sha1(text[:200000].encode("utf-8")).hexdigest()

    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"

    return row


# ----------------------------- RSRS flag -------------------------------------

def rsrs_flag(
    kmv_sim: float,
    delta_days: Optional[float],
    len_pct_change: float,
    delta_flesch: float,
    *,
    rapid_days: int,
    min_similarity: float,
    max_abs_len_change: float,
    min_flesch_gain: float,
) -> bool:
    timing_ok = True
    if delta_days is not None:
        timing_ok = delta_days <= rapid_days
    return bool(
        timing_ok
        and kmv_sim >= min_similarity
        and abs(len_pct_change) <= max_abs_len_change
        and delta_flesch >= min_flesch_gain
    )


# ----------------------------- Main ------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing PDFs")
    ap.add_argument("--outdir", default="./out", help="Output folder")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--max_chars", type=int, default=2_000_000, help="Max extracted chars per PDF")
    ap.add_argument("--kmv_k", type=int, default=64, help="KMV sketch size (64 default; larger = better similarity, bigger CSV)")
    ap.add_argument("--resume", action="store_true", help="Resume: skip PDFs already OK in per_version.csv")
    ap.add_argument("--shard_count", type=int, default=0, help="Number of shards (0 = none)")
    ap.add_argument("--shard_index", type=int, default=0, help="Shard index (0..shard_count-1)")
    ap.add_argument("--max_files", type=int, default=0, help="Cap PDFs scanned (0 = no cap)")

    ap.add_argument("--meta_csv", default="", help="Optional CSV: doi,version,date (YYYY-MM-DD)")
    ap.add_argument("--rapid_days", type=int, default=14)
    ap.add_argument("--min_similarity", type=float, default=0.85)
    ap.add_argument("--max_abs_len_change", type=float, default=0.15)
    ap.add_argument("--min_flesch_gain", type=float, default=5.0)

    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    per_version_csv = os.path.join(outdir, "per_version.csv")
    per_pair_csv = os.path.join(outdir, "per_pair.csv")
    run_summary_txt = os.path.join(outdir, "run_summary.txt")

    pdfs = sorted(list(iter_pdfs(input_dir)))

    if args.shard_count and args.shard_count > 0:
        if not (0 <= args.shard_index < args.shard_count):
            raise ValueError("--shard_index must be in [0, shard_count-1]")
        pdfs = [p for p in pdfs if sha1_mod(p, args.shard_count) == args.shard_index]

    if args.max_files and args.max_files > 0:
        pdfs = pdfs[: args.max_files]

    done_ok = read_done_paths_ok(per_version_csv) if args.resume else set()
    to_scan = [p for p in pdfs if p not in done_ok]

    # ---- Scan PDFs (parallel) ----
    new_rows: List[Dict[str, Any]] = []
    if to_scan:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(scan_one_pdf, (p, args.max_chars, args.kmv_k)) for p in to_scan]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Scanning PDFs"):
                new_rows.append(fut.result())

        df_new = pd.DataFrame(new_rows)

        if os.path.exists(per_version_csv) and os.path.getsize(per_version_csv) > 0:
            df_old = pd.read_csv(per_version_csv)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        # keep last result per pdf_path (so re-scans overwrite earlier errors)
        df_all = df_all.sort_values(by=["pdf_path"]).drop_duplicates(subset=["pdf_path"], keep="last")
        df_all.to_csv(per_version_csv, index=False)
    else:
        df_all = pd.read_csv(per_version_csv) if os.path.exists(per_version_csv) else pd.DataFrame()

    # ---- Build pairs from per_version only (no re-reading PDFs) ----
    if df_all.empty:
        with open(run_summary_txt, "w", encoding="utf-8") as f:
            f.write("No data in per_version.csv\n")
        print("No data.")
        return

    meta = None
    if args.meta_csv:
        meta = pd.read_csv(args.meta_csv)
        meta["doi"] = meta["doi"].astype(str)
        meta["version"] = pd.to_numeric(meta["version"], errors="coerce").astype("Int64")
        meta["date"] = meta["date"].astype(str)

    df_ok = df_all[df_all["status"].astype(str) == "ok"].copy()
    df_ok["doi"] = df_ok["doi"].astype(str)
    df_ok["version_num"] = pd.to_numeric(df_ok["version"], errors="coerce")
    df_ok = df_ok[~df_ok["doi"].eq("") & df_ok["version_num"].notna()].copy()
    df_ok["version_num"] = df_ok["version_num"].astype(int)

    # parse headings and kmv sig
    df_ok["headings_list"] = df_ok.get("feat_headings", "").astype(str).apply(lambda x: [h for h in x.split(";") if h])
    df_ok["kmv_list"] = df_ok.get("kmv_sig", "").astype(str).apply(str_to_sig)

    pairs: List[Dict[str, Any]] = []

    for doi, g in tqdm(df_ok.sort_values(["doi", "version_num"]).groupby("doi", sort=False), desc="Building version pairs"):
        g = g.sort_values("version_num")
        vers = g["version_num"].tolist()

        for i in range(len(vers) - 1):
            v1 = vers[i]
            v2 = vers[i + 1]
            r1 = g.iloc[i]
            r2 = g.iloc[i + 1]

            # optional dates
            d1 = d2 = ""
            delta_days: Optional[float] = None
            if meta is not None:
                m1 = meta[(meta["doi"] == doi) & (meta["version"] == v1)]
                m2 = meta[(meta["doi"] == doi) & (meta["version"] == v2)]
                if len(m1) == 1 and len(m2) == 1:
                    d1 = str(m1.iloc[0]["date"])
                    d2 = str(m2.iloc[0]["date"])
                    try:
                        dt1 = datetime.strptime(d1, "%Y-%m-%d")
                        dt2 = datetime.strptime(d2, "%Y-%m-%d")
                        delta_days = float((dt2 - dt1).days)
                    except Exception:
                        delta_days = None

            # similarities
            head_j = jaccard_set(r1["headings_list"], r2["headings_list"])
            kmv_sim = kmv_jaccard(r1["kmv_list"], r2["kmv_list"])
            sim = max(kmv_sim, head_j)

            # deltas
            w1 = float(r1.get("feat_words", 0) or 0)
            w2 = float(r2.get("feat_words", 0) or 0)
            w1 = max(1.0, w1)
            len_pct_change = (w2 - w1) / w1

            f1 = float(r1.get("feat_flesch_reading_ease", 0.0) or 0.0)
            f2 = float(r2.get("feat_flesch_reading_ease", 0.0) or 0.0)
            delta_flesch = f2 - f1

            rsrs = rsrs_flag(
                kmv_sim=sim,
                delta_days=delta_days,
                len_pct_change=len_pct_change,
                delta_flesch=delta_flesch,
                rapid_days=args.rapid_days,
                min_similarity=args.min_similarity,
                max_abs_len_change=args.max_abs_len_change,
                min_flesch_gain=args.min_flesch_gain,
            )

            pairs.append(
                {
                    "doi": doi,
                    "v_from": v1,
                    "v_to": v2,
                    "pdf_from": str(r1["pdf_path"]),
                    "pdf_to": str(r2["pdf_path"]),
                    "date_from": d1,
                    "date_to": d2,
                    "delta_days": delta_days if delta_days is not None else "",
                    "heading_jaccard": head_j,
                    "kmv_jaccard": kmv_sim,
                    "sim_used": sim,
                    "len_pct_change": len_pct_change,
                    "delta_flesch": delta_flesch,
                    "rsrs": bool(rsrs),
                    "disclosed_v_to": bool(r2.get("disclosure_any", False)),
                    "disclosure_terms_v_to": str(r2.get("disclosure_terms", "")),
                    "disclosureish_v_to": bool(r2.get("disclosureish_language", False)),
                }
            )

    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(per_pair_csv, index=False)

    # ---- Summary ----
    n_total = len(df_all)
    n_ok = int((df_all["status"].astype(str) == "ok").sum())
    n_disc = int((df_all.get("disclosure_any", pd.Series([False] * len(df_all))) == True).sum())  # noqa: E712
    n_pairs = len(df_pairs)
    n_rsrs = int((df_pairs.get("rsrs", pd.Series([False] * len(df_pairs))) == True).sum()) if n_pairs else 0  # noqa: E712

    with open(run_summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Input dir: {input_dir}\n")
        f.write(f"Output dir: {outdir}\n")
        f.write(f"PDFs in scope: {len(pdfs)}\n")
        f.write(f"Scanned this run: {len(to_scan)}\n")
        f.write(f"Total versions (rows): {n_total}\n")
        f.write(f"OK versions: {n_ok}\n")
        f.write(f"Disclosure-any versions: {n_disc}\n")
        f.write(f"Pairs built: {n_pairs}\n")
        f.write(f"RSRS-positive pairs: {n_rsrs}\n\n")
        f.write("RSRS thresholds:\n")
        f.write(f"  rapid_days <= {args.rapid_days} (only if meta dates provided)\n")
        f.write(f"  min_similarity >= {args.min_similarity}\n")
        f.write(f"  max_abs_len_change <= {args.max_abs_len_change}\n")
        f.write(f"  min_flesch_gain >= {args.min_flesch_gain}\n\n")
        f.write("KMV sketch:\n")
        f.write(f"  kmv_k = {args.kmv_k}\n")

    print("Done.")
    print(f"  {per_version_csv}")
    print(f"  {per_pair_csv}")
    print(f"  {run_summary_txt}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
