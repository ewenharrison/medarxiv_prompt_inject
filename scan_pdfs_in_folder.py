#!/usr/bin/env python3
"""
scan_pdfs_in_folder.py

Scan local PDFs for potentially invisible prompt-injection content that could affect
LLM-assisted peer review.

What it checks:
- Invisible / suspicious text spans via PyMuPDF get_texttrace() (hidden render mode, opacity,
  tiny font, near-white color, layer text, microtext density)
- Prompt-like text patterns (regex) across:
  * span text
  * annotations
  * form fields/widgets
  * PDF metadata (Info + XMP)
  * embedded-file metadata (name/desc)
- Presence of embedded files (attachments), layers/OCGs, annotations/widgets counts

Persistence & robustness:
- Streaming CSV writes (append + flush) -> no progress loss on transient failures
- Resume: reads existing *_paper_summary.csv and skips already-processed pdf paths
- Sharding: --shard_count K --shard_index i processes only 1/K of files

Dependencies:
  pip install PyMuPDF tqdm pandas

Usage:
  python scan_pdfs_in_folder.py --input_dir /path/to/pdfs --out findings.csv

Resume (default):
  python scan_pdfs_in_folder.py --input_dir /path/to/pdfs --out findings.csv

Disable resume:
  python scan_pdfs_in_folder.py --input_dir /path/to/pdfs --out findings.csv --no_resume

Shard across 4 runs:
  python scan_pdfs_in_folder.py --input_dir /path --out findings.csv --shard_count 4 --shard_index 0
  python scan_pdfs_in_folder.py --input_dir /path --out findings.csv --shard_count 4 --shard_index 1
  ...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import pandas as pd
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


# ------------------------------ Output schema -------------------------------

FINDINGS_FIELDS = [
    "pdf_path", "source", "page",
    "text", "text_len",
    "size_pt", "opacity", "type", "layer",
    "color", "bbox",
    "prompty", "reasons",
    # extra context fields:
    "annot_subtype", "annot_title", "annot_subject",
    "widget_name", avoid_key_error := "widget_type",  # keep stable column name even if blank
    "meta_field", "embedded_filename",
]

PAPER_FIELDS = [
    "pdf_path",
    "file_size_bytes",
    "mtime_iso",
    "status",
    "error",
    "n_pages",
    "n_findings_total",
    "n_span_findings",
    "n_annotation_findings",
    "n_widget_findings",
    "n_metadata_findings",
    "n_embedded_findings",
    "any_prompty",
    "any_hidden_or_transparent",
    "any_layer_text",
    "has_annotations",
    "has_widgets",
    "embedded_file_count",
    "ocg_count",
]


# ------------------------------ Small utilities -----------------------------

def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _open_csv_writer(path: str, fieldnames: List[str]) -> Tuple[Any, csv.DictWriter]:
    _ensure_parent_dir(path)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if not file_exists:
        w.writeheader()
        f.flush()
    return f, w


def _read_processed_paths_from_summary(summary_csv: str) -> Set[str]:
    """
    Resume mechanism: read existing paper summary and skip those pdf_path values.
    """
    if not os.path.exists(summary_csv) or os.path.getsize(summary_csv) == 0:
        return set()
    out: Set[str] = set()
    try:
        with open(summary_csv, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if "pdf_path" not in (r.fieldnames or []):
                return set()
            for row in r:
                p = row.get("pdf_path")
                if p:
                    out.add(p)
    except Exception:
        return set()
    return out


def _file_mtime_iso(path: str) -> str:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    except Exception:
        return ""


def _sha1_mod(s: str, mod: int) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def _color_is_near_white(color: Tuple[float, ...]) -> bool:
    if not color:
        return False
    if len(color) == 1:
        return color[0] >= NEAR_WHITE_MIN
    if len(color) >= 3:
        return color[0] >= NEAR_WHITE_MIN and color[1] >= NEAR_WHITE_MIN and color[2] >= NEAR_WHITE_MIN
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


def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    try:
        t = str(s)
    except Exception:
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_prompty(text: str) -> bool:
    return bool(text) and bool(PROMPTY_RE.search(text))


# ------------------------------ File discovery ------------------------------

def iter_pdfs(input_dir: str, recursive: bool = True) -> Iterator[str]:
    """
    Yield absolute paths to PDFs in input_dir.
    """
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Not a directory: {input_dir}")

    if recursive:
        for root, _, files in os.walk(input_dir):
            for fn in files:
                if fn.lower().endswith(".pdf"):
                    yield os.path.join(root, fn)
    else:
        for fn in os.listdir(input_dir):
            if fn.lower().endswith(".pdf"):
                yield os.path.join(input_dir, fn)


# ------------------------------ Expanded checks -----------------------------

def scan_spans(doc: Any, pdf_path: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

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

            size = float(sp.get("size", 0.0) or 0.0)
            opacity = float(sp.get("opacity", 1.0) or 1.0)
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

            prompty = _is_prompty(text)
            visibility_suspicious = bool(reasons) and (
                len(text) >= MIN_TEXT_LEN or any("micro_font" in r for r in reasons)
            )

            if visibility_suspicious or prompty:
                findings.append(
                    dict(
                        pdf_path=pdf_path,
                        source="span",
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

    return findings


def scan_annotations(doc: Any, pdf_path: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        try:
            annots = page.annots()
        except Exception:
            annots = None

        if not annots:
            continue

        for a in annots:
            try:
                info = a.info or {}
            except Exception:
                info = {}

            subtype = _norm_text(info.get("subtype"))
            title = _norm_text(info.get("title"))
            subject = _norm_text(info.get("subject"))
            content = _norm_text(info.get("content")) or _norm_text(info.get("contents"))

            blob = " | ".join([subtype, title, subject, content]).strip(" |")
            if not blob:
                continue

            prompty = _is_prompty(blob)
            # Many extractors ingest annotation text even if visually subtle; we flag prompty ones.
            if prompty:
                findings.append(
                    dict(
                        pdf_path=pdf_path,
                        source="annotation",
                        page=page_index + 1,
                        text=blob[:400],
                        text_len=len(blob),
                        prompty=True,
                        reasons="annotation_text_prompty",
                        annot_subtype=subtype,
                        annot_title=title,
                        annot_subject=subject,
                    )
                )
    return findings


def scan_widgets(doc: Any, pdf_path: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        widgets = []
        try:
            widgets = page.widgets() or []
        except Exception:
            widgets = []

        for w in widgets:
            # Field values can be ingested by some parsers; treat as another channel.
            try:
                name = _norm_text(getattr(w, "field_name", "") or "")
                ftype = _norm_text(getattr(w, "field_type", "") or "")
                value = _norm_text(getattr(w, "field_value", "") or "")
            except Exception:
                continue

            blob = " | ".join([name, ftype, value]).strip(" |")
            if not blob:
                continue

            prompty = _is_prompty(blob)
            if prompty:
                findings.append(
                    dict(
                        pdf_path=pdf_path,
                        source="widget",
                        page=page_index + 1,
                        text=blob[:400],
                        text_len=len(blob),
                        prompty=True,
                        reasons="widget_text_prompty",
                        widget_name=name,
                        widget_type=ftype,
                    )
                )
    return findings


def scan_metadata(doc: Any, pdf_path: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    # Info dict (Title/Subject/Keywords/Author/Creator/Producer etc.)
    try:
        md = doc.metadata or {}
    except Exception:
        md = {}

    for k, v in (md or {}).items():
        val = _norm_text(v)
        if not val:
            continue
        if _is_prompty(val):
            findings.append(
                dict(
                    pdf_path=pdf_path,
                    source="metadata",
                    page=0,
                    text=f"{k}: {val}"[:400],
                    text_len=len(val),
                    prompty=True,
                    reasons="metadata_prompty",
                    meta_field=k,
                )
            )

    # XMP metadata (XML) – scan raw text (simple but effective)
    try:
        xmp = doc.get_xml_metadata()  # may raise if absent
    except Exception:
        xmp = ""

    xmp = _norm_text(xmp)
    if xmp and _is_prompty(xmp):
        findings.append(
            dict(
                pdf_path=pdf_path,
                source="xmp",
                page=0,
                text=xmp[:400],
                text_len=len(xmp),
                prompty=True,
                reasons="xmp_prompty",
                meta_field="xmp",
            )
        )

    return findings


def scan_embedded_files(doc: Any, pdf_path: str) -> Tuple[int, List[Dict[str, Any]]]:
    findings: List[Dict[str, Any]] = []
    count = 0
    try:
        count = int(doc.embfile_count())
    except Exception:
        count = 0

    if count <= 0:
        return 0, []

    # Presence alone is noteworthy; additionally scan names/descriptions for prompty text.
    for i in range(count):
        try:
            info = doc.embfile_info(i) or {}
        except Exception:
            info = {}

        name = _norm_text(info.get("filename") or info.get("name") or f"index_{i}")
        desc = _norm_text(info.get("desc") or info.get("description") or "")
        blob = " | ".join([name, desc]).strip(" |")

        prompty = _is_prompty(blob)
        reasons = ["embedded_file_present"]
        if prompty:
            reasons.append("embedded_file_text_prompty")

        findings.append(
            dict(
                pdf_path=pdf_path,
                source="embedded_file",
                page=0,
                text=blob[:400] if blob else name,
                text_len=len(blob) if blob else len(name),
                prompty=prompty,
                reasons=";".join(reasons),
                embedded_filename=name,
            )
        )

    return count, findings


def get_ocg_count(doc: Any) -> int:
    # Optional content groups (layers). Not all PDFs have this.
    try:
        ocgs = doc.get_ocgs()
        if isinstance(ocgs, dict):
            return len(ocgs)
    except Exception:
        pass
    return 0


# ------------------------------ Orchestration -------------------------------

def run(
    input_dir: str,
    *,
    out: str = "findings.csv",
    recursive: bool = True,
    resume: bool = True,
    max_files: int = 0,
    shard_count: int = 0,
    shard_index: int = 0,
) -> Dict[str, Any]:
    """
    Scan PDFs in a folder. Writes:
      - out (findings)
      - out with suffix _paper_summary.csv

    Resume (default): skips pdf_path already present in paper summary.
    Sharding: if shard_count>0, only processes files where sha1(path) % shard_count == shard_index.
    """
    input_dir = os.path.abspath(input_dir)
    out_csv = out
    summary_csv = os.path.splitext(out)[0] + "_paper_summary.csv"

    processed_paths: Set[str] = set()
    if resume:
        processed_paths = _read_processed_paths_from_summary(summary_csv)

    findings_fh, findings_writer = _open_csv_writer(out_csv, FINDINGS_FIELDS)
    paper_fh, paper_writer = _open_csv_writer(summary_csv, PAPER_FIELDS)

    processed_n = 0
    skipped_n = 0
    errored_n = 0

    try:
        pdfs = list(iter_pdfs(input_dir, recursive=recursive))

        # sharding filter
        if shard_count and shard_count > 0:
            if not (0 <= shard_index < shard_count):
                raise ValueError("--shard_index must be in [0, shard_count-1]")
            pdfs = [p for p in pdfs if _sha1_mod(p, shard_count) == shard_index]

        pdfs.sort()

        if max_files and max_files > 0:
            pdfs = pdfs[:max_files]

        for pdf_path in tqdm(pdfs, desc="Scanning PDFs"):
            if resume and pdf_path in processed_paths:
                skipped_n += 1
                continue

            paper_row: Dict[str, Any] = dict(
                pdf_path=pdf_path,
                file_size_bytes=os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0,
                mtime_iso=_file_mtime_iso(pdf_path),
                status="started",
                error="",
                n_pages=0,
                n_findings_total=0,
                n_span_findings=0,
                n_annotation_findings=0,
                n_widget_findings=0,
                n_metadata_findings=0,
                n_embedded_findings=0,
                any_prompty=False,
                any_hidden_or_transparent=False,
                any_layer_text=False,
                has_annotations=False,
                has_widgets=False,
                embedded_file_count=0,
                ocg_count=0,
            )

            try:
                doc = pymupdf.open(pdf_path)
                paper_row["n_pages"] = int(doc.page_count)
                paper_row["ocg_count"] = get_ocg_count(doc)

                # Expanded checks
                meta_findings = scan_metadata(doc, pdf_path)
                emb_count, emb_findings = scan_embedded_files(doc, pdf_path)
                annot_findings = scan_annotations(doc, pdf_path)
                widget_findings = scan_widgets(doc, pdf_path)
                span_findings = scan_spans(doc, pdf_path)

                # Derived booleans
                paper_row["embedded_file_count"] = emb_count
                paper_row["has_annotations"] = len(annot_findings) > 0  # prompty-only by current logic
                paper_row["has_widgets"] = len(widget_findings) > 0     # prompty-only by current logic

                all_findings = meta_findings + emb_findings + annot_findings + widget_findings + span_findings

                # Counters
                paper_row["n_metadata_findings"] = len(meta_findings)
                paper_row["n_embedded_findings"] = len(emb_findings)
                paper_row["n_annotation_findings"] = len(annot_findings)
                paper_row["n_widget_findings"] = len(widget_findings)
                paper_row["n_span_findings"] = len(span_findings)
                paper_row["n_findings_total"] = len(all_findings)

                any_prompty = False
                any_hidden_or_transparent = False
                any_layer_text = False

                for f in all_findings:
                    # Fill any missing schema fields to keep CSV stable
                    for col in FINDINGS_FIELDS:
                        f.setdefault(col, "")

                    # Update summary booleans
                    if bool(f.get("prompty")):
                        any_prompty = True

                    reasons = str(f.get("reasons") or "")
                    if ("hidden_render_mode" in reasons) or ("opacity" in reasons):
                        any_hidden_or_transparent = True
                    if ("optional_content_layer" in reasons):
                        any_layer_text = True

                    findings_writer.writerow(f)

                findings_fh.flush()

                paper_row["any_prompty"] = any_prompty
                paper_row["any_hidden_or_transparent"] = any_hidden_or_transparent
                paper_row["any_layer_text"] = any_layer_text

                paper_row["status"] = "ok"
                doc.close()

            except Exception as e:
                paper_row["status"] = "error"
                paper_row["error"] = f"{type(e).__name__}: {e}"
                errored_n += 1
                try:
                    # Ensure doc closed if partially opened
                    if "doc" in locals():
                        doc.close()
                except Exception:
                    pass

            # Always write paper summary row
            paper_writer.writerow(paper_row)
            paper_fh.flush()

            processed_paths.add(pdf_path)
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
        errored_n=errored_n,
    )


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing PDFs")
    ap.add_argument("--out", default="findings.csv", help="Findings CSV path")
    ap.add_argument("--no_recursive", action="store_true", help="Do not scan subfolders")
    ap.add_argument("--no_resume", action="store_true", help="Do not skip already processed PDFs")
    ap.add_argument("--max_files", type=int, default=0, help="Cap number of PDFs (0 = no cap)")
    ap.add_argument("--shard_count", type=int, default=0, help="Number of shards (0 = no sharding)")
    ap.add_argument("--shard_index", type=int, default=0, help="Shard index (0..shard_count-1)")

    args = ap.parse_args(argv)

    res = run(
        input_dir=args.input_dir,
        out=args.out,
        recursive=not args.no_recursive,
        resume=not args.no_resume,
        max_files=args.max_files,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )

    print(f"Processed (this run): {res['processed_n']} | Skipped (resume): {res['skipped_n']} | Errored: {res['errored_n']}")
    print(f"Outputs: {res['out_csv']} and {res['summary_csv']}")


if __name__ == "__main__":
    main()
