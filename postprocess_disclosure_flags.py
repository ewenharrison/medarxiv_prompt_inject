#!/usr/bin/env python3
"""
postprocess_disclosure_flags.py

Takes per_version.csv from medrxiv_genai_disclosure_versiondiff.py and produces
refined disclosure flags that are *more specific* than "any mention".

Why:
- "Disclosure-any" can be inflated by tool mentions in methods (e.g., OpenAI API, Gemini model used for analysis),
  or by casual mentions not framed as writing assistance.
- These refined flags are designed to be defensible: they require BOTH (a) a tool mention and (b) a writing/editing
  framing cue, and optionally (c) proximity to likely disclosure sections.

Input:
  per_version.csv (must include columns: pdf_path, doi, version, status, disclosure_terms, disclosure_context, feat_headings)

Output:
  per_version_refined.csv (adds columns)
  disclosure_term_counts.csv (counts of terms by category)

Usage:
  python postprocess_disclosure_flags.py \
    --in_csv /home/eharrison/medarxiv_hidden_text/out/per_version.csv \
    --out_csv /home/eharrison/medarxiv_hidden_text/out/per_version_refined.csv
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Set, Tuple

import pandas as pd


# ----------------------------- Term groupings --------------------------------

# "Tool mention" groups
LLM_TERMS = {"chatgpt", "gpt4", "gpt", "openai", "llm", "genai", "claude", "gemini", "copilot"}
NON_LLM_WRITING_TOOLS = {"grammarly", "deepl", "quillbot", "writefull"}

# High precision cues that suggest writing/editing assistance disclosure
WRITING_CUE_RE = re.compile(
    r"(?is)\b("
    r"ai[-\s]*assisted\s+(writing|drafting|editing)|"
    r"assisted\s+with\s+(writing|drafting|editing)|"
    r"used\s+(chatgpt|gpt|a\s+large\s+language\s+model|an?\s+llm)\s+to\s+"
    r"(draft|edit|rewrite|proofread|revise|improve\s+the\s+language)|"
    r"(draft|edit|rewrite|proofread|revise|improve)\s+the\s+(manuscript|text|writing)\s+"
    r"(using|with)\s+(chatgpt|gpt|openai|a\s+large\s+language\s+model|an?\s+llm)|"
    r"language\s+editing\s+(was\s+)?(performed|assisted)\s+by|"
    r"proofread(ing)?\s+(was\s+)?(performed|assisted)\s+by"
    r")\b"
)

# Weaker cues: can support "probable writing assistance" when combined with tool mention
WEAK_WRITING_CUE_RE = re.compile(
    r"(?is)\b("
    r"assisted|help(ed)?|support(ed)?|"
    r"edited|editing|proofread|proofreading|"
    r"rewrite|rewriting|revise|revised|"
    r"language\s+editing|writing\s+assistance|"
    r"grammar|spelling|style"
    r")\b"
)

# Detect likely disclosure sections based on headings extracted in your scan
DISCLOSURE_SECTION_HEADINGS = {
    "acknowledgements", "acknowledgments",
    "author contributions", "contributions",
    "conflicts of interest", "competing interests",
    "funding", "data availability", "ethics",
}

SECTION_HINT_RE = re.compile(
    r"(?i)\b(acknowledg(e)?ments?|author\s+contributions?|competing\s+interests?|conflicts?\s+of\s+interest|funding)\b"
)


# ----------------------------- Helper functions ------------------------------

def parse_terms(term_str: str) -> Set[str]:
    if not isinstance(term_str, str) or not term_str.strip():
        return set()
    return {t.strip() for t in term_str.split(";") if t.strip()}


def has_disclosure_section(headings_str: str) -> bool:
    if not isinstance(headings_str, str) or not headings_str.strip():
        return False
    hs = {h.strip().casefold() for h in headings_str.split(";") if h.strip()}
    if hs & DISCLOSURE_SECTION_HEADINGS:
        return True
    # fallback: hint in headings text
    return bool(SECTION_HINT_RE.search(headings_str))


def term_group(terms: Set[str]) -> Tuple[bool, bool]:
    has_llm = bool(terms & LLM_TERMS)
    has_nonllm = bool(terms & NON_LLM_WRITING_TOOLS)
    return has_llm, has_nonllm


def refined_flags(terms: Set[str], context: str, headings: str) -> Dict[str, object]:
    ctx = context if isinstance(context, str) else ""
    ctx_lc = ctx.casefold()

    has_llm, has_nonllm = term_group(terms)
    in_disclosure_section = has_disclosure_section(headings)

    # Core refinements
    any_mention = bool(terms)

    # High precision writing-assistance disclosure:
    # requires tool mention + strong cue in context
    llm_writing_disclosure_hp = bool(has_llm and WRITING_CUE_RE.search(ctx))
    nonllm_writing_disclosure_hp = bool(has_nonllm and WRITING_CUE_RE.search(ctx))

    # “Probable” writing assistance disclosure:
    # tool mention + weak cue + (either section context or strong cue)
    llm_writing_disclosure_prob = bool(
        has_llm and (WEAK_WRITING_CUE_RE.search(ctx) is not None) and (in_disclosure_section or WRITING_CUE_RE.search(ctx))
    )
    nonllm_writing_disclosure_prob = bool(
        has_nonllm and (WEAK_WRITING_CUE_RE.search(ctx) is not None) and (in_disclosure_section or WRITING_CUE_RE.search(ctx))
    )

    # “LLM mention likely not writing assistance” (e.g., used for analysis)
    # heuristic: tool mention present but NO writing cues; helps you sanity-check false inflation.
    llm_mention_no_writing_cues = bool(has_llm and not WEAK_WRITING_CUE_RE.search(ctx))

    return {
        "disclosure_any_mention": any_mention,
        "has_llm_term": has_llm,
        "has_nonllm_writing_tool_term": has_nonllm,
        "in_disclosure_section_hint": in_disclosure_section,

        "llm_writing_disclosure_hp": llm_writing_disclosure_hp,
        "nonllm_writing_disclosure_hp": nonllm_writing_disclosure_hp,
        "writing_disclosure_hp_any": bool(llm_writing_disclosure_hp or nonllm_writing_disclosure_hp),

        "llm_writing_disclosure_prob": llm_writing_disclosure_prob,
        "nonllm_writing_disclosure_prob": nonllm_writing_disclosure_prob,
        "writing_disclosure_prob_any": bool(llm_writing_disclosure_prob or nonllm_writing_disclosure_prob),

        "llm_mention_no_writing_cues": llm_mention_no_writing_cues,
    }


def term_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        terms = parse_terms(r.get("disclosure_terms", ""))
        for t in terms:
            rows.append({"term": t})
    if not rows:
        return pd.DataFrame(columns=["term", "n"])
    out = pd.DataFrame(rows).value_counts("term").reset_index(name="n").sort_values("n", ascending=False)
    return out


# ----------------------------- Main ------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="per_version.csv")
    ap.add_argument("--out_csv", required=True, help="per_version_refined.csv")
    ap.add_argument("--term_counts_csv", default="", help="optional disclosure_term_counts.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # Only refine for successfully parsed PDFs
    df_ok = df[df["status"].astype(str) == "ok"].copy()

    # Ensure required columns exist
    for col in ("disclosure_terms", "disclosure_context", "feat_headings"):
        if col not in df_ok.columns:
            df_ok[col] = ""

    refined = []
    for _, r in df_ok.iterrows():
        terms = parse_terms(r.get("disclosure_terms", ""))
        ctx = r.get("disclosure_context", "")
        headings = r.get("feat_headings", "")
        refined.append(refined_flags(terms, ctx, headings))

    df_ref = pd.concat([df_ok.reset_index(drop=True), pd.DataFrame(refined)], axis=1)
    df_ref.to_csv(args.out_csv, index=False)

    if args.term_counts_csv:
        tc = term_counts(df_ok)
        tc.to_csv(args.term_counts_csv, index=False)

    print(f"Wrote: {args.out_csv}")
    if args.term_counts_csv:
        print(f"Wrote: {args.term_counts_csv}")


if __name__ == "__main__":
    main()
