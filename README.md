# Identify prompt injections in academic preprints from folder of PDFs

`scan_pdfs_in_folder_expansion.py` is standalone. 

Scan local PDFs for potentially invisible / adversarial prompt-injection content that
could affect LLM-assisted peer review.

## Checks:
1) Text spans via PyMuPDF get_texttrace():
   - hidden render mode (type==3)
   - low/zero opacity
   - tiny/micro font
   - near-white color
   - optional content layer text (OCG/layer)
   - microtext density
2) Manipulation text detection (expanded):
   - normalization (NFKC + casefold + remove zero-width + remove bidi controls)
   - regex "concept buckets" scoring (instruction + role/channel + review manipulation)
   - non-contiguous detection: concatenated page text built from spans (sorted by y/x)
   - reversed / ROT13 checks
   - base64/hex decoding attempts on long tokens, then re-scan decoded text
   - fuzzy matching against a small library of prompt-injection templates (rapidfuzz if available)
3) Additional channels:
   - annotations
   - form fields/widgets (AcroForm)
   - metadata (Info dict + XMP)
   - embedded files (presence + name/desc)

## Persistence & robustness:
- Streaming CSV writes (append + flush) => progress is not lost
- Resume: skips PDFs already present in *_paper_summary.csv (by pdf_path)
- Sharding: --shard_count K --shard_index i to split workload (do NOT run shards writing
  to the same output file concurrently; use separate --out per shard)

## Dependencies:
  `pip install PyMuPDF tqdm pandas`
Optional:
  `pip install rapidfuzz`  (faster/better fuzzy matching)

## Usage:
  `python scan_pdfs_in_folder_expansion.py --input_dir /path/to/pdfs --out findings.csv`
  
  `python scan_pdfs_in_folder_expansion.py --input_dir /path --out findings.csv --shard_count 4 --shard_index 0`
