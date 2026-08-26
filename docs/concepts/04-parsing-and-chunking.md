# Parsing & chunking: turning files into retrievable pieces

## Parsing (Docling)
LLMs and embedding models eat text, but knowledge lives in PDFs, Word
files, slides, and scans. **Docling** converts all of them into
structured markdown, *layout-aware*: it detects headings, reading
order, and tables (a table becomes a markdown table, not shredded
words). Images and scanned PDFs go through OCR. The parser also inserts
a page-break marker between pages so citations can point at a page.

## Chunking
Documents are too long to embed whole, and retrieval works best on
focused passages. Naive chunking ("every 500 characters") slices
sentences and tables mid-way. Ours is **structure-aware**
([chunking.py](../../rag_core/chunking.py)):

- a heading starts a new chunk, and the heading text is prepended to
  every chunk under it — so the embedded text carries its own context
  ("Refund policy: … within 30 days" instead of a floating "within 30
  days")
- tables are never split
- chunks cap at ~1200 characters, splitting only at paragraph borders
- each chunk records its **source file, page, and heading** — the
  provenance that citations link back to

## Deterministic IDs
A chunk's ID is a uuid5 hash of (filename, position). Re-uploading a
file produces the same IDs, so the store *overwrites* instead of
duplicating — and we delete the file's old chunks first, so re-ingesting
is a clean replace.
