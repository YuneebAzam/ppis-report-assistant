"""
Ingest the PPIS Annual Report PDF into a searchable index.

Run once (or whenever the PDF changes):
    python ingest.py

What it does:
  1. Extracts every page's text with pdfplumber, at the word level, so
     table columns can be reconstructed precisely (x_tolerance=1 keeps
     tightly-packed columns apart).
  2. Decodes the report's broken font encoding ((cid:NN) -> character).
  3. Groups consecutive pages that repeat the same header text -- i.e. a
     single table continuing across pages -- into "sections", and detects
     each section's column boundaries from ALL of its rows at once (a
     single page's rows are often not enough: e.g. the "Operator" column
     may only be printed on a table's first row, blank everywhere else).
  4. For each table section, extracts every data row as a structured
     record: {table title, page, entity (company/operator, filled down
     from the header column when the report leaves it blank), is_total
     (row is a printed "Total"/"Sub-Total" line), cells (column label ->
     value)}. This is the "master table": index/rows.json.
  5. Non-table pages (narrative text) are chunked and embedded the old
     way, for free-text questions -- index/chunks.json + embeddings.npy.

Why structured rows instead of just embedding table text: asking a small
local LLM to (a) figure out which row/column a number belongs to from
loosely-formatted text, or (b) add up several rows itself, is unreliable.
Here Python does the lookup and any arithmetic; the model only phrases
the answer from already-clean, already-computed numbers.

Requires the Ollama service to be running (it starts automatically on
Windows; otherwise run `ollama serve`).
"""
import bisect
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pdfplumber
import requests

PDF_PATH = Path(__file__).parent / "data" / "PPIS_Annual_2014-15.pdf"
INDEX_DIR = Path(__file__).parent / "index"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

LINES_PER_CHUNK = 16
OVERLAP_LINES = 3
HEADER_LINES = 6  # page heading lines repeated on top of every chunk

CID_RE = re.compile(r"\(cid:(\d+)\)")
# Two money figures (3 decimals each) printed touching, e.g. 32592.26733990.144
GLUED_RE = re.compile(r"(\d\.\d{3})(?=\d{1,6}\.\d{3}\b)")


def decode_cid(s):
    """The report's fonts have no Unicode map, so pdfplumber emits
    (cid:NN) tokens -- but NN is the character code itself."""
    if not s:
        return ""
    return CID_RE.sub(lambda m: chr(int(m.group(1))), s)


def get_word_lines(page):
    """Words grouped into visual lines, with the cid encoding decoded."""
    words = page.extract_words(x_tolerance=1)
    for w in words:
        w["text"] = re.sub(r"\s+", " ", decode_cid(w["text"])).strip()
    words = [w for w in words if w["text"]]
    words.sort(key=lambda w: (w["top"], w["x0"]))
    lines, cur, cur_top = [], [], None
    for w in words:
        if cur_top is None or abs(w["top"] - cur_top) <= 3:
            cur.append(w)
            cur_top = w["top"] if cur_top is None else cur_top
        else:
            lines.append(cur)
            cur, cur_top = [w], w["top"]
    if cur:
        lines.append(cur)
    for ln in lines:
        ln.sort(key=lambda w: w["x0"])
    return lines


def detect_boundaries(lines):
    """Find x-positions of the whitespace channels that run vertically
    through the page's table, by looking at where words are NOT."""
    rows = [ln for ln in lines if len(ln) >= 3]
    if len(rows) < 4:
        return None
    x0 = min(w["x0"] for ln in rows for w in ln)
    x1 = max(w["x1"] for ln in rows for w in ln)
    width = int(x1 - x0) + 2
    cov = np.zeros(width)
    for ln in rows:
        for w in ln:
            cov[max(0, int(w["x0"] - x0)):int(w["x1"] - x0) + 1] += 1
    thresh = max(1.0, 0.15 * len(rows))
    free = cov <= thresh
    bounds, i = [], 0
    while i < width:
        if free[i]:
            j = i
            while j < width and free[j]:
                j += 1
            if j - i >= 3 and i > 0 and j < width:  # interior channel only
                bounds.append(x0 + (i + j) / 2)
            i = j
        else:
            i += 1
    return bounds if len(bounds) >= 2 else None


SUBTOTAL_RE = re.compile(r"^Sub-?\s?Totals?:?$", re.IGNORECASE)
TOTALROW_RE = re.compile(r"^(Sub-?\s?Totals?:?( for .*)?|Grand )?Totals?"
                         r"( area)?:?$", re.IGNORECASE)
NUMCELL_RE = re.compile(r"^-?[\d.,()/%\s]+$")
ENTITY_COL_RE = re.compile(r"company|operator|operating", re.IGNORECASE)
FRAG_STOP = {"operating", "company", "financial", "activities",
             "concession", "total", "sub-total"}
# header-row words that must never be mistaken for an entity name
FRAG_SKIP_RE = re.compile(r"^(s\s*\.?\s*no\.?|no\.?|block|lease|name)\b",
                          re.IGNORECASE)


def header_signature(lines, n=2):
    """Normalized text of a page's first few lines, used to detect when
    consecutive pages are continuing the same table (pdfplumber has no
    notion of this; the report just repeats the table header verbatim)."""
    def plain(ln):
        return re.sub(r"\s+", " ", " ".join(w["text"] for w in ln)).strip().lower()
    sig = tuple(plain(ln) for ln in lines[:n] if ln)
    return sig if any(sig) else None


def group_into_sections(pages_words):
    """Group consecutive pages that share the same header text -- i.e. a
    single table continuing across pages -- so column boundaries can be
    shared across all of a table's pages instead of derived per page."""
    sections, prev_sig = [], object()
    for idx, lines in enumerate(pages_words):
        sig = header_signature(lines)
        if sig is not None and sig == prev_sig:
            sections[-1].append(idx)
        else:
            sections.append([idx])
        prev_sig = sig
    return sections


def section_boundaries(section, pages_words):
    """Union of every page's own column boundaries, clustered.

    Pooling all pages' words into one detection pass fails when pages are
    offset by a few points (channels get smeared shut); per-page detection
    fails when a column is empty on a continuation page (its channel is
    invisible there). The union keeps every channel any page can see,
    while clustering merges the same channel seen at slightly different
    x-positions on different pages."""
    all_bounds = []
    for idx in section:
        b = detect_boundaries(pages_words[idx])
        if b:
            all_bounds.extend(b)
    if not all_bounds:
        return None
    all_bounds.sort()
    clusters, cur = [], [all_bounds[0]]
    for x in all_bounds[1:]:
        if x - cur[-1] <= 4:
            cur.append(x)
        else:
            clusters.append(sum(cur) / len(cur))
            cur = [x]
    clusters.append(sum(cur) / len(cur))
    return clusters if len(clusters) >= 2 else None


def section_title(first_page_lines):
    """The section's display title: the most informative of the page's
    first few lines (chapter names like 'Concession Activities' repeat on
    every page and say nothing; the actual table name is usually the
    longest line, e.g. 'Exploration Licences held by companies ...')."""
    def plain(ln):
        return re.sub(r"\s+", " ", " ".join(w["text"] for w in ln)).strip()
    header = [plain(ln) for ln in first_page_lines[:4]]
    header = [t for t in header if t]
    if not header:
        return "Untitled"
    candidates = [t for t in header if not re.search(r"\bTable\s+\d", t)]
    title = max(candidates, key=len) if candidates else header[0]
    for t in header:
        m = re.search(r"\bTable\s+\d+(\.\d+)?", t)
        if m:
            title = f"{title} ({m.group(0)})"
            break
    return title


def find_entity_col(col_labels):
    for i, lbl in enumerate(col_labels):
        if lbl and ENTITY_COL_RE.search(lbl):
            return i
    return 0


def render_page(lines, bounds, state, pageno, title, rows):
    """Render one page's visual lines into '| cell | cell |' text rows
    AND append structured row records to `rows`, using state shared
    across every page of the same table section (see group_into_sections).

    Entity (company/operator) tracking: whichever column the header calls
    Company/Operator is read directly each row; when the report leaves it
    blank (common on continuation rows/pages), the last non-blank value in
    that column is carried forward -- this both labels rows individually
    and lets anonymous "Sub-Total" rows be attributed to the right entity.
    """
    out = []
    for ln in lines:
        if bounds:
            cells = [[] for _ in range(len(bounds) + 1)]
            for w in ln:
                mid = (w["x0"] + w["x1"]) / 2
                cells[bisect.bisect_left(bounds, mid)].append(w["text"])
            cells = [GLUED_RE.sub(r"\1 ", " ".join(c).strip())
                     for c in cells]

            if any(re.search(r"\bTable\s+\d", c) for c in cells):
                state["header_cells"], state["col_labels"] = [], None
                state["current_entity"], state["entity_col"] = "", 0

            # Column labels: join the last few pre-data header lines
            # vertically, so each column gets its printed heading.
            n_num = sum(1 for c in cells if c and NUMCELL_RE.match(c))
            if state["col_labels"] is None:
                if n_num >= 2 and sum(1 for c in cells if c) >= 3:
                    col_labels = []
                    for i in range(len(cells)):
                        parts = [h[i] for h in state["header_cells"][-3:]
                                 if i < len(h) and h[i]]
                        col_labels.append(" ".join(parts).strip())
                    state["col_labels"] = col_labels
                    state["entity_col"] = find_entity_col(col_labels)
                else:
                    state["header_cells"].append(cells)
            col_labels = state["col_labels"]
            is_subtotal = any(SUBTOTAL_RE.match(c) for c in cells)

            entity = ""
            if col_labels is not None and n_num >= 1:
                ecol = state.get("entity_col", 0)
                ecell = cells[ecol] if ecol < len(cells) else ""
                if not is_subtotal and ecell \
                        and ecell.lower() not in FRAG_STOP \
                        and not FRAG_SKIP_RE.match(ecell) \
                        and not re.match(r"^[\d.,()-]+$", ecell):
                    state["current_entity"] = ecell
                entity = state.get("current_entity", "")
                if is_subtotal and entity:
                    for i, c in enumerate(cells):
                        if SUBTOTAL_RE.match(c):
                            cells[i] = f"Sub-Total for {entity}"
                    state["current_entity"] = ""

                row_cells = {}
                for i in range(min(len(cells), len(col_labels))):
                    if not (col_labels[i] and cells[i]):
                        continue
                    key = col_labels[i]
                    n = 2
                    while key in row_cells:  # duplicate label: keep both
                        key = f"{col_labels[i]} ({n})"
                        n += 1
                    row_cells[key] = cells[i]
                if row_cells:
                    is_total = is_subtotal or any(
                        TOTALROW_RE.match(c) for c in cells)
                    rows.append({
                        "page": pageno,
                        "table": title,
                        "entity": entity,
                        "is_total": is_total,
                        "cells": row_cells,
                    })
            row = "| " + " | ".join(cells) + " |"
        else:
            row = " ".join(w["text"] for w in ln)
        row = GLUED_RE.sub(r"\1 ", row).strip()
        if row and row.replace("|", "").strip():
            out.append(row)
    return out


def chunk_narrative_pages(section, page_text_lines, chunks):
    """Old-style line chunking, used only for pages that never turned out
    to contain a table (pure narrative / methodology text)."""
    for idx in section:
        pageno = idx + 1
        lines = page_text_lines.get(idx, [])
        if not lines:
            continue
        header = lines[:HEADER_LINES]
        body = lines[HEADER_LINES:]

        def plain(s):
            return re.sub(r"\s+", " ", s.replace("|", " ")).strip()

        title = plain(header[0]) if header else f"Page {pageno}"
        if not body:
            body, header = header, []

        step = LINES_PER_CHUNK - OVERLAP_LINES
        for start in range(0, len(body), step):
            part = body[start:start + LINES_PER_CHUNK]
            if not part:
                break
            text = "\n".join([f"[Page {pageno}] {title}"] + header + part)
            chunks.append({"page": pageno, "title": title, "text": text})
            if start + LINES_PER_CHUNK >= len(body):
                break


def extract_all(pdf_path=PDF_PATH, progress=None):
    """progress: optional callback(stage_label, done, total)."""
    with pdfplumber.open(pdf_path) as pdf:
        n = len(pdf.pages)
        pages_words = []
        for i, page in enumerate(pdf.pages):
            pages_words.append(get_word_lines(page))
            if progress:
                progress("Reading pages", i + 1, n)

    sections = group_into_sections(pages_words)
    rows, chunks = [], []
    for section in sections:
        title = section_title(pages_words[section[0]])
        bounds = section_boundaries(section, pages_words)
        state = {"col_labels": None, "header_cells": [],
                 "entity_col": 0, "current_entity": ""}
        page_text_lines = {}
        n_rows_before = len(rows)
        for idx in section:
            pageno = idx + 1
            page_text_lines[idx] = render_page(
                pages_words[idx], bounds, state, pageno, title, rows)
        if len(rows) == n_rows_before:  # no table detected: narrative text
            chunk_narrative_pages(section, page_text_lines, chunks)
    return rows, chunks


def embed_all(texts, progress=None):
    vecs = []
    B = 16
    for i in range(0, len(texts), B):
        batch = texts[i:i + B]
        r = requests.post(f"{OLLAMA_URL}/api/embed",
                          json={"model": EMBED_MODEL, "input": batch},
                          timeout=300)
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
        if progress:
            progress("Embedding text", min(i + B, len(texts)), len(texts))
    arr = np.array(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


def ollama_available():
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return True
    except requests.ConnectionError:
        return False


def build_index(pdf_path=PDF_PATH, progress=None):
    """Full ingestion for one PDF: extract -> (embed narrative) -> write
    index files. Returns the meta dict. Used by both the CLI (main) and
    the app's upload button."""
    pdf_path = Path(pdf_path)
    rows, chunks = extract_all(pdf_path, progress)

    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # remove any stale narrative index from a previously ingested PDF
    for name in ("chunks.json", "embeddings.npy"):
        stale = INDEX_DIR / name
        if stale.exists():
            stale.unlink()
    if chunks:
        emb = embed_all([c["text"] for c in chunks], progress)
        (INDEX_DIR / "chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        np.save(INDEX_DIR / "embeddings.npy", emb)

    meta = {"pdf": pdf_path.name, "rows": len(rows), "chunks": len(chunks),
            "ingested": time.strftime("%Y-%m-%d %H:%M")}
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def main():
    if not PDF_PATH.exists():
        sys.exit(f"PDF not found: {PDF_PATH}")
    if not ollama_available():
        sys.exit("Ollama is not running. Start it first (ollama serve).")

    last = {}

    def progress(stage, done, total):
        if stage != last.get("stage") or done == total or done % 10 == 0:
            print(f"  {stage}: {done}/{total}")
            last["stage"] = stage

    meta = build_index(PDF_PATH, progress)
    print(f"Done. {meta['rows']} table rows, {meta['chunks']} narrative "
          f"chunks. Index written to {INDEX_DIR}")


if __name__ == "__main__":
    main()
