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
import shutil
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


def group_words_into_lines(words):
    """Shared by both extraction paths (pdfplumber's real text layer and
    the OCR fallback below): cluster words into visual lines by vertical
    position, each word a dict with text/x0/x1/top in PDF point units."""
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


def get_word_lines(page):
    """Words grouped into visual lines, with the cid encoding decoded."""
    words = page.extract_words(x_tolerance=1)
    for w in words:
        w["text"] = re.sub(r"\s+", " ", decode_cid(w["text"])).strip()
    return group_words_into_lines(words)


def text_is_usable(decoded_text):
    """Distinguishes this report's known (cid:NN) quirk (which decode_cid
    fixes, producing normal readable characters) from a genuinely broken
    or missing font encoding in the source PDF, where pdfminer can't
    resolve glyphs to Unicode at all and emits raw control characters
    when decode_cid naively maps the cid number to chr(). No text-layer
    fix exists for that case -- see the OCR fallback below."""
    if not decoded_text.strip():
        return False
    printable = sum(1 for c in decoded_text if c.isprintable() or c in "\n\t")
    return printable / len(decoded_text) > 0.85


def pdf_needs_ocr(pdf_path):
    """Sample a few pages spread across the document; if none of them
    decode to usable text, the whole PDF's font encoding is broken and
    every page will need OCR (checked once, not per-page, since a
    document's font setup doesn't usually vary page to page)."""
    with pdfplumber.open(pdf_path) as pdf:
        n = len(pdf.pages)
        idxs = sorted({0, n // 2, n - 1, min(n - 1, 5)})
        for i in idxs:
            raw = pdf.pages[i].extract_text() or ""
            if text_is_usable(decode_cid(raw)):
                return False
    return True


def locate_tesseract():
    """pytesseract needs to be told where the tesseract binary is on
    Windows (it isn't reliably on PATH after installation); on Linux
    (e.g. a Codespace with `apt-get install tesseract-ocr`) it's on PATH
    already, so this returns None and pytesseract finds it itself."""
    if shutil.which("tesseract"):
        return None
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if Path(p).exists():
            return p
    return None


OCR_DPI = 400  # 200-300 loses real detail on this report's dense, small-
               # font tables (e.g. an operator column reading blank or a
               # single garbled letter instead of "PPL") -- 400 recovers
               # it at negligible extra time cost (~3% slower per page).


def get_word_lines_ocr(fitz_page, dpi=OCR_DPI):
    """OCR-based equivalent of get_word_lines(), used when the PDF's own
    text layer is unusable (see pdf_needs_ocr). Renders the page to an
    image and reads it visually with Tesseract, which sidesteps the
    broken font encoding entirely since it never touches embedded text.
    Word pixel coordinates are converted back to PDF point units
    (dividing by dpi/72) so they're compatible with the same
    column-boundary and line-clustering thresholds already tuned for
    get_word_lines()'s output."""
    import fitz
    import pytesseract
    from PIL import Image

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = fitz_page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    tcmd = locate_tesseract()
    if tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words = []
    for i in range(len(data["text"])):
        text = re.sub(r"\s+", " ", data["text"][i]).strip()
        if not text or int(data["conf"][i]) < 0:
            continue
        words.append({
            "text": text,
            "x0": data["left"][i] / zoom,
            "x1": (data["left"][i] + data["width"][i]) / zoom,
            "top": data["top"][i] / zoom,
        })
    return group_words_into_lines(words)


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
SUBTOTAL_NAMED_RE = re.compile(r"^Sub[\s-]?Totals?\s*\(([^)]+)\)$", re.IGNORECASE)
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


TITLE_CONNECTOR_RE = re.compile(r"\b(as|of|on|during|held|by|for|in)\b",
                                re.IGNORECASE)


def section_title(first_page_lines):
    """The section's display title: the most informative of the page's
    first few lines (chapter names like 'Concession Activities' repeat on
    every page and say nothing; the actual table name is usually the
    longest line, e.g. 'Exploration Licences held by companies ...').

    A column-header row (e.g. 'Company Field Discovery Year Original
    Comulative Balance') can be longer, in raw character count, than the
    true title line ('Oil Reserves as on 30th June 2015 ...') -- but a
    real title reads as a phrase with connecting words, while a header
    row is just a list of capitalized labels with none. Prefer lines
    with a connector when picking the longest candidate, since losing a
    disambiguating word here (e.g. "Oil" vs "Gas") makes two differently
    named tables in the report indistinguishable by title alone."""
    def plain(ln):
        return re.sub(r"\s+", " ", " ".join(w["text"] for w in ln)).strip()
    header = [plain(ln) for ln in first_page_lines[:4]]
    header = [t for t in header if t]
    if not header:
        return "Untitled"
    candidates = [t for t in header if not re.search(r"\bTable\s+\d", t)]
    with_connector = [t for t in candidates if TITLE_CONNECTOR_RE.search(t)]
    pool = with_connector or candidates
    title = max(pool, key=len) if pool else header[0]
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


ZONE_WORD_RE = re.compile(r"\b(Exploration|Appraisal|Development)\b",
                          re.IGNORECASE)


NUM_TOKEN_RE = re.compile(r"-?[\d,]+\.\d+")


ZONE_COLUMN_LABELS = [
    "G & G Studies (Exploration)", "Exploratory Drilling (Exploration)",
    "Other Expenditure (Exploration)", "Exploration Total (In MM Rs.)",
    "G & G Studies (Appraisal)", "Appraisal Drilling (Appraisal)",
    "Other Expenditure (Appraisal)", "Appraisal Total (In MM Rs.)",
    "Development Drilling", "Production Facilities",
    "Other Expenditure (Development)", "Development Total (In MM Rs.)",
]


def relabel_zone_totals(row_cells, table_title):
    """Table 5.2 always has exactly 12 numbers per data row: 4 per section
    (G&G/drilling/Other/Total), always in Exploration -> Appraisal ->
    Development order -- a fixed structural property of this table, used
    instead of matching column-label text because columns here are packed
    only a few points apart (breaking text alignment) and the merged
    group-header words don't survive per-column splitting cleanly.

    Rather than relabelling only the 3 section-total columns and leaving
    the rest of the row's fragile, sometimes-nonsensical auto-joined
    labels in place, replace all 12 numeric columns with clean labels
    (keeping only the genuinely non-numeric cells -- company/block name
    -- as they were). A row full of garbled intermediate labels gives a
    small model more surface area to misread than a clean one."""
    # Gate on "Expenditure" alone, not the full "Exploration"+"Development"
    # phrase -- section_title() can legitimately pick different header
    # lines for the same table (e.g. "Exploration and Production
    # Expenditure During 2014-2015" instead of the zone-line "Exploration
    # Expenditure Appraisal Expenditure Development Expenditure"), and
    # this must not depend on exact title wording. The 12-number check
    # below is what actually makes this specific to this table's shape.
    if "Expenditure" not in table_title:
        return row_cells
    nums = [n for v in row_cells.values() for n in NUM_TOKEN_RE.findall(v)]
    if len(nums) != 12:
        return row_cells  # not a fully-populated row of this table's shape
    out = {k: v for k, v in row_cells.items() if not NUM_TOKEN_RE.search(v)}
    out.update(zip(ZONE_COLUMN_LABELS, nums))
    return out


def fill_zone_header(header_line):
    """Some report tables print a merged-cell group header spanning
    several sub-columns (e.g. "Exploration Expenditure | Appraisal
    Expenditure | Development Expenditure", each covering 4 numeric
    columns beneath it). In the PDF's raw text each zone word only has
    ONE x-position, so it lands in exactly one column bin -- the other
    columns in that zone would otherwise get no label at all, making
    e.g. every "Total (In MM Rs.)" column indistinguishable. If this
    line has 2+ zone words, forward-fill each one across the empty
    columns to its right until the next zone word appears."""
    hits = sum(1 for c in header_line if c and ZONE_WORD_RE.search(c))
    if hits < 2:
        return header_line
    filled, current = list(header_line), ""
    for i, c in enumerate(filled):
        if c:
            m = ZONE_WORD_RE.search(c)
            if m:
                current = m.group(1).title()
            # non-empty cell with no zone word: it's a real column label
            # (e.g. "Licence/", "Develop.") -- never overwrite it.
        elif current:
            filled[i] = current  # empty cell: forward-fill the zone name
    return filled


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
                    header_lines = [fill_zone_header(h)
                                    for h in state["header_cells"][-3:]]
                    col_labels = []
                    for i in range(len(cells)):
                        parts = [h[i] for h in header_lines
                                 if i < len(h) and h[i]]
                        col_labels.append(" ".join(parts).strip())
                    state["col_labels"] = col_labels
                    state["entity_col"] = find_entity_col(col_labels)
                else:
                    state["header_cells"].append(cells)
            col_labels = state["col_labels"]
            is_subtotal = any(SUBTOTAL_RE.match(c) for c in cells)
            # Some tables print the company name directly inside the
            # subtotal cell, e.g. "Sub Total (UEPL)" as one unit, instead
            # of a bare "Sub-Total" relying on the tracked entity -- this
            # is a more reliable signal than fill-down when present, and
            # doesn't depend on the company name having been seen earlier.
            named_subtotal = next(
                (m for c in cells if (m := SUBTOTAL_NAMED_RE.match(c))), None)

            entity = ""
            if col_labels is not None and n_num >= 1:
                ecol = state.get("entity_col", 0)
                # Column boundaries are shared across the whole section,
                # but different company codes' text sits at slightly
                # different x-positions (sub-point rendering variance),
                # so the same boundary can bisect one company's word but
                # not another's, landing it one bin to the left of where
                # the header text put entity_col. Only fall back to that
                # bin when the primary one is empty -- never glue the two
                # together, since the left bin can hold unrelated content
                # (a stray number, another column's word) when the
                # primary bin already has the real company name.
                ecell = cells[ecol] if ecol < len(cells) else ""
                if not ecell and ecol > 0:
                    ecell = cells[ecol - 1]
                if not is_subtotal and ecell \
                        and ecell.lower() not in FRAG_STOP \
                        and not FRAG_SKIP_RE.match(ecell) \
                        and not re.match(r"^[\d.,()-]+$", ecell):
                    state["current_entity"] = ecell
                entity = state.get("current_entity", "")
                if named_subtotal:
                    entity = named_subtotal.group(1).strip()
                    is_subtotal = True
                    state["current_entity"] = ""
                    for i, c in enumerate(cells):
                        if SUBTOTAL_NAMED_RE.match(c):
                            cells[i] = f"Sub-Total for {entity}"
                elif is_subtotal and entity:
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
                row_cells = relabel_zone_totals(row_cells, title)
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


def extract_all(pdf_path=PDF_PATH, progress=None, ocr_flag=None):
    """progress: optional callback(stage_label, done, total).
    ocr_flag: optional dict; ["used"] is set True if OCR was needed, so
    the caller can record it in the index metadata (retrieval treats
    OCR-sourced tables as less trustworthy -- see app.py)."""
    use_ocr = pdf_needs_ocr(pdf_path)
    if ocr_flag is not None:
        ocr_flag["used"] = use_ocr
    pages_words = []
    if use_ocr:
        # This PDF's embedded text has no usable font encoding at all (a
        # defect in the source file -- confirmed by testing two separate
        # PDF libraries, both failing identically). Read every page
        # visually with OCR instead; much slower than the text layer, so
        # progress reporting says so.
        import fitz
        doc = fitz.open(pdf_path)
        n = len(doc)
        for i in range(n):
            pages_words.append(get_word_lines_ocr(doc[i]))
            if progress:
                progress("Reading pages (OCR -- slower, no text layer "
                         "in this PDF)", i + 1, n)
        doc.close()
    else:
        with pdfplumber.open(pdf_path) as pdf:
            n = len(pdf.pages)
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
    ocr_flag = {"used": False}
    rows, chunks = extract_all(pdf_path, progress, ocr_flag)

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
            "ocr": ocr_flag["used"],
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
