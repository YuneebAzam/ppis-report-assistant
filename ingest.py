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
    lines, cur, cur_sum = [], [], 0.0
    for w in words:
        # Compare against the running MEAN top of the line so far, not
        # the first word's top. A single outlier -- typically a table
        # rule misread as "|", sitting a couple of points high -- would
        # otherwise anchor the whole line and push genuinely aligned
        # words just past the tolerance onto a second line, splitting one
        # table row in two (observed on a Grand Total row, which then
        # lost half its figures).
        if cur and abs(w["top"] - cur_sum / len(cur)) > 3.5:
            lines.append(cur)
            cur, cur_sum = [], 0.0
        cur.append(w)
        cur_sum += w["top"]
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
    # --psm 6 ("assume a single uniform block of text") instead of the
    # default automatic page segmentation. On these report pages the
    # automatic mode splits a table into separate blocks and then drops
    # whole columns: on one reserves page it read only 1 of the 3 numbers
    # in each sub-total row (returning Cumulative Production where
    # Original Recoverable was asked for), and across sampled table pages
    # it found ~1.3x fewer numbers overall (3x fewer on the worst page).
    # Treating the page as one block keeps columns aligned to their rows.
    data = pytesseract.image_to_data(img, config="--psm 6",
                                     output_type=pytesseract.Output.DICT)

    words = words_from_ocr(data, zoom)
    lines = group_words_into_lines(words)

    # The two segmentation modes fail in different, complementary ways:
    # --psm 6 keeps a table's columns aligned to their rows but can miss
    # a whole row that sits apart from the main block (a Grand Total in
    # its own box under the table), while the automatic mode finds that
    # row but drops columns elsewhere. Take psm 6 as the base and add
    # only rows the automatic pass found in vertical gaps psm 6 left
    # empty, so nothing is overwritten -- purely additive recovery.
    auto = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    lines = merge_missing_lines(lines, group_words_into_lines(
        words_from_ocr(auto, zoom)))
    return reread_header_lines(lines, img, zoom, pytesseract)


def words_from_ocr(data, zoom):
    """Tesseract's per-word output -> word dicts in PDF point units."""
    out = []
    for i in range(len(data["text"])):
        text = re.sub(r"\s+", " ", data["text"][i]).strip()
        if not text or int(data["conf"][i]) < 0:
            continue
        if is_ocr_noise(text, int(data["conf"][i])):
            continue
        out.append({
            "text": text,
            "x0": data["left"][i] / zoom,
            "x1": (data["left"][i] + data["width"][i]) / zoom,
            "top": data["top"][i] / zoom,
        })
    return out


def merge_missing_lines(base_lines, extra_lines, v_tol=6):
    """Fill in what the base OCR pass missed, using a second pass.

    Works word by word rather than line by line. A word from the second
    pass is taken only when the base pass has nothing in that space --
    no word overlapping it horizontally within the same horizontal band.
    That covers both failure shapes seen in practice: an entire row the
    base pass skipped (a Grand Total sitting in its own box), and a
    single value missing from a row the base pass otherwise read (one
    page's grand total kept its Original and Balance figures but lost
    Cumulative). Existing words are never replaced, so a value the base
    pass did read always wins over the second pass's version of it."""
    if not extra_lines:
        return base_lines
    base_words = [w for ln in base_lines for w in ln]
    added = []
    for ln in extra_lines:
        for w in ln:
            collides = False
            for b in base_words:
                if abs(b["top"] - w["top"]) > v_tol:
                    continue
                if w["x0"] < b["x1"] and b["x0"] < w["x1"]:
                    collides = True
                    break
            if not collides:
                added.append(w)
    if not added:
        return base_lines
    return group_words_into_lines(base_words + added)


HAS_DIGIT_RE = re.compile(r"\d")


def is_ocr_noise(text, conf, min_conf=40, min_len=11):
    """Long, low-confidence, all-alphabetic tokens are OCR hallucinating
    words out of table rules and shaded cell backgrounds, e.g.
    "Mthetieieieietintal" or "Denadiunatinn" appearing inside a column
    header. Left in, they bury the real header word next to them
    ("origina") and the model then cannot tell one column from another.

    Uses Tesseract's own per-word confidence rather than a hand-rolled
    text-shape heuristic. Deliberately narrow so it cannot eat real
    content: anything containing a digit is always kept (no figure is
    ever dropped), and genuine short misreads keep their place --
    observed noise tokens ran 11-19 characters at confidence 9-22, while
    real misread header words ("orignal", "comutatve", 7-9 chars) sit
    below the length bar and survive even at low confidence. Losing one
    of those costs a column its name, so the bar only catches tokens
    longer than any real word seen in these headers."""
    if HAS_DIGIT_RE.search(text):
        return False
    letters = re.sub(r"[^A-Za-z]", "", text)
    return len(letters) >= min_len and conf < min_conf


def reread_header_lines(lines, img, zoom, pytesseract, max_scan=8):
    """Re-OCR a table's column-header lines from their own tight crop.

    In the whole-page pass the descenders of the big title printed just
    above a table bleed into the header line below it, and Tesseract then
    returns character salad for the header even though the header itself
    is perfectly crisp -- one page's "Company Field Discovery Year
    Original Recoverable ..." came back as "Pew fre ore pe Temes [ia".
    Losing those words matters more than losing any single cell, because
    they are what tells every value below which column it belongs to
    (without them "Original Recoverable" and "Balance Recoverable" become
    indistinguishable). Re-reading just the header's own horizontal band,
    with nothing above it to bleed in, recovers them.

    Only lines above the table body are considered (scanning stops at the
    first line that looks like data, i.e. mostly numbers), and a re-read
    is kept only when it yields more alphabetic content than the original
    -- so this cannot make a page worse than it already was."""
    for idx, ln in enumerate(lines[:max_scan]):
        n_num = sum(1 for w in ln if HAS_DIGIT_RE.search(w["text"]))
        if len(ln) >= 3 and n_num >= len(ln) / 2:
            break  # reached the data rows
        if len(ln) < 2:
            continue
        # Band this line only: from its own top down to the next line's
        # top. Cropping a fixed small height instead cuts tall header
        # text in half and makes the re-read worse than the original.
        top = min(w["top"] for w in ln) * zoom
        if idx + 1 < len(lines):
            bot = min(w["top"] for w in lines[idx + 1]) * zoom
        else:
            bot = top + 60 * zoom / (400 / 72)
        crop = img.crop((0, max(int(top) - 6, 0), img.width,
                         min(int(bot) + 4, img.height)))
        if crop.height < 12:
            continue
        try:
            d = pytesseract.image_to_data(
                crop, config="--psm 6", output_type=pytesseract.Output.DICT)
        except Exception:
            continue
        new_words = []
        for i in range(len(d["text"])):
            t = re.sub(r"\s+", " ", d["text"][i]).strip()
            if not t or int(d["conf"][i]) < 0:
                continue
            if is_ocr_noise(t, int(d["conf"][i])):
                continue
            new_words.append({
                "text": t,
                "x0": d["left"][i] / zoom,
                "x1": (d["left"][i] + d["width"][i]) / zoom,
                "top": min(w["top"] for w in ln),
            })
        def real_words(ws):
            """Count tokens of 4+ letters -- a proxy for actual header
            words. Plain letter-count is not enough: OCR noise like
            "ne en nn En EEE SEE SEES" has many letters but no real
            words, and would otherwise be accepted over the original."""
            return sum(1 for w in ws
                       if len(re.sub(r"[^A-Za-z]", "", w["text"])) >= 4)
        if new_words and real_words(new_words) > real_words(ln):
            new_words.sort(key=lambda w: w["x0"])
            lines[idx] = new_words
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
SUBTOTAL_NAMED_RE = re.compile(r"^Sub[\s-]?Totals?\s*\(([^)]+)\)$", re.IGNORECASE)
# A standalone "(Name)" cell -- the tail of a "Sub Total (Name)" label that
# OCR split across two column bins. Requires a letter so it cannot match a
# parenthesised number (some tables show negatives as "(1.234)").
PAREN_NAME_RE = re.compile(r"^\(\s*([^)\d][^)]*)\)$")
# A surviving fragment of a "Sub Total" label after OCR dropped the rest.
SUBTOTAL_FRAGMENT_RE = re.compile(r"^(Sub|Total|Sub-|Subtotal)$", re.IGNORECASE)
# Matches a row-label cell that announces a total. Anchored at the start
# but open-ended at the end, because a report may qualify the label --
# "Grand total of E&P Expenditure" is still a grand total, and requiring
# an exact match left that row unflagged and therefore unfindable.
TOTALROW_RE = re.compile(r"^(grand\s+|sub[\s-]?)?totals?\b", re.IGNORECASE)
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


def signatures_match(a, b, min_ratio=0.5):
    """Whether two page header signatures denote the same table.

    Exact equality is right for a clean text layer, where a repeated
    header is reproduced byte for byte. It fails completely on OCR: the
    same printed header comes back slightly differently on every page
    ("Right Holdings in Mining/D&P Leases" vs "Right | sno] holdings
    Lease in Mining/D&P | Area Leases"), so a 10-page table gets split
    into several one-page "sections". That is costly, because column
    boundaries and the entity fill-down are shared per section: split
    them and continuation pages lose the company each row belongs to,
    which is exactly what a question about a multi-page subtotal needs.

    Compares sets of alphabetic word-stems (first 4 letters, which
    absorbs OCR letter damage) by Jaccard similarity -- intersection over
    UNION. Using the smaller set as the denominator instead would call
    two different tables the same whenever one title's words are mostly a
    subset of the other's: "Oil 2P Reserves as on 30th June 2018
    (Million US Barrels)" and "Gas 2P Reserves as on 30th June 2019
    (Billion Cubic Feet)" share everything except the few words that
    actually distinguish them. Jaccard counts those differing words
    against the score, separating the two clearly."""
    if a is None or b is None:
        return False

    def stems(sig):
        words = re.findall(r"[A-Za-z]{3,}", " ".join(sig).lower())
        return {w[:4] for w in words}

    sa, sb = stems(a), stems(b)
    if not sa or not sb:
        return a == b
    return len(sa & sb) / len(sa | sb) >= min_ratio


def group_into_sections(pages_words):
    """Group consecutive pages that share the same header text -- i.e. a
    single table continuing across pages -- so column boundaries can be
    shared across all of a table's pages instead of derived per page."""
    sections, prev_sig = [], None
    for idx, lines in enumerate(pages_words):
        sig = header_signature(lines)
        if sections and signatures_match(sig, prev_sig):
            sections[-1].append(idx)
        else:
            sections.append([idx])
        if sig is not None:
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


DECIMAL_TAIL_RE = re.compile(r"^\.\d+$")
INT_HEAD_RE = re.compile(r"^-?[\d,]+$")
# A reported figure, as opposed to a serial number: has a decimal part
# or thousands separators, or is simply too long to be a row number.
FIGURE_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$|^-?\d+\.\d+$|^-?\d{4,}$")


def rejoin_split_decimals(cells):
    """Repair a number whose decimal part was binned as its own column.

    A wide figure sitting near a column boundary can have its fractional
    part pushed into the next cell, so "282,491.643" arrives as
    "282,491" followed by ".643" -- the figure is then wrong by three
    orders of magnitude wherever it is used. Only joins when the left
    cell is a bare integer and the right is exactly a decimal tail, so
    two genuine values are never merged."""
    out = list(cells)
    for i, left in enumerate(out):
        left = left.strip()
        if not INT_HEAD_RE.match(left):
            continue
        # The decimal tail is not always in the very next bin -- empty
        # columns can sit between the two halves -- so scan forward past
        # blanks for it.
        for j in range(i + 1, len(out)):
            right = out[j].strip()
            if not right:
                continue
            if DECIMAL_TAIL_RE.match(right):
                out[i] = left + right
                out[j] = ""
            break
    return out


def plausible_entity(text):
    """Whether a string can reasonably be a company or category name,
    used to spot entities that are really OCR debris. Table rules and
    shaded backgrounds produce fragments like "on = FN", "— =z" or "a",
    which then become the owner of a subtotal row and make it
    unfindable by its real name."""
    if not text:
        return False
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) < 3:
        return False
    if re.search(r"[=|~^{}\\<>]", text):
        return False
    return len(letters) / len(text) >= 0.5


def best_section_labels(section, pages_words, bounds):
    """Column labels from whichever page of a table OCR read best.

    Runs the normal label derivation once per page of the section and
    keeps the set containing the most real words (4+ letters), which is
    a direct proxy for how much of the printed header survived scanning.
    Returns None if no page yields labels, leaving the caller's usual
    first-page-wins behaviour untouched."""
    if not bounds:
        return None
    best, best_score = None, 0
    for idx in section:
        probe = {"col_labels": None, "header_cells": [],
                 "entity_col": 0, "current_entity": ""}
        render_page(pages_words[idx], bounds, probe, 0, "", [])
        labels = probe.get("col_labels")
        if not labels:
            continue
        score = sum(1 for l in labels
                    for w in re.findall(r"[A-Za-z]{4,}", l))
        if score > best_score:
            best, best_score = labels, score
    return best


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
                # Numbers in these tables are right-aligned, so their
                # midpoint is the stable way to place them. Text labels
                # are LEFT-aligned, and a long one (e.g. "Recoverable")
                # has a midpoint far to the right of where it starts --
                # enough to fall into the next column and detach a header
                # word from the column it labels. Binning alphabetic
                # tokens by their left edge keeps them with their column.
                anchor = (w["x0"] if not HAS_DIGIT_RE.search(w["text"])
                          else (w["x0"] + w["x1"]) / 2)
                cells[bisect.bisect_left(bounds, anchor)].append(w["text"])
            cells = [GLUED_RE.sub(r"\1 ", " ".join(c).strip())
                     for c in cells]
            cells = rejoin_split_decimals(cells)

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
            # OCR can split "Sub Total (OGDCL)" across two column bins,
            # leaving a bare "Sub Total" in one cell and "(OGDCL)" in the
            # next. The parenthesised name is still the row's own label
            # and beats the filled-down entity (which by then is whatever
            # data row came last, e.g. a field name). Generic: any
            # "(Name)" on a subtotal row, no report-specific wording.
            if not named_subtotal:
                paren = next(
                    (m for c in cells if (m := PAREN_NAME_RE.match(c))), None)
                # OCR sometimes loses part of the label too, leaving just
                # "Sub" (or just "Total") beside the "(Name)" cell, which
                # the strict Sub-Total patterns above will not match. A
                # numeric row carrying a standalone "(Name)" next to a
                # Sub/Total fragment is still that company's subtotal
                # line: without this the row keeps a junk entity ("Sub")
                # and cannot be found by the company's name at all.
                if paren and any(SUBTOTAL_FRAGMENT_RE.match(c) for c in cells):
                    named_subtotal, is_subtotal = paren, True

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
                # Some tables group rows under a heading in the same
                # column as the rows themselves (GOP Receipts: "Royalty"
                # over Oil/Gas, then "Sub total") rather than in a
                # separate company column. Filling down then leaves the
                # subtotal owned by the last detail row ("Gas"), or by
                # OCR junk from a garbled column. Fall back to the last
                # heading -- a labelled row carrying no figures, which is
                # what a category line looks like -- so the subtotal is
                # attributed to "Royalty"/"Dividend" and can be found by
                # that name.
                if is_subtotal and not plausible_entity(entity):
                    entity = state.get("last_heading", "") or entity
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

                # Remember category headings for the fallback above: a
                # row carrying one short text label and no figures. A
                # serial number beside the label ("2  Royalty") does not
                # disqualify it -- only a real figure does, so FIGURE_RE
                # ignores bare small integers.
                if not is_subtotal:
                    texts = [c for c in cells
                             if c and not FIGURE_RE.match(c.strip())
                             and not re.fullmatch(r"\d{1,3}", c.strip())]
                    figures = [c for c in cells
                               if c and FIGURE_RE.match(c.strip())]
                    if (not figures and len(texts) == 1
                            and plausible_entity(texts[0])
                            and len(texts[0]) <= 40):
                        state["last_heading"] = texts[0]

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
        # A table repeats its column header on every page it spans, and
        # OCR damages that header differently each time. Taking the
        # first page's version can saddle the whole table with unusable
        # labels ("freee | viscovery Year" over a column of figures)
        # while a later page read the same header cleanly -- and a table
        # whose columns have no usable names cannot be searched by
        # column. Seed the section with the best header any of its pages
        # produced.
        best = best_section_labels(section, pages_words, bounds)
        if best:
            state["col_labels"], state["entity_col"] = best, find_entity_col(best)
        page_text_lines = {}
        n_rows_before = len(rows)
        for idx in section:
            pageno = idx + 1
            page_text_lines[idx] = render_page(
                pages_words[idx], bounds, state, pageno, title, rows)
        if len(rows) == n_rows_before:  # no table detected: narrative text
            chunk_narrative_pages(section, page_text_lines, chunks)
    return drop_non_table_pages(rows), chunks


def drop_non_table_pages(rows, min_cells=3):
    """Discard "rows" scraped from pages that hold a chart rather than a
    table. The report's bar charts are full-page images like every other
    page, so image/vector counts cannot tell them apart; what does is
    that OCR finds no multi-column structure on them -- just axis
    numbers, rotated company names and a legend, which arrive as a
    handful of one-cell rows with junk labels ("a", "fon", "- =z").
    Left in, they answer questions with meaningless figures, and the
    report's charts only restate numbers its tables already give. A page
    keeps its rows as soon as ANY row has min_cells columns, so a real
    table is never dropped."""
    by_page = {}
    for r in rows:
        by_page.setdefault(r["page"], []).append(r)
    keep_pages = {p for p, rs in by_page.items()
                  if any(len(r["cells"]) >= min_cells for r in rs)}
    return [r for r in rows if r["page"] in keep_pages]


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
