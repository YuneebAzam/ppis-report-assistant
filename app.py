"""
PPIS Report Assistant -- offline chatbot over the PPIS Annual Report.

Run:  streamlit run app.py
Needs: Ollama running locally, and `python ingest.py` run once beforehand.

How a question is answered:
  1. The user's question is keyword-matched against the master table of
     structured rows (index/rows.json) extracted from every table in the
     PDF: a row scores by how many question words hit its table title,
     entity name, or cell contents.
  2. Printed "Total"/"Sub-Total for X" rows that match are preferred and
     placed first -- the report's own totals beat anything computed.
  3. If the question asks for a total and NO printed total matched,
     Python sums the matching rows per numeric column and passes that
     computed sum along (clearly labelled) -- the model never does math.
  4. Narrative (non-table) questions fall back to embedding retrieval
     over index/chunks.json.
  5. The matching rows are serialized as "column = value" lines and the
     local LLM (via Ollama) writes the final plain-English answer.
"""
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np
import requests
import streamlit as st

from ingest import build_index, ollama_available

INDEX_DIR = Path(__file__).parent / "index"
DATA_DIR = Path(__file__).parent / "data"
SESSIONS_DIR = Path(__file__).parent / "sessions"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
MAX_ROWS = 12       # table rows handed to the model
TOP_K_CHUNKS = 4    # narrative chunks handed to the model

DEFAULT_DOC = ("the PPIS Annual Report 2014-15 (Petroleum Exploration & "
               "Production Activities in Pakistan)")

SYSTEM_PROMPT = """You are a helpful assistant for {doc}. \
You answer questions from non-technical staff.

You are given rows extracted from the report's tables. Each row shows its \
table name, page number, and "column = value" pairs.

Rules:
- Answer ONLY from the rows/excerpts provided. Never invent numbers.
- Quote figures exactly as given, with the unit if shown in the column name.
- Always mention the page number, like (page 94).
- Rows starting with [PRINTED TOTAL] are the report's own total lines -- \
prefer them when the question asks for a total.
- A line starting with [COMPUTED] is a sum calculated by this app from \
the matching rows; present it as a computed sum, not a printed figure.
- A line starting with [ANSWER CELL] is the exact figure this app \
matched to the question, already checked against the column names. When \
it is present, answer with THAT figure and its page, even if other rows \
below show similar-looking numbers. Column names may contain scanning \
errors ("comutatve" for Cumulative, "origina" for Original); the \
[ANSWER CELL] line has already accounted for them.
- NEVER do arithmetic yourself. If the answer needs math that is not \
already given, say the report doesn't state it directly.
- If the question says "total" but a single row directly gives the value \
(e.g. the total area of one block), just answer with that row's value -- \
"total" in a question often simply means "the value".
- If the given rows cover SEVERAL DIFFERENT companies and the question \
did not name a specific one, do NOT silently pick one company and answer \
as if that were the whole answer. Instead say the report breaks this \
figure down by company rather than giving one combined number, and name \
2-3 examples with their values and pages.
- If the rows don't contain the answer, say plainly: "I couldn't find \
that in the report" -- do not guess.
- Sometimes both table rows AND narrative text excerpts are given. They \
were retrieved independently and may not both be relevant -- use \
whichever one actually answers the question, and ignore the other. \
Structured rows are not automatically more trustworthy than a clear \
narrative sentence that directly states the answer.
- If a section is labelled as the more reliable source, and a narrative \
sentence there directly answers the question, use that sentence's figure \
even if a table row shows a different number for the same thing. Copy \
figures digit for digit; never rescale them (14.73 million must not \
become 14,730).
- NEVER repeat or quote the raw row data lines back to the user. Reply \
with ONE short sentence containing just the specific value asked for \
(plus unit and page number), in this shape: \
"<company>'s <metric> was <figure> <unit> (page <n>)." \
The placeholders show the SHAPE only -- never emit a company, figure or \
page that is not in the rows above.
- Keep answers short, clear and friendly."""

STOPWORDS = {"what", "was", "the", "for", "total", "how", "many", "much",
             "and", "are", "were", "does", "did", "report", "tell", "about",
             "which", "with", "that", "this", "have", "has", "had", "is",
             "in", "of", "during", "a", "an", "by", "to", "me", "show",
             "give", "list", "all", "per", "each"}
TOTAL_WORDS = {"total", "sum", "overall", "combined", "altogether"}
NUM_RE = re.compile(r"^-?[\d,]+(\.\d+)?$")


@st.cache_resource
def load_index():
    rows = json.loads((INDEX_DIR / "rows.json").read_text(encoding="utf-8"))
    for r in rows:
        r["_blob"] = " ".join(
            [r["table"], r["entity"]] +
            [f"{k} {v}" for k, v in r["cells"].items()]).lower()
    chunks, emb = [], None
    if (INDEX_DIR / "chunks.json").exists():
        chunks = json.loads(
            (INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        emb = np.load(INDEX_DIR / "embeddings.npy")
    return rows, chunks, emb


def read_meta():
    p = INDEX_DIR / "meta.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def index_used_ocr():
    """True when the loaded document had no usable embedded text layer and
    every table had to be reconstructed by OCR -- recorded at ingest time
    (see ingest.build_index). Older indexes lack the key, so default False."""
    meta = read_meta()
    return bool(meta and meta.get("ocr"))


def doc_description():
    meta = read_meta()
    if meta and not meta["pdf"].startswith("PPIS_Annual"):
        return f'the report "{meta["pdf"]}"'
    return DEFAULT_DOC


def conversation_title(messages):
    for m in messages:
        if m["role"] == "user":
            t = m["content"].strip()
            return t if len(t) <= 60 else t[:57] + "..."
    return "New chat"


def save_conversation(conv_id, messages):
    if not messages:
        return
    SESSIONS_DIR.mkdir(exist_ok=True)
    path = SESSIONS_DIR / f"{conv_id}.json"
    created = time.time()
    if path.exists():
        try:
            created = json.loads(path.read_text(encoding="utf-8"))["created"]
        except Exception:
            pass
    meta = read_meta()
    data = {
        "id": conv_id,
        "title": conversation_title(messages),
        "doc": meta["pdf"] if meta else None,
        "created": created,
        "updated": time.time(),
        "messages": messages,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                    encoding="utf-8")


def list_conversations():
    if not SESSIONS_DIR.exists():
        return []
    convos = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            convos.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    convos.sort(key=lambda c: c.get("updated", 0), reverse=True)
    return convos


def delete_conversation(conv_id):
    path = SESSIONS_DIR / f"{conv_id}.json"
    if path.exists():
        path.unlink()


MONTH_ABBR = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "may": "may", "june": "jun", "july": "jul", "august": "aug",
    "september": "sep", "october": "oct", "november": "nov", "december": "dec",
}
MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(MONTH_ABBR) + r")\.?\s+(\d{4})\b", re.IGNORECASE)


def query_words(question):
    # Block/lease codes like "2467-16" start with a digit -- they must
    # still count as query words, since they're often the most specific
    # (and only) identifying term in a question about a particular block.
    words = [w for w in re.findall(r"[A-Za-z0-9][\w&-]*", question.lower())
             if w not in STOPWORDS and len(w) >= 2 and not w.isdigit()]
    # Monthly tables print dates as "Jul-14", not "July 2014" -- add the
    # report's own abbreviation as an extra search word so a natural date
    # phrase can still hit the right row directly.
    for month, year in MONTH_YEAR_RE.findall(question):
        words.append(f"{MONTH_ABBR[month.lower()]}-{year[2:]}")
    return words


def word_patterns(w):
    """Exact words must match whole words ('tal' must not hit
    'Tallahassee' or 'total'); alphabetic words longer than 5 letters
    also try a 5-letter prefix stem so 'development' finds the report's
    'Develop. drilling' and 'operated' finds 'Operator'. Codes with
    digits/hyphens (block numbers like '2467-16') never get the stem
    treatment -- '2467-16'[:5] is '2467-', which would prefix-match
    every other block starting with 2467 (2467-9, 2467-11, ...), turning
    a specific code into a near-meaningless wildcard."""
    pats = [re.compile(rf"\b{re.escape(w)}\b")]
    if len(w) > 5 and w.isalpha():
        pats.append(re.compile(rf"\b{re.escape(w[:5])}"))
    return pats


def score_row(word_info, row, wants_total):
    """Sum of the rarity (idf) weights of the question words hitting this
    row. Rare words like a block name count far more than generic ones
    like 'company'; entity hits count double, table-title hits normal,
    other cell hits half.

    A coverage multiplier then rewards rows that match MORE of the
    question's distinct words over ones that only match one -- without
    it, a single lucky rare-word hit in an unrelated row's entity (e.g.
    "Shah" inside an unrelated well named "Fateh Shah North-2") can
    outscore the real row that matches several of the question's words
    together (e.g. "Shah" AND "Bandar" both appearing in one block's
    data), since one entity-level hit (2x weight) can otherwise beat two
    or more blob-level hits (0.5x each)."""
    score = 0.0
    hits = 0
    ent = row["entity"].lower()
    tbl = row["table"].lower()
    for pats, idf in word_info:
        if any(p.search(ent) for p in pats):
            score += 2.0 * idf
            hits += 1
        elif any(p.search(tbl) for p in pats):
            score += 1.0 * idf
            hits += 1
        elif any(p.search(row["_blob"]) for p in pats):
            score += 0.5 * idf
            hits += 1
    if word_info:
        score *= 0.4 + 0.6 * hits / len(word_info)
    if wants_total and row["is_total"] and score > 0:
        score *= 1.15  # the printed total should win near-ties
    return score


def search_rows(question, rows):
    words = query_words(question)
    if not words:
        return []
    wants_total = any(w in question.lower() for w in TOTAL_WORDS)
    n = len(rows)
    word_info = []
    for w in words:
        pats = word_patterns(w)
        df = sum(1 for r in rows if any(p.search(r["_blob"]) for p in pats))
        word_info.append((pats, np.log(n / (1 + df)) + 0.1 if df else 0.0))
    scored = [(score_row(word_info, r, wants_total), i, r)
              for i, r in enumerate(rows)]
    scored = [(s, i, r) for s, i, r in scored if s > 0]
    if not scored:
        return []
    best = max(s for s, _, _ in scored)
    # Keep only rows scoring close to the best match. Sort printed totals
    # first, then by score; on a tie, an entity-less "grand total" row
    # (e.g. a table's own overall total) wins over a specific company's
    # row -- otherwise it always loses ties to whichever company happens
    # to appear earlier in the document, since it's usually the very
    # last row of its table. Original position is the final tiebreaker.
    keep = [(s, i, r) for s, i, r in scored if s >= 0.7 * best]
    keep.sort(key=lambda t: (not t[2]["is_total"], -t[0],
                             bool(t[2]["entity"]), t[1]))
    return [r for _, _, r in keep[:MAX_ROWS]]


def computed_sums(question, hits):
    """If the user asked for a total and no printed total row matched,
    sum each numeric column across the matching rows in Python."""
    ql = question.lower()
    if not any(w in ql for w in TOTAL_WORDS):
        return None
    if any(r["is_total"] for r in hits):
        return None  # a printed total is available; prefer it
    sums, counts = {}, {}
    for r in hits:
        for col, val in r["cells"].items():
            v = val.replace(",", "").strip()
            if NUM_RE.match(v):
                sums[col] = sums.get(col, 0.0) + float(v)
                counts[col] = counts.get(col, 0) + 1
    parts = [f"{col} = {sums[col]:,.2f} (sum of {counts[col]} rows)"
             for col in sums if counts[col] >= 2]
    return "; ".join(parts) if parts else None


def pinpoint_cell(question, hits):
    """Name the single cell that answers the question, when one column
    clearly matches, so the model does not have to pick from a row of
    lookalike figures.

    OCR mangles column headers just enough to break plain word matching
    while leaving them readable to a person: a "Comulative Production"
    column arrives as "comutatve Production" and "Original Recoverable"
    as "origina Recoverable". The question's own words then match no
    label exactly, and a small model picks a neighbouring number
    instead. Fuzzy-matching each question word against the label's words
    tolerates that damage, and picking the column here keeps the value
    selection in Python -- the model is left only to phrase it, the same
    split already used for sums."""
    import difflib
    words = [w for w in query_words(question) if len(w) >= 5]
    if not words or not hits:
        return None
    # Consider printed total rows first and stop if one matches: a
    # question naming a company wants that company's total line, not one
    # of the individual field rows above it that share its wording.
    ordered = [h for h in hits if h["is_total"]] or list(hits)
    best = (0.0, None, None, None)
    for h in ordered[:4]:
        for label, value in h["cells"].items():
            if not NUM_RE.match(value.replace(",", "").strip()):
                continue
            label_words = re.findall(r"[A-Za-z]{3,}", label.lower())
            if not label_words:
                continue
            score = 0.0
            for qw in words:
                m = max((difflib.SequenceMatcher(None, qw, lw).ratio()
                         for lw in label_words), default=0.0)
                if m >= 0.7:          # tolerate OCR letter damage
                    score += m
            if score > best[0]:
                best = (score, label, value, h)
    # Require two matching label words (e.g. "original"+"recoverable"),
    # so a single generic hit like "production" cannot claim the row.
    if best[0] >= 1.5:
        _, label, value, h = best
        return f"[ANSWER CELL] (page {h['page']}) {label} = {value}"
    return None


def essential_cells(cells):
    """When a row carries 2+ pre-computed '... Total (In MM Rs.)' columns
    (Table 5.2's rows: Exploration/Appraisal/Development totals, each
    already the correct sum of that section's raw sub-columns), show only
    those totals plus the non-numeric label cells. The raw sub-columns
    stay in rows.json for inspection, but showing them next to the
    already-computed totals invites a small model to "helpfully" re-add
    them itself instead of just reading the given total -- which is
    exactly the kind of arithmetic it tends to get wrong."""
    total_keys = [k for k in cells if k.endswith("Total (In MM Rs.)")]
    if len(total_keys) < 2:
        return cells
    return {k: v for k, v in cells.items()
            if k in total_keys or not NUM_RE.match(v.replace(",", ""))}


def format_rows(hits, extra, pinpoint=None):
    lines = []
    if pinpoint:
        lines.append(pinpoint)
    for r in hits:
        tag = "[PRINTED TOTAL] " if r["is_total"] else ""
        ent = f" | entity: {r['entity']}" if r["entity"] else ""
        cells = "; ".join(f"{k} = {v}"
                          for k, v in essential_cells(r["cells"]).items())
        lines.append(f"{tag}(page {r['page']}) {r['table']}{ent} | {cells}")
    if extra:
        lines.append(f"[COMPUTED] {extra}")
    return "\n".join(lines)


def embed_query(text):
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": EMBED_MODEL, "input": [text]}, timeout=120)
    r.raise_for_status()
    v = np.array(r.json()["embeddings"][0], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


MAX_CHUNKS_PER_PAGE = 2  # a long narrative page can get split into several
                         # overlapping windows that all score similarly
                         # well for the same question -- without a cap,
                         # they can crowd out every retrieval slot with
                         # near-duplicate text, leaving no room for a
                         # genuinely different page that's also relevant.


def search_chunks(question, chunks, emb, k=TOP_K_CHUNKS):
    if not chunks:
        return []
    qv = embed_query(question)
    sim = emb @ qv
    ranked = np.argsort(-sim)
    picked, per_page = [], {}
    for i in ranked:
        page = chunks[i]["page"]
        if per_page.get(page, 0) >= MAX_CHUNKS_PER_PAGE:
            continue
        picked.append(chunks[i])
        per_page[page] = per_page.get(page, 0) + 1
        if len(picked) >= k:
            break
    return picked


DIFFUSE_ENTITY_LIMIT = 3  # matches across more distinct companies than
                          # this means the question isn't about one
                          # specific company -- it likely wants an
                          # industry-wide figure, which this report states
                          # in prose (e.g. page 92), not as a single row.


def collapse_to_totals(hits):
    """When both a company's own Sub-Total row and its individual
    line-item rows (blocks/fields) are present in the same result set,
    keep only the Sub-Total -- otherwise a small model can grab a line
    item's own similarly-shaped "Total" column instead of the
    company-wide figure a plain "what is X's <metric>" question wants.
    Entities with no Sub-Total among the hits (or a blank/report-wide
    entity) are left untouched. This runs regardless of whether the
    question used the word "total", since that word being absent
    doesn't mean the user wants one specific line item."""
    by_entity = {}
    order = {}
    for i, h in enumerate(hits):
        by_entity.setdefault(h["entity"], []).append(h)
        order[id(h)] = i
    out = []
    for entity, group in by_entity.items():
        totals = [h for h in group if h["is_total"]]
        out.extend(totals if entity and totals else group)
    out.sort(key=lambda h: order[id(h)])
    return out


def restrict_to_named_table(question, rows):
    """Pick the table first when the question names one, then let row
    ranking work inside it.

    Sibling tables ("Oil 2P Reserves ...", "Gas 2P Reserves ...") share
    companies and column wording, so row scores alone can send an oil
    question to the gas table -- especially when OCR left the oil
    table's column labels more damaged than the gas table's, making the
    wrong table the better *textual* match. The table title is the one
    place the distinguishing word reliably survives, so a word that
    appears in one table's title and not others decides the table
    outright. Returns all rows unchanged when no title stands out."""
    words = [w for w in query_words(question) if len(w) >= 3]
    if not words:
        return rows
    titles = {r["table"] for r in rows if r["table"]}
    if len(titles) < 2:
        return rows
    # A word is a table selector only if it is in some titles but not
    # most -- "reserves" appears in both sibling titles and decides
    # nothing, while "oil" or "gas" appears in one.
    scores = {}
    for t in titles:
        tl = t.lower()
        scores[t] = sum(1 for w in words
                        if re.search(rf"\b{re.escape(w)}", tl))
    best = max(scores.values())
    if best == 0:
        return rows
    winners = {t for t, s in scores.items() if s == best}
    discriminating = sum(1 for s in scores.values() if s == best)
    if discriminating > 3:      # too many tables tie -- not a selector
        return rows
    picked = [r for r in rows if r["table"] in winners]
    return picked or rows


def same_table_as_best(hits):
    """Keep only rows from the best-matching table.

    Reports often carry near-identical sibling tables -- "Oil 2P
    Reserves ..." and "Gas 2P Reserves ...", same companies, same column
    names, different subject. A question naming one of them still
    retrieves the other's rows, because the company name and the column
    wording match there too. Showing both lets the model answer an oil
    question from the gas table. The top-ranked hit already encodes
    which table the question matched best, so anything from a different
    table is dropped. Rows without a table label are always kept."""
    if len(hits) < 2:
        return hits
    best_table = hits[0].get("table", "")
    if not best_table:
        return hits
    return [h for h in hits
            if h.get("table", "") == best_table or not h.get("table")]


def build_context(question, rows, chunks, emb):
    hits = search_rows(question, restrict_to_named_table(question, rows))
    hits = collapse_to_totals(hits)
    hits = same_table_as_best(hits)
    wants_total = any(w in question.lower() for w in TOTAL_WORDS)

    # Once a printed total/sub-total row is available, drop the
    # non-total line-item rows entirely -- individual rows sometimes
    # carry their own similarly-labelled "Total" column too (e.g. one
    # block's own development total), and showing both lets a small
    # model latch onto the wrong one instead of the row that actually
    # answers "the total".
    if wants_total and any(h["is_total"] for h in hits):
        hits = [h for h in hits if h["is_total"]]

    # A report-wide row (no specific company, e.g. a table's own grand
    # total) is a strong, unambiguous match for a question that didn't
    # name a company -- prefer it outright over per-company rows. Require
    # a literal bare "Total" cell (not "Sub-Total"): an empty entity on a
    # Sub-Total row means entity-tracking failed for that row, not that
    # it's a genuine report-wide total.
    global_hits = [h for h in hits if not h["entity"] and h["is_total"]
                  and any(v.strip() == "Total" for v in h["cells"].values())]
    if global_hits:
        return (format_rows(global_hits, computed_sums(question, global_hits),
                            pinpoint_cell(question, global_hits)), global_hits, [])

    # Diffuseness is judged from only the top few hits (already
    # score-sorted), not the whole (up to MAX_ROWS) window -- a handful
    # of low-scoring long-tail matches from unrelated companies (e.g. a
    # different well whose name happens to share one query word) can
    # otherwise trigger a false "diffuse" verdict even when there's a
    # clear best match at the top.
    distinct_entities = {h["entity"] for h in hits[:3] if h["entity"]}
    if hits and len(distinct_entities) <= DIFFUSE_ENTITY_LIMIT:
        row_text = format_rows(hits, computed_sums(question, hits),
                               pinpoint_cell(question, hits))
        # Below the strong, unambiguous global_hits case above, no
        # heuristic reliably tells a genuinely relevant table match apart
        # from a coincidental keyword collision (e.g. a "Reserves"
        # table's "Cumulative Production" column, or a company literally
        # named "...Oil..." matching a question about oil production) --
        # tables and company names in an unknown future PDF could contain
        # any word. Rather than a confidence classifier that just shifts
        # which edge cases it misses, always also offer the
        # independently-scored narrative text alongside the rows, and
        # let the model (already instructed to use whichever source
        # actually answers) discriminate -- more robust across arbitrary
        # PDFs than any keyword-based confidence rule.
        fallback = search_chunks(question, chunks, emb)
        if fallback:
            chunk_text = "\n\n---\n\n".join(c["text"] for c in fallback)
            if index_used_ocr():
                # This document had no usable text layer, so every table
                # was reconstructed from OCR. OCR reads flowing prose far
                # more reliably than dense tabular layout (a general
                # property of OCR, not of any particular report): narrow
                # columns of small digits are its hardest case, and
                # observed damage here included both wrong column labels
                # and a misread digit (568.44 -> 968.446). So put the
                # narrative first and say it is the more reliable source.
                combined = (
                    f"NARRATIVE TEXT (more reliable for this document):\n\n"
                    f"{chunk_text}\n\n--- Table rows (reconstructed by OCR; "
                    f"labels and digits may be imperfect -- prefer the "
                    f"narrative text above when the two disagree) ---\n\n"
                    f"{row_text}")
            else:
                combined = (f"{row_text}\n\n--- Additional excerpts (may or "
                           f"may not be more relevant than the rows above) "
                           f"---\n\n{chunk_text}")
            return combined, hits, fallback
        return row_text, hits, []

    # No row match, or a diffuse match spanning many different companies:
    # try the narrative text, which is where report-wide totals live.
    fallback = search_chunks(question, chunks, emb)
    if fallback:
        text = "\n\n---\n\n".join(c["text"] for c in fallback)
        return text, [], fallback
    if hits:  # no narrative alternative -- fall back to the diffuse rows
        return (format_rows(hits, computed_sums(question, hits),
                            pinpoint_cell(question, hits)), hits, [])
    return "", [], []


def stream_answer(question, context, history):
    user_msg = (f"Report data:\n\n{context}\n\n---\n\n"
                f"Question: {question}")
    messages = [{"role": "system",
                 "content": SYSTEM_PROMPT.format(doc=doc_description())}]
    # Carry only the earlier QUESTIONS, never the earlier answers.
    # Answers are re-derived from the index every time, so replaying old
    # ones lets a figure from a previous session outrank what retrieval
    # just found -- reopening a saved chat had the model repeating a
    # number that no longer exists in the index at all. Past questions
    # still give follow-ups their context ("what about MOL?") without
    # putting a stale figure in front of the model.
    for h in history[-6:]:
        if h["role"] == "user":
            messages.append({"role": "user", "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})

    with requests.post(f"{OLLAMA_URL}/api/chat",
                       json={"model": CHAT_MODEL, "messages": messages,
                             "stream": True,
                             "options": {"temperature": 0.1,
                                         "num_ctx": 8192}},
                       stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token


st.set_page_config(page_title="PPIS Report Assistant", page_icon="🛢️")
st.title("🛢️ PPIS Report Assistant")
st.caption("Ask anything about the loaded report — "
           "runs fully offline on this PC.")

if not ollama_available():
    st.error("Ollama is not running. Start Ollama and refresh this page.")
    st.stop()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

DELETE_BTN_CSS = """
<style>
div[class*="st-key-del_"] button {
    background-color: transparent !important;
    border: none !important;
    width: 2.1em !important;
    height: 2.1em !important;
    min-width: 2.1em !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-top: 0.15em;
}
div[class*="st-key-del_"] button:hover {
    background-color: rgba(229, 57, 53, 0.12) !important;
    border-radius: 50% !important;
}
div[class*="st-key-del_"] button * {
    color: #e53935 !important;
    fill: #e53935 !important;
}
</style>
"""

with st.sidebar:
    st.subheader("🕘 Chat history")
    st.markdown(DELETE_BTN_CSS, unsafe_allow_html=True)
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()
    meta_now = read_meta()
    index_built = None
    if meta_now and meta_now.get("ingested"):
        try:
            index_built = time.mktime(time.strptime(meta_now["ingested"],
                                                    "%Y-%m-%d %H:%M"))
        except ValueError:
            index_built = None
    for c in list_conversations():
        label = c.get("title", "New chat")
        # Answers in a chat older than the current index were produced
        # from data that has since changed; mark them so an out-of-date
        # figure on screen is not mistaken for a current one.
        if index_built and c.get("updated", 0) < index_built:
            label = "⚠ " + label
        is_current = c["id"] == st.session_state.conversation_id
        col1, col2 = st.columns([5, 1], vertical_alignment="center")
        with col1:
            if st.button(("• " if is_current else "") + label,
                        key=f"conv_{c['id']}", use_container_width=True):
                st.session_state.messages = c["messages"]
                st.session_state.conversation_id = c["id"]
                st.rerun()
        with col2:
            with st.container(key=f"del_{c['id']}"):
                if st.button("", icon=":material/delete:",
                            key=f"delbtn_{c['id']}"):
                    delete_conversation(c["id"])
                    if is_current:
                        st.session_state.messages = []
                        st.session_state.conversation_id = str(uuid.uuid4())
                    st.rerun()

    st.divider()
    st.subheader("📄 Report document")
    meta = read_meta()
    if meta:
        st.caption(f"Loaded: **{meta['pdf']}**\n\n"
                   f"{meta['rows']} table rows · {meta['chunks']} text "
                   f"chunks · ingested {meta['ingested']}")
    else:
        st.caption("No document ingested yet.")
    uploaded = st.file_uploader("Upload a report PDF", type="pdf")
    if uploaded is not None and st.button("Upload & ingest",
                                          type="primary",
                                          use_container_width=True):
        DATA_DIR.mkdir(exist_ok=True)
        pdf_path = DATA_DIR / uploaded.name
        pdf_path.write_bytes(uploaded.getbuffer())
        bar = st.progress(0.0, text="Starting ingestion ...")

        def on_progress(stage, done, total):
            # reading pages is ~80% of the work, embedding the rest
            frac = done / max(total, 1)
            pct = 0.8 * frac if stage == "Reading pages" else 0.8 + 0.2 * frac
            bar.progress(min(pct, 1.0), text=f"{stage} ({done}/{total})")

        try:
            new_meta = build_index(pdf_path, on_progress)
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            st.stop()
        bar.progress(1.0, text="Done!")
        st.success(f"Ingested {new_meta['pdf']}: {new_meta['rows']} table "
                   f"rows, {new_meta['chunks']} text chunks.")
        st.cache_resource.clear()
        st.session_state.messages = []  # old chat is about the old document
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()

if not (INDEX_DIR / "rows.json").exists():
    st.info("No report loaded yet — upload a PDF in the sidebar and click "
            "**Upload & ingest**.")
    st.stop()

rows, chunks, emb = load_index()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("e.g. What was MOL's development expenditure?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the report..."):
            context, row_hits, chunk_hits = build_context(
                question, rows, chunks, emb)
        if not context:
            answer = "I couldn't find that in the report."
            st.markdown(answer)
        else:
            answer = st.write_stream(stream_answer(
                question, context, st.session_state.messages[:-1]))
            with st.expander("Sources used"):
                for r in row_hits:
                    st.markdown(f"**Page {r['page']}** — {r['table']}"
                                + (f" ({r['entity']})" if r["entity"] else ""))
                for c in chunk_hits:
                    st.markdown(f"**Page {c['page']}** — {c['title']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_conversation(st.session_state.conversation_id, st.session_state.messages)
