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
from pathlib import Path

import numpy as np
import requests
import streamlit as st

from ingest import build_index, ollama_available

INDEX_DIR = Path(__file__).parent / "index"
DATA_DIR = Path(__file__).parent / "data"
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
- NEVER do arithmetic yourself. If the answer needs math that is not \
already given, say the report doesn't state it directly.
- If the question says "total" but a single row directly gives the value \
(e.g. the total area of one block), just answer with that row's value -- \
"total" in a question often simply means "the value".
- If the rows don't contain the answer, say plainly: "I couldn't find \
that in the report" -- do not guess.
- NEVER repeat or quote the raw row data lines back to the user. Reply \
with ONE short sentence containing just the specific value asked for \
(plus unit and page number). Example: "OGDCL's total development \
expenditure was 73,097.873 million rupees (page 96)."
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


def doc_description():
    meta = read_meta()
    if meta and not meta["pdf"].startswith("PPIS_Annual"):
        return f'the report "{meta["pdf"]}"'
    return DEFAULT_DOC


def query_words(question):
    return [w for w in re.findall(r"[A-Za-z][\w&-]*", question.lower())
            if w not in STOPWORDS and len(w) >= 2]


def word_patterns(w):
    """Exact words must match whole words ('tal' must not hit
    'Tallahassee' or 'total'); words longer than 5 letters also try a
    5-letter prefix stem so 'development' finds the report's
    'Develop. drilling' and 'operated' finds 'Operator'."""
    pats = [re.compile(rf"\b{re.escape(w)}\b")]
    if len(w) > 5:
        pats.append(re.compile(rf"\b{re.escape(w[:5])}"))
    return pats


def score_row(word_info, row, wants_total):
    """Sum of the rarity (idf) weights of the question words hitting this
    row. Rare words like a block name count far more than generic ones
    like 'company'; entity hits count double, table-title hits normal,
    other cell hits half."""
    score = 0.0
    ent = row["entity"].lower()
    tbl = row["table"].lower()
    for pats, idf in word_info:
        if any(p.search(ent) for p in pats):
            score += 2.0 * idf
        elif any(p.search(tbl) for p in pats):
            score += 1.0 * idf
        elif any(p.search(row["_blob"]) for p in pats):
            score += 0.5 * idf
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
    # keep only rows scoring close to the best match, printed totals first
    keep = [(s, i, r) for s, i, r in scored if s >= 0.7 * best]
    keep.sort(key=lambda t: (not t[2]["is_total"], -t[0], t[1]))
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


def format_rows(hits, extra):
    lines = []
    for r in hits:
        tag = "[PRINTED TOTAL] " if r["is_total"] else ""
        ent = f" | entity: {r['entity']}" if r["entity"] else ""
        cells = "; ".join(f"{k} = {v}" for k, v in r["cells"].items())
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


def search_chunks(question, chunks, emb, k=TOP_K_CHUNKS):
    if not chunks:
        return []
    qv = embed_query(question)
    sim = emb @ qv
    top = np.argsort(-sim)[:k]
    return [chunks[i] for i in top]


def build_context(question, rows, chunks, emb):
    hits = search_rows(question, rows)
    if hits:
        return format_rows(hits, computed_sums(question, hits)), hits, []
    fallback = search_chunks(question, chunks, emb)
    text = "\n\n---\n\n".join(c["text"] for c in fallback)
    return text, [], fallback


def stream_answer(question, context, history):
    user_msg = (f"Report data:\n\n{context}\n\n---\n\n"
                f"Question: {question}")
    messages = [{"role": "system",
                 "content": SYSTEM_PROMPT.format(doc=doc_description())}]
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
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

with st.sidebar:
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
        st.rerun()

if not (INDEX_DIR / "rows.json").exists():
    st.info("No report loaded yet — upload a PDF in the sidebar and click "
            "**Upload & ingest**.")
    st.stop()

rows, chunks, emb = load_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

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
