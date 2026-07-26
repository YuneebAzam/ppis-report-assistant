# PPIS Report Assistant

A fully **offline** chatbot that answers questions about the
**PPIS Annual Report 2014-15** (Petroleum Exploration & Production
Activities in Pakistan) — its text and all of its tables.
No internet, no API keys: everything runs on the local machine using a
small language model via [Ollama](https://ollama.com).

Example questions it answers (with page citations):

> *What was the total development expenditure for MOL?*
> *What was the total area for Dadyal?*
> *Which company operates the Tal block?*

## How it works

```
PDF ──(ingest.py, run once)──► index/rows.json    (master table: every table row,
                               index/chunks.json   structured; narrative text embedded)
                                   │
user question ──► app.py ──► search matching rows ──► local LLM ──► answer
                             (printed totals first;
                              Python does any sums)
```

- `ingest.py` extracts every page with **pdfplumber** (decoding the PDF's
  broken `(cid:NN)` font encoding), rebuilds each table's columns from the
  words' x-positions, and writes every table row as a structured record —
  table name, page, company/operator (filled down when the report leaves
  it blank), column = value pairs, and whether the row is a printed
  Total/Sub-Total. Non-table pages are chunked and embedded with
  `nomic-embed-text` for free-text questions.
- `app.py` is a Streamlit chat UI. For each question it keyword-searches
  the master table (question words weighted by rarity, so a block name
  outweighs generic words like "company"), prefers the report's own
  printed Total/Sub-Total rows, computes any needed sums in Python (the
  LLM never does arithmetic), and has `qwen2.5:3b-instruct-q4_K_M` phrase
  the answer **only from those rows**, citing page numbers. Questions
  that match no table row fall back to embedding search over the
  narrative text.

## Setup (once, needs internet)

1. Install [Python 3.10+](https://python.org) and [Ollama](https://ollama.com/download).
2. ```
   pip install -r requirements.txt
   ollama pull qwen2.5:3b-instruct-q4_K_M
   ollama pull nomic-embed-text
   ```
3. Put the report at `data/PPIS_Annual_2014-15.pdf` (already there).
4. Build the index (takes ~3 minutes):
   ```
   python ingest.py
   ```

## Run

```
streamlit run app.py
```

Opens at http://localhost:8501. Ollama must be running (it starts
automatically on Windows).

## Moving to offline work PCs

Nothing needs internet after setup. To install on a machine with no
internet access:

1. Copy this whole project folder.
2. Install Python + Ollama from their offline installers (download the
   installers beforehand).
3. `pip install` from a wheels folder prepared in advance:
   `pip download -r requirements.txt -d wheels\` (on the online PC), then
   `pip install --no-index --find-links wheels\ -r requirements.txt`.
4. Copy the Ollama models: the models folder
   (`C:\Users\<user>\.ollama\models`, or wherever `OLLAMA_MODELS` points)
   from the online PC to the same place on the offline PC.
5. The `index/` folder copies with the project, so `ingest.py` does not
   need to be run again.

## Weak PCs / server option

- The 3B model needs ~4 GB RAM and typically answers in 10–30 seconds on
  a decent CPU (the model only reads a dozen table rows per question);
  expect around a minute on weak all-in-one PCs, plus a one-time delay on
  the first question while the model loads. If a demo PC is too slow,
  pull `qwen2.5:1.5b-instruct` and change `CHAT_MODEL` at the top of
  `app.py` (faster, slightly less accurate).
- To serve many users from one machine: run Ollama and
  `streamlit run app.py --server.address 0.0.0.0` on the server, and staff
  open `http://<server-ip>:8501` in a browser. Change `OLLAMA_URL` in
  `app.py`/`ingest.py` if Ollama runs on a different machine than the app.

## Files

| File | Purpose |
|---|---|
| `ingest.py` | One-time PDF → searchable index |
| `app.py` | Streamlit chat app |
| `requirements.txt` | Python dependencies |
| `data/` | The report PDF |
| `index/` | Generated master row table (`rows.json`) + narrative chunks/embeddings (safe to regenerate) |
