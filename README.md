# RentWise-Boston

Project Description RentWise Boston is a mobile-friendly website with an integrated AI chatbot that helps people make informed decisions before renting a home in Boston.

## Contributors

- [Toci Nwaoha](https://github.com/TociNwaoha)
- [Harika Gummadi](https://github.com/harikagummadi582)
- [pradyotosan](https://github.com/prady2909)
- [Nandini Bommireddy](https://github.com/NandiniBommireddy)
- [Jonathan Blocksom](https://github.com/jblox26)

## Front end

A single-page app (Svelte 5 + Vite + TypeScript) where you ask a question about Boston's [RentSmart](https://data.boston.gov/dataset/rentsmart) housing data and get back an LLM-generated answer alongside a Mapbox map pinning the locations it references.

The app now calls the real RAG backend through [`src/lib/api/client.ts`](src/lib/api/client.ts) (`POST /api/ask` → `{ answer, locations, sources }`, contract in [`src/lib/types.ts`](src/lib/types.ts)). Vite proxies `/api` to the Python service on port 8000.

**You need the backend running, or every question will error.** See [Backend](#backend) below.

To work on the UI without running Python, set `VITE_USE_MOCK=true` in `.env` — that falls back to [`src/lib/api/mockBackend.ts`](src/lib/api/mockBackend.ts), which keyword-matches a few scenarios (rodents/pests, heat complaints, unsafe buildings, Dorchester) using real RentSmart rows.

### Prerequisites

- [Node.js](https://nodejs.org/) 20 or newer (with npm)
- A free [Mapbox access token](https://account.mapbox.com/access-tokens/) — the map won't render without one

### Setup

```bash
npm install
cp .env.example .env
```

Then edit `.env` and set `VITE_MAPBOX_TOKEN` to your Mapbox token. The file is gitignored — never commit it.

### Run

```bash
npm run dev
```

Open http://localhost:5173. Try one of the suggestion chips, or ask about an issue ("where are the rodents?") or a neighborhood ("what's happening in Dorchester?").

If the map panel says "Mapbox token needed", your `.env` is missing or the token isn't set; fix it and refresh (Vite restarts automatically when `.env` changes).

### Other commands

| Command | What it does |
| --- | --- |
| `npm run build` | Production build into `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run check` | Type-check Svelte + TypeScript sources |

## Backend

Agentic RAG over three Analyze Boston datasets. Full documentation — architecture,
tool design, data caveats, configuration — is in **[`backend/README.md`](backend/README.md)**.

**Everything runs locally. There are no cloud APIs and no API keys anywhere in this
project.** Generation is a local model served by [ollama](https://ollama.com); retrieval
is DuckDB plus local ONNX embeddings.

### Prerequisites

- Python 3.10 or newer (3.13 recommended)
- [ollama](https://ollama.com) — or any OpenAI-compatible local server (llama.cpp, LM Studio, vLLM)
- **8 GB RAM minimum**, 16 GB comfortable. See the model table below.

### Setup, from a fresh clone

Run these from the repo root. The data is **not** in git (190 MB) — you download it.

```bash
# 1. Python environment
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt

# 2. Download the three datasets from data.boston.gov (~110 MB, a minute or two)
.venv/bin/python -m backend.download

# 3. Build the DuckDB database + BM25 index (~7 seconds)
.venv/bin/python -m backend.ingest

# 4. The local model
brew install ollama
ollama pull qwen2.5:3b-instruct
```

#### Pick a model for your machine

**Use a non-reasoning instruct model.** This is measured, not a preference. Qwen3 is a
hybrid-reasoning model and emits 700–900 hidden thinking tokens even for a
two-sentence answer; at the ~26 tok/s an M2 sustains, that is 30–70 s per question.
Neither ollama's `think: false` nor Qwen3's `/no_think` switch suppressed it.

Measured on an 8 GB M2, same questions, same pipeline:

| Model | Address question | Owner question | Output tokens |
| --- | --- | --- | --- |
| `qwen3:4b` | 33.7 s | 74.2 s | 760–900 |
| `qwen2.5:3b-instruct` | **2.0 s** | **3.4 s** | **56–75** |

| Your RAM | Model | Notes |
| --- | --- | --- |
| 8 GB | `qwen2.5:3b-instruct` | 1.9 GB. The default |
| 16 GB | `qwen2.5:7b-instruct` | Better tool selection and SQL |
| 32 GB+ | `qwen2.5:14b-instruct` | Best of the three |

Set a non-default model with `RENTWISE_LOCAL_MODEL=qwen2.5:7b-instruct`.

Step 2 writes `data/downloads/*.csv`; step 3 writes `data/rentwise.duckdb`. Both are
gitignored and fully rebuildable, so never commit them. Re-run steps 2–3 any time to
pick up fresh data — RentSmart and the rental-eligibility file both refresh daily.

**Optional — dense retrieval.** Hybrid search works without this (it degrades to
BM25-only), but embeddings make paraphrased questions much better:

```bash
.venv/bin/python -m backend.build_embeddings          # busiest 20k cards (default)
```

The default embeds the 20,000 properties with the most history — the ones questions are
actually about — because that fits comfortably in memory.

> **Don't pass `--all` on 8 GB.** Embedding all 86k cards swaps heavily and takes over
> an hour; it will make the machine unusable while it runs. It is opt-in for that
> reason. Hybrid search works fine on the bounded set, and works at all with no
> embeddings whatsoever (BM25-only).

### Run

Three processes. The model server has to come first:

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve                 # terminal 1 — model
.venv/bin/uvicorn backend.app:app --port 8000 --reload   # terminal 2 — backend
npm run dev                                              # terminal 3 — frontend
```

Then open http://localhost:5173.

> **`OLLAMA_CONTEXT_LENGTH` is not optional.** ollama defaults to 4096 tokens, and the
> agent's system prompt, tool schemas and retrieved property cards exceed that. ollama
> does not error when you overflow — it silently drops the oldest tokens, which usually
> throws away the tool definitions, and the model then stops calling tools for no
> visible reason. The backend warns at startup if it detects a small context.

### Check it's working

```bash
curl localhost:8000/api/health
```

```json
{"status":"ok","llm_backend":"local","dense_retrieval":true,
 "tables":{"rentsmart":389121,"str_eligibility":396167,"property_cards":86397}}
```

- `llm_backend: "local"` — model reachable, agent is live.
- `llm_backend: "none"` — **no model server reachable, or the model isn't pulled.** The
  server still runs and retrieval still works, but answers are raw retrieval output
  instead of written prose. Check that `ollama serve` is up and `ollama list` shows your
  model. The backend logs the specific reason at startup.
- `dense_retrieval: false` — embeddings not built; BM25-only. Fine, just less accurate
  on paraphrased questions.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| "Could not reach the RAG backend" in the UI | uvicorn isn't running on port 8000 |
| `FileNotFoundError: data/rentwise.duckdb` | you skipped `backend.ingest` |
| `Missing required CSV(s)` | you skipped `backend.download` |
| Answers look like bullet lists of addresses | no model reachable — see `llm_backend` above |
| `No local model server at ...` | `ollama serve` isn't running |
| `Model '...' is not available` | run `ollama pull qwen2.5:3b-instruct` |
| Model answers but never uses the data | context too small — restart with `OLLAMA_CONTEXT_LENGTH=16384` |
| Very slow first answer | the model is loading into RAM; the second query is much faster |
| Map says "Mapbox token needed" | root `.env` missing `VITE_MAPBOX_TOKEN` |

### Questions worth trying

These take the fast catalog route and are reliable (1–4 s):

- `is 44 Portsmouth St a good place to rent?` — exact address lookup
- `where are the rat problems in Dorchester?` — catalog-expanded hybrid search
- `what else does GBM Portfolio Owner LLC own?` — owner-graph traversal
- `can I list 73 Hemenway St on Airbnb?` — short-term-rental eligibility
- `what is the meaning of life?` — should decline rather than invent

Counting, ranking and trend questions ("how many…", "which neighborhood has the
most…", "getting better or worse") are **not reliable yet** — see Known limitations in
[`backend/README.md`](backend/README.md#known-limitations).
