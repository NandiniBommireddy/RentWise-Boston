# RentWise-Boston

Project Description RentWise Boston is a mobile-friendly website with an integrated AI chatbot that helps people make informed decisions before renting a home in Boston.

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

### Prerequisites

- Python 3.10 or newer (3.13 recommended)
- An [Anthropic API key](https://console.anthropic.com/settings/keys) — or run without one, see below

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

# 4. Your Anthropic key
echo 'ANTHROPIC_API_KEY=sk-ant-...' > backend/.env
```

Step 2 writes `data/downloads/*.csv`; step 3 writes `data/rentwise.duckdb`. Both are
gitignored and fully rebuildable, so never commit them. Re-run steps 2–3 any time to
pick up fresh data — RentSmart and the rental-eligibility file both refresh daily.

**Optional — dense retrieval.** Hybrid search works without this (it degrades to
BM25-only), but embeddings make paraphrased questions much better:

```bash
.venv/bin/python -m backend.build_embeddings                  # all 86k cards, slow
.venv/bin/python -m backend.build_embeddings --limit 20000    # busiest 20k, ~2 min
```

### Run

Two processes, two terminals:

```bash
.venv/bin/uvicorn backend.app:app --port 8000 --reload   # terminal 1 — backend
npm run dev                                              # terminal 2 — frontend
```

Then open http://localhost:5173.

### Check it's working

```bash
curl localhost:8000/api/health
```

```json
{"status":"ok","llm_backend":"claude","dense_retrieval":true,
 "tables":{"rentsmart":389121,"str_eligibility":396167,"property_cards":86397}}
```

- `llm_backend: "claude"` — key found, agent is live.
- `llm_backend: "none"` — **no key found.** The server still runs and retrieval still
  works, but answers are raw retrieval output instead of written prose. Check
  `backend/.env`.
- `dense_retrieval: false` — embeddings not built; BM25-only. Fine, just less accurate
  on paraphrased questions.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| "Could not reach the RAG backend" in the UI | uvicorn isn't running on port 8000 |
| `FileNotFoundError: data/rentwise.duckdb` | you skipped `backend.ingest` |
| `Missing required CSV(s)` | you skipped `backend.download` |
| Answers look like bullet lists of addresses | no API key — see `llm_backend` above |
| Map says "Mapbox token needed" | root `.env` missing `VITE_MAPBOX_TOKEN` |

### Questions worth trying

- `is 44 Portsmouth St a good place to rent?` — exact address lookup
- `which neighborhood has the worst rat problem?` — routes to SQL, since retrieval can't count
- `what else does GBM Portfolio Owner own?` — owner-graph traversal, 144 properties
- `can I list my place in Dorchester on Airbnb?` — short-term-rental eligibility
