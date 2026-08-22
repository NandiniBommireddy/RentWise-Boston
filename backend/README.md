# RentWise Boston — RAG backend

Agentic RAG over Boston open data. Answers `POST /api/ask {query}` with the
`RagResponse` contract the Svelte frontend already expects (`src/lib/types.ts`),
plus additive `citations` and `trace` fields.

## Quick start

```bash
# 1. Python env
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt

# 2. Data: download the CSVs, then build the DuckDB database (~7s)
.venv/bin/python -m backend.download
.venv/bin/python -m backend.ingest

# 3. Optional: local embeddings for hybrid search. Slow on Apple Silicon --
#    measured well over 30 min for all 86k cards on an 8 GB M2. Prefer --limit,
#    which covers the busiest properties (the ones questions are actually about).
.venv/bin/python -m backend.build_embeddings --limit 20000

# 4. Credentials — put your key in backend/.env
echo 'ANTHROPIC_API_KEY=sk-ant-...' > backend/.env

# 5. Run both processes
.venv/bin/uvicorn backend.app:app --port 8000 --reload   # terminal 1
npm run dev                                              # terminal 2
```

Open http://localhost:5173. The frontend needs `VITE_MAPBOX_TOKEN` in the root
`.env` for the map (copy from `.env.example`).

Check what's live: `curl localhost:8000/api/health`

```json
{"status":"ok","llm_backend":"claude","dense_retrieval":true,
 "tables":{"rentsmart":389121,"str_eligibility":396167,"property_cards":86397}}
```

## Architecture

```
                    ┌─ lookup_property   → exact address → property card
question → agent ───┼─ search_properties → hybrid BM25 + dense over 86k cards
                    ├─ query_database    → read-only DuckDB SQL over 790k rows
                    └─ owner_portfolio   → owner → properties → violations graph
```

The agent picks the tools; nothing is hardcoded to a question pattern.

**Why property cards.** The naive approach is to embed all 790k CSV rows. That fails:
one sanitation row is near-identical in embedding space to 38k others, so retrieval
collapses. Instead `ingest.py` pre-aggregates **one card per address** (86,397 of
them), joining that property's entire RentSmart history to its short-term-rental
eligibility, and renders it as prose. Cards are the unit of semantic retrieval; the
raw tables stay intact for SQL.

**Why both retrievers.** BM25 catches literal tokens a renter types (street names,
owner LLCs, "Rodent Activity"). Dense catches paraphrase ("falling apart" →
"Unsafe Dangerous Conditions"). They are fused with Reciprocal Rank Fusion, because
BM25 scores and cosine similarities are not on comparable scales.

**Why SQL at all.** Retrieval cannot count. "Which neighborhood has the most rat
reports" is an aggregation over 389k rows, and approximating it with top-k retrieval
gives a confidently wrong answer.

## Datasets

| Dataset | Rows | Refresh | Role |
|---|---|---|---|
| [RentSmart](https://data.boston.gov/dataset/rentsmart) | 389,121 | daily | violation/complaint history |
| [Short-Term Rental Eligibility](https://data.boston.gov/dataset/short-term-rental-eligibility) | 396,167 | nightly | open violations, problem-property flags, legality |
| [Income-Restricted Housing](https://data.boston.gov/dataset/income-restricted-housing) | 1,491 | annual | affordability context |

RentSmart and STR join on a normalized address at **100% coverage of distinct
RentSmart addresses** — both derive from the City's SAM address registry.

## Data caveats the agent is told about

Encoded in the system prompt and derived from the data at startup, so they survive a
re-ingest:

- The resource is labelled "2016 – present" but **actually starts 2021-09-17**.
- 2021 and the current year are **partial** — comparing them to a full year invents
  trends that aren't there.
- "Housing Violations" went 651 (2022) → 5,458 (2025). That is a **recording-practice
  change**, not an 8× rise.
- **77% of rows are Enforcement Violations** — mostly trash, recycling and sidewalk
  citations, not housing-condition problems.
- Absence of records ≠ good property. It may simply never have been reported.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required for `RENTWISE_LLM=claude` |
| `RENTWISE_LLM` | `claude` | `claude` \| `local` \| `none` |
| `RENTWISE_CLAUDE_MODEL` | `claude-opus-5` | model id |
| `RENTWISE_LOCAL_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible local server |
| `RENTWISE_LOCAL_MODEL` | `qwen3:4b` | local model id |

Generation is the only cloud step, and it is swappable. Retrieval — DuckDB, BM25 via
the FTS extension, `bge-small-en-v1.5` embeddings via fastembed/ONNX — is entirely
local and open source, with no network calls.

`RENTWISE_LLM=none` runs retrieval only and templates the answer. It exists so a
missing key never blocks a demo; it is not a real answer path.
