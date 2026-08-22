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

# 3. Optional: local embeddings for hybrid search. Defaults to the busiest 20k
#    cards, which fits in memory. Do NOT pass --all on 8 GB: embedding all 86k
#    swaps heavily, takes 1h+, and makes the machine unusable while it runs.
.venv/bin/python -m backend.build_embeddings

# 4. The local model
brew install ollama
ollama pull qwen2.5:3b-instruct   # non-reasoning instruct model; see README.md

# 5. Run all three processes
OLLAMA_CONTEXT_LENGTH=16384 ollama serve                 # terminal 1
.venv/bin/uvicorn backend.app:app --port 8000 --reload   # terminal 2
npm run dev                                              # terminal 3
```

`OLLAMA_CONTEXT_LENGTH` is required, not cosmetic — see the note in `llm.py`. ollama
defaults to 4096 tokens and silently truncates rather than erroring, and what gets
truncated is usually the tool definitions.

Open http://localhost:5173. The frontend needs `VITE_MAPBOX_TOKEN` in the root
`.env` for the map (copy from `.env.example`).

Check what's live: `curl localhost:8000/api/health`

```json
{"status":"ok","llm_backend":"local","dense_retrieval":true,
 "tables":{"rentsmart":389121,"str_eligibility":396167,"property_cards":86397}}
```

## Architecture

```
                  ┌── catalog route (0.1ms, no model) ──┐
question ─────────┤                                     ├──► tool ──► headline (1 gen)
                  └── model chooses tool (agentic) ─────┘

  tools: lookup_property   → exact address → property card
         search_properties → hybrid BM25 + dense over 86k cards
         query_database    → read-only DuckDB SQL over 790k rows
         owner_portfolio   → owner → properties → violations graph
         describe_data     → full schema + value vocabulary, on demand
```

**Two routing paths.** The knowledge catalog (`catalog.py`) indexes every dataset,
field, and the *value vocabularies actually present in the data* — the six violation
types, fifteen neighborhoods, real `description` strings — plus a synonym layer mapping
how residents speak ("rats", "falling apart") onto those values. When a question's
shape is unambiguous, the catalog names the tool and its arguments in ~0.1 ms and the
model's tool-selection turn is skipped entirely. Anything ambiguous defers to the
model, because being wrong is worse than being slow.

That matters because generation, not retrieval, is the cost here: retrieval is
single-digit milliseconds, while a local model at ~26 tok/s makes every generated token
~38 ms. Catalog routing removes a whole turn.

**Progressive disclosure.** Only the headline is generated, and it is capped to three
sentences. The `detail` array (per-property facts) and `citations` are assembled from
catalog-typed data at zero model cost — so they are instant, and immune to paraphrase
drift. The UI shows the headline and reveals detail on demand.

**Query expansion replaces some of what embeddings do.** BM25 cannot bridge "rats" →
"Rodent Activity". The catalog does it explicitly from the data's own vocabulary, which
is why the system is useful even with `dense_retrieval: false`.

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
| `RENTWISE_LLM` | `local` | `local` \| `none` |
| `RENTWISE_LOCAL_MODEL` | `qwen2.5:3b-instruct` | model id as your server names it |
| `RENTWISE_LOCAL_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `RENTWISE_LOCAL_MAX_TOKENS` | `2048` | per-turn generation cap |
| `RENTWISE_LOCAL_TIMEOUT` | `600` | seconds; local generation is slow |
| `OLLAMA_CONTEXT_LENGTH` | — | **set to 16384** on the `ollama serve` process |

`RENTWISE_LLM=none` runs retrieval only and templates the answer. It exists so a
missing model never blocks a demo; it is not a real answer path.

## No cloud, no keys

Every component runs on the machine and is open source:

| Layer | Tool | License |
|---|---|---|
| Storage, BM25, text-to-SQL | DuckDB + `fts` extension | MIT |
| Dense embeddings | `bge-small-en-v1.5` via fastembed / ONNX Runtime | MIT / Apache-2.0 |
| Generation | Qwen2.5-Instruct via ollama | Apache-2.0 / MIT |
| API | FastAPI + uvicorn | MIT / BSD-3 |

There is no hosted-LLM SDK in `requirements.txt` and no API key anywhere in the
project. The only network access is `backend.download` fetching CSVs from
data.boston.gov, and the one-time model pull. After that it runs fully offline.

The frontend's `mapbox-gl` is the one non-open-source dependency; swap it for
[MapLibre GL JS](https://maplibre.org/) (BSD-3) if that matters.

## Known limitations

Measured across 16 question shapes on `qwen2.5:3b-instruct`. **Everything on the
catalog fast path works; the failures are all on the path that defers to the model.**
A 3B model is good enough to write one grounded sentence over evidence handed to it,
and not good enough to select tools and author correct DuckDB SQL across turns.

**Working (9/16)** — property safety, property history, ownership lookup, issue +
neighborhood search, vague-condition search, short-term-rental eligibility,
cross-domain safe-and-sublettable, unknown address (correctly reports not found),
out-of-scope question (correctly declines). All 1–4 s.

**Broken (4/16)** — do not demo these:

| Question shape | Failure |
|---|---|
| "Which neighborhood has the worst rat problem?" | Defers to model, which picks `search_properties` instead of SQL. Answer is ungrounded — says "3 properties" where SQL gives 6,885 pest reports. |
| "How many housing violations in Roxbury?" | Generated SQL fails twice; the model then narrates its retry instead of answering. Zero citations. |
| "Are pest problems getting better or worse?" | Model emits tool-call *text* instead of calling a tool. Zero tools executed. |
| "Where is income-restricted housing in Roxbury?" | Routed to property cards, which do not contain that table. Claims the data is absent when 1,491 projects exist. No tool reaches `income_restricted`. |

**Partial (3/16)** — `owner_portfolio` names one property when the owner has 144;
occasional invented statistics in owner answers; evidence indices (`[1]`, `[2]`) can
leak into prose.

The fix for the first three is to extend the catalog to emit **parameterized SQL from
templates** for the common aggregate shapes (rank-by-neighborhood, count-by-filter,
trend-by-year) rather than asking a 3B model to write it. The fourth needs an
`income_restricted` tool. Neither requires leaving local models.

There is no automated evaluation suite yet; the numbers above come from a manual probe.
