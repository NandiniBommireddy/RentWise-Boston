# RentWise-Boston

Project Description RentWise Boston is a mobile-friendly website with an integrated AI chatbot that helps people make informed decisions before renting a home in Boston.

## Front end

A single-page app (Svelte 5 + Vite + TypeScript) where you ask a question about Boston's [RentSmart](https://data.boston.gov/dataset/rentsmart) housing data and get back an LLM-generated answer alongside a Mapbox map pinning the locations it references.

The backend is currently mocked in [`src/lib/api/mockBackend.ts`](src/lib/api/mockBackend.ts) — it keyword-matches a few scenarios (rodents/pests, heat complaints, unsafe buildings, Dorchester) with real rows from the RentSmart dataset, and defines the API contract the real RAG backend should implement (`POST /api/ask` → `{ answer, locations, sources }`, see [`src/lib/types.ts`](src/lib/types.ts)).

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
