<script lang="ts">
  import QueryBar from './lib/components/QueryBar.svelte'
  import AnswerPanel from './lib/components/AnswerPanel.svelte'
  import SourcesPanel from './lib/components/SourcesPanel.svelte'
  import MapPanel from './lib/components/MapPanel.svelte'
  import { askRentWise, fetchHealth, type Citation, type HealthInfo, type RagResponseFull } from './lib/api/client'
  import type { AskState } from './lib/types'

  let ask = $state<AskState>({ status: 'idle', query: '', response: null, error: null })
  let health = $state<HealthInfo | null>(null)

  $effect(() => {
    fetchHealth().then((h) => (health = h))
  })

  const citations: Citation[] = $derived(
    ask.response ? ((ask.response as RagResponseFull).citations ?? []) : []
  )

  async function handleAsk(query: string) {
    ask = { status: 'loading', query, response: null, error: null }
    try {
      const response = await askRentWise(query)
      ask = { status: 'done', query, response, error: null }
    } catch (e) {
      ask = { status: 'error', query, response: null, error: e instanceof Error ? e.message : 'Something went wrong.' }
    }
  }
</script>

<div class="page">
  <header>
    <h1>RentWise Boston</h1>
    <p class="tagline">
      Ask questions about Boston's RentSmart housing data — violations, complaints, and
      sanitation reports, mapped.
    </p>
    {#if health?.model}
      <p class="powered">
        Powered by {health.model} with
        {health.embeddings > 0
          ? `${health.embeddings.toLocaleString('en-US')} embeddings`
          : 'BM25 search'} in DuckDB
      </p>
    {/if}
  </header>

  <QueryBar loading={ask.status === 'loading'} onask={handleAsk} />

  <main>
    {#if ask.status === 'idle'}
      <div class="placeholder">
        <p>Ask a question above to explore 389,569 RentSmart records across Boston.</p>
      </div>
    {:else if ask.status === 'loading'}
      <div class="placeholder loading" role="status" aria-live="polite">
        <div class="spinner" aria-hidden="true"></div>
        <p>Searching RentSmart records for “{ask.query}”…</p>
      </div>
    {:else if ask.status === 'error'}
      <div class="placeholder error" role="alert">
        <p>{ask.error}</p>
      </div>
    {:else if ask.response}
      <div class="results">
        <section class="answer-row card" aria-label="Answer">
          <AnswerPanel answer={ask.response.answer} />
        </section>
        <div class="bottom">
          <section class="sources-col card" aria-label="Sources">
            <SourcesPanel response={ask.response} {citations} />
          </section>
          <section class="map-col" aria-label="Map of locations">
            <MapPanel locations={ask.response.locations} />
          </section>
        </div>
      </div>
    {/if}
  </main>

  <footer>
    Data: City of Boston RentSmart · Responses are generated and may contain errors —
    verify against official records.
  </footer>
</div>

<style>
  .page {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem 1.5rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    height: 100vh;
  }
  header {
    text-align: left;
  }
  h1 {
    margin: 0;
    font-size: 1.7rem;
    letter-spacing: -0.02em;
  }
  .tagline {
    margin: 0.35rem 0 0;
    color: var(--text-muted);
    line-height: 1.5;
    font-size: 0.95rem;
  }
  .powered {
    margin: 0.25rem 0 0;
    color: var(--text-muted);
    font-style: italic;
    font-size: 0.82rem;
  }
  main {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    flex: 1;
    border: 1.5px dashed var(--border);
    border-radius: 1rem;
    color: var(--text-muted);
    text-align: center;
    padding: 2rem;
  }
  .placeholder.error {
    border-color: #dc2626;
    color: #dc2626;
  }
  .spinner {
    width: 28px;
    height: 28px;
    border: 3px solid var(--accent-soft);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  .results {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 1rem;
    padding: 1.25rem 1.5rem;
  }
  .answer-row {
    flex: none;
    max-height: 32vh;
    overflow-y: auto;
  }
  .bottom {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(300px, 2fr) minmax(0, 3fr);
    gap: 1rem;
  }
  .sources-col {
    min-height: 0;
    overflow: hidden;
  }
  .map-col {
    min-height: 0;
  }
  @media (max-width: 860px) {
    .page {
      height: auto;
      min-height: 100vh;
    }
    .answer-row {
      max-height: none;
    }
    .bottom {
      grid-template-columns: 1fr;
    }
    .map-col {
      height: 400px;
    }
  }
  footer {
    flex: none;
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-muted);
    padding: 0.25rem 0;
  }
</style>
