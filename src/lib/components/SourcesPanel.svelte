<script lang="ts">
  import type { RagResponse } from '../types'
  import type { Citation } from '../api/client'

  let { response, citations = [] }: { response: RagResponse; citations?: Citation[] } = $props()

  // Short dataset tag for each citation row, e.g. "RentSmart Boston (2016–present)"
  // -> "RentSmart". Falls back to the full dataset title.
  function datasetTag(citation: Citation): string {
    return citation.dataset.split(/[( ]/, 1)[0] || citation.dataset
  }
</script>

<div class="sources-panel">
  {#if response.locations.length > 0}
    <h3>Locations referenced ({response.locations.length})</h3>
    <ul class="locations">
      {#each response.locations as loc}
        <li>
          <span class="dot" aria-hidden="true"></span>
          <div>
            <strong>{loc.address}</strong>
            <span class="hood">· {loc.neighborhood}</span>
            <div class="loc-label">{loc.details ? `${loc.label} — ${loc.details}` : loc.label}</div>
          </div>
        </li>
      {/each}
    </ul>
  {:else}
    <p class="empty">No specific locations for this answer — try asking about an issue or neighborhood.</p>
  {/if}

  {#if citations.length > 0}
    <div class="grounding">
      <h3>Grounded in {citations.length} records</h3>
      <ul class="citations">
        {#each citations as citation}
          <li>
            <a href={citation.url} target="_blank" rel="noopener noreferrer" title={citation.detail}>
              {citation.label}
            </a>
            <span class="dataset-tag">{datasetTag(citation)}</span>
          </li>
        {/each}
      </ul>
    </div>
  {:else if response.sources?.length}
    <div class="grounding">
      <h3>Grounded in</h3>
      <p class="sources-text">{response.sources.join(' · ')}</p>
    </div>
  {/if}
</div>

<style>
  .sources-panel {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    height: 100%;
    min-height: 0;
  }
  h3 {
    margin: 0;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
  }
  .locations {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
    overflow-y: auto;
    flex: 1;
    min-height: 0;
  }
  .locations li {
    display: flex;
    gap: 0.6rem;
    align-items: baseline;
    font-size: 0.92rem;
  }
  .dot {
    flex: none;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--accent);
    position: relative;
    top: -1px;
  }
  .hood {
    color: var(--text-muted);
  }
  .loc-label {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 0.1rem;
  }
  .empty {
    margin: 0;
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.5;
  }
  .grounding {
    border-top: 1px solid var(--border);
    padding-top: 0.65rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-height: 40%;
    min-height: 0;
  }
  .citations {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    overflow-y: auto;
    min-height: 0;
  }
  .citations li {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    font-size: 0.8rem;
    line-height: 1.45;
  }
  .citations a {
    color: var(--accent);
    text-decoration: none;
    min-width: 0;
  }
  .citations a:hover {
    text-decoration: underline;
  }
  .dataset-tag {
    flex: none;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.05rem 0.5rem;
    white-space: nowrap;
  }
  .sources-text {
    margin: 0;
    font-size: 0.8rem;
    color: var(--text-muted);
  }
</style>
