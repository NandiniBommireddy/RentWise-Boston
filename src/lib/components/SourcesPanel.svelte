<script lang="ts">
  import type { RagResponse } from '../types'

  let { response }: { response: RagResponse } = $props()
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

  {#if response.sources?.length}
    <div class="grounding">
      Grounded in: {response.sources.join(' · ')}
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
    font-size: 0.78rem;
    color: var(--text-muted);
  }
</style>
