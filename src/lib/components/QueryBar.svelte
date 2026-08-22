<script lang="ts">
  let {
    loading = false,
    onask,
  }: { loading?: boolean; onask: (query: string) => void } = $props()

  let query = $state('')

  const suggestions = [
    'Where has rodent activity been reported recently?',
    'Show me heat complaints',
    'Any unsafe buildings?',
    "What's happening in Dorchester?",
  ]

  function submit(event: SubmitEvent) {
    event.preventDefault()
    const trimmed = query.trim()
    if (trimmed && !loading) onask(trimmed)
  }

  function useSuggestion(text: string) {
    query = text
    if (!loading) onask(text)
  }
</script>

<form class="query-bar" onsubmit={submit}>
  <div class="input-row">
    <input
      type="text"
      bind:value={query}
      placeholder="Ask about Boston rental housing — violations, complaints, neighborhoods…"
      aria-label="Ask a question about RentSmart data"
      disabled={loading}
    />
    <button type="submit" disabled={loading || !query.trim()}>
      {loading ? 'Thinking…' : 'Ask'}
    </button>
  </div>
  <div class="suggestions">
    {#each suggestions as text}
      <button type="button" class="chip" onclick={() => useSuggestion(text)} disabled={loading}>
        {text}
      </button>
    {/each}
  </div>
</form>

<style>
  .query-bar {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .input-row {
    display: flex;
    gap: 0.5rem;
  }
  input {
    flex: 1;
    font: inherit;
    font-size: 1.05rem;
    padding: 0.8rem 1rem;
    border: 1.5px solid var(--border);
    border-radius: 0.75rem;
    background: var(--surface);
    color: var(--text);
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  button[type='submit'] {
    font: inherit;
    font-size: 1.05rem;
    font-weight: 600;
    padding: 0.8rem 1.5rem;
    border: none;
    border-radius: 0.75rem;
    background: var(--accent);
    color: #fff;
    cursor: pointer;
    transition: background 0.15s;
  }
  button[type='submit']:hover:not(:disabled) {
    background: var(--accent-hover);
  }
  button:disabled {
    opacity: 0.55;
    cursor: default;
  }
  .suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .chip {
    font: inherit;
    font-size: 0.85rem;
    padding: 0.35rem 0.8rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--surface);
    color: var(--text-muted);
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
  }
  .chip:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--accent);
  }
</style>
