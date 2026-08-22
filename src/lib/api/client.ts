import type { RagResponse } from '../types'
import { askRentWise as askMock } from './mockBackend'

/**
 * Real backend client for POST /api/ask.
 *
 * Vite proxies /api to the Python RAG service (see vite.config.ts), so this is
 * same-origin in dev. Set VITE_USE_MOCK=true in .env to force the mock backend --
 * useful for working on the UI without running the Python side.
 */

/** Citation as returned by the backend: one identifiable Analyze Boston record. */
export interface Citation {
  dataset: string
  dataset_key: string
  record_id: number | string | null
  label: string
  detail: string
  /** Deep link to the row on data.boston.gov. */
  url: string
  updated: string
  license: string
}

/** What the agent actually did — which tools it chose, what SQL it ran. */
export interface AgentTrace {
  backend: string
  turns: number
  tool_calls: { tool: string; arguments: Record<string, unknown>; ms: number }[]
  sql: string[]
  dense_enabled: boolean
  elapsed_ms: number
  usage: Record<string, number>
}

/** The backend returns the frontend contract plus these additive fields. */
export interface RagResponseFull extends RagResponse {
  citations?: Citation[]
  trace?: AgentTrace
}

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'

export async function askRentWise(query: string): Promise<RagResponseFull> {
  if (USE_MOCK) return askMock(query)

  let res: Response
  try {
    res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    })
  } catch {
    throw new Error(
      'Could not reach the RAG backend. Start it with:  uvicorn backend.app:app --port 8000',
    )
  }

  if (!res.ok) {
    // FastAPI reports errors as {detail: string | ValidationError[]}.
    let detail = `Backend returned ${res.status}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: any) => d.msg).join('; ')
    } catch {
      /* keep the status-code message */
    }
    throw new Error(detail)
  }

  return (await res.json()) as RagResponseFull
}
