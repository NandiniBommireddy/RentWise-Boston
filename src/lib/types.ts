/** A single mapped location returned by the RAG backend. */
export interface RagLocation {
  address: string
  neighborhood: string
  latitude: number
  longitude: number
  /** Short label for the map popup title, e.g. "Rodent Activity" */
  label: string
  /** Extra popup detail, e.g. date, owner, property type */
  details?: string
}

/** Response contract for POST /api/ask — the mock and the real backend both return this shape. */
export interface RagResponse {
  /** Natural-language answer from the LLM. */
  answer: string
  /** Locations referenced by the answer; may be empty for dataset-wide answers. */
  locations: RagLocation[]
  /** Which records/fields grounded the answer (for a future "sources" UI). */
  sources?: string[]
}

export interface AskState {
  status: 'idle' | 'loading' | 'done' | 'error'
  query: string
  response: RagResponse | null
  error: string | null
}
