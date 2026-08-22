"""The RentWise agentic RAG loop.

The agent is given four retrieval tools and decides for itself which to use. That
choice is the whole point: the three question shapes a renter actually asks need
fundamentally different machinery, and a single retriever answers at most one well.

    "is 44 Portsmouth St safe?"            -> lookup_property   (exact resolution)
    "somewhere in Dorchester with rats"    -> search_properties  (hybrid retrieval)
    "which neighborhood is worst for rats" -> query_database     (aggregation)
    "is my landlord a repeat offender?"    -> owner_portfolio    (graph traversal)

Citations are assembled programmatically from what the tools actually returned, not
from what the model claims it used. The model can misattribute; the trace cannot.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from .llm import LLMBackend, ToolCall, build_backend
from .retrieval import RentWiseIndex, RetrievalTrace, SqlNotAllowed
from .sources import RENTSMART, STR_ELIGIBILITY, SourceRef

log = logging.getLogger(__name__)

MAX_TURNS = 8
MAX_LOCATIONS = 12

SYSTEM_TEMPLATE = """\
You are RentWise Boston, a housing analyst for people deciding whether to rent a home \
in Boston. You answer only from the City of Boston open data available through your \
tools. You are talking to residents, not analysts: be concrete and plain-spoken.

# Data you can reach
1. RentSmart (rentsmart table) — {rentsmart_rows} records, 2016 to present, refreshed \
daily. Housing/building/enforcement violations, housing complaints, sanitation and \
civic maintenance requests. One row per reported issue, with address, neighborhood, \
owner, year built, property type and coordinates.
2. Short-Term Rental Eligibility (str_eligibility table) — {str_rows} unique Boston \
addresses from the City's SAM registry, refreshed nightly. Open violation counts, \
problem-property flags, owner-occupancy, units in building, and whether the unit may \
be listed as a short-term rental.
3. Income-Restricted Housing Inventory (income_restricted table) — {income_rows} \
income-restricted projects, updated annually.
4. property_cards / property_docs — one pre-aggregated row per distinct address \
({card_rows} properties), joining a property's full RentSmart history to its \
short-term-rental eligibility. This is what search_properties and lookup_property read.
5. neighborhood_stats — per-neighborhood rollups, including pest/heat/unsafe counts.

# Database schema (for query_database)
{schema}

# How to choose a tool
- A specific street address in the question -> lookup_property.
- Counting, ranking, comparing, trends over time, "most/least/average" -> query_database.
- A described problem with no specific address ("somewhere with rats", "unsafe \
building") -> search_properties.
- Questions about a landlord or owner across multiple buildings -> owner_portfolio.
- Combine tools when a question needs it. A question like "is this building safe and \
can I sublet it" needs lookup_property for the history and the eligibility flags.

# Rules that matter
- Ground every number and claim in a tool result. Never estimate or recall from \
training data.
- If the tools do not support an answer, say plainly that the data does not show it. \
"I don't know" is a correct answer; a plausible guess is not.
- Quote real addresses, real dates and real counts from the results.
- RentSmart records are *reports*, not convictions. A complaint means someone \
reported an issue, not that it was upheld. Say so when it matters.
- A property with no records is not necessarily problem-free — it may simply never \
have been reported. Do not present absence of records as proof of quality.
- Enforcement Violations dominate the dataset ({enforcement_share}% of rows) and are \
mostly trash, recycling and sidewalk citations against the property. Do not present \
them as housing-condition problems.
- Today is {today}. "Recent" means the last 12 months unless the user says otherwise.
- Be concise: 2-5 sentences for simple questions. No preamble, no bullet lists unless \
comparing several things.

# Temporal traps in this data — read before answering any trend question
The resource is labelled "2016 - present" but actually begins {coverage_start}. \
Coverage runs to {coverage_end}.
- {first_year} and {last_year} are PARTIAL years. Never compare a partial year against \
a full one and call the difference a trend. Either restrict to whole years \
({full_year_start}-{full_year_end}), or compare the same months across years.
- "Housing Violations" rose from {hv_early} records in {hv_early_year} to \
{hv_late} in {hv_late_year}. That is a change in how the City recorded them, not a \
{hv_multiple}x rise in violations. Do not report it as one.
- Counts grew across every category over this window, partly because reporting \
expanded. Prefer per-property rates over raw counts when comparing places or periods, \
and say which you used.
"""

TOOLS = [
    {
        "name": "search_properties",
        "description": (
            "Hybrid search (BM25 + dense vectors) over pre-aggregated property cards. "
            "Use for described problems without a specific address, e.g. 'buildings "
            "with rat problems in Dorchester' or 'unsafe conditions in old triple "
            "deckers'. Returns each property's full violation history and rental "
            "eligibility. Not for counting — use query_database for that."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Descriptive search text. Include the neighborhood if the user named one.",
                },
                "k": {
                    "type": "integer",
                    "description": "How many properties to return (1-20, default 8).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "lookup_property",
        "description": (
            "Resolve a specific street address to its property card: full RentSmart "
            "history plus short-term-rental eligibility. Use whenever the user names "
            "an address. Accepts messy input ('44 Portsmouth', '44 Portsmouth St, 02135')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Street address as the user wrote it."}
            },
            "required": ["address"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_database",
        "description": (
            "Run a read-only DuckDB SELECT for counting, ranking, comparing or "
            "time-series questions. This is the only correct way to answer aggregate "
            "questions — retrieval cannot count. Select latitude/longitude alongside "
            "address when the result should appear on the map."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single read-only SELECT or WITH statement. No semicolons.",
                },
                "purpose": {
                    "type": "string",
                    "description": "One line on what this query is meant to establish.",
                },
            },
            "required": ["sql", "purpose"],
            "additionalProperties": False,
        },
    },
    {
        "name": "owner_portfolio",
        "description": (
            "Traverse the owner graph: given an owner or landlord name, return every "
            "other Boston property they own in the data, with each property's violation "
            "counts. Use for 'is my landlord a repeat offender', 'what else does this "
            "company own', or to establish whether a problem is building-specific or "
            "portfolio-wide. Accepts partial names ('GBM PORTFOLIO')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Owner name or fragment."},
                "k": {"type": "integer", "description": "Max properties to return (default 15)."},
            },
            "required": ["owner"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class AgentResult:
    answer: str
    locations: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)

    def to_response(self) -> dict:
        """The exact shape src/lib/types.ts expects, plus additive debug fields."""
        return {
            "answer": self.answer,
            "locations": self.locations,
            "sources": self.sources,
            "citations": self.citations,
            "trace": self.trace,
        }


class RentWiseAgent:
    def __init__(self, index: RentWiseIndex, backend: LLMBackend | None = None) -> None:
        self.ix = index
        self.backend = backend or build_backend()
        self._system = self._build_system()

    def _build_system(self) -> str:
        con = self.ix.con
        rentsmart_rows = con.execute("SELECT count(*) FROM rentsmart").fetchone()[0]
        str_rows = con.execute("SELECT count(*) FROM str_eligibility").fetchone()[0]
        card_rows = con.execute("SELECT count(*) FROM property_cards").fetchone()[0]
        try:
            income_rows = con.execute("SELECT count(*) FROM income_restricted").fetchone()[0]
        except Exception:  # noqa: BLE001 - table is optional
            income_rows = 0
        enforcement = con.execute(
            "SELECT count(*) FROM rentsmart WHERE violation_type = 'Enforcement Violations'"
        ).fetchone()[0]

        # Derive the temporal caveats from the data rather than hardcoding them, so
        # they stay true after a daily re-ingest.
        start, end = con.execute(
            "SELECT min(occurred_at), max(occurred_at) FROM rentsmart"
        ).fetchone()
        first_year, last_year = start.year, end.year
        hv = con.execute(
            """
            SELECT year(occurred_at) AS yr, count(*) AS n
            FROM rentsmart WHERE violation_type = 'Housing Violations'
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
        # Compare the first and last *complete* years of the housing-violation series.
        complete = [(y, n) for y, n in hv if first_year < y < last_year] or hv
        hv_early_year, hv_early = complete[0]
        hv_late_year, hv_late = complete[-1]

        return SYSTEM_TEMPLATE.format(
            rentsmart_rows=f"{rentsmart_rows:,}",
            str_rows=f"{str_rows:,}",
            income_rows=f"{income_rows:,}",
            card_rows=f"{card_rows:,}",
            schema=self.ix.schema_prompt(),
            enforcement_share=round(100 * enforcement / max(rentsmart_rows, 1)),
            today=time.strftime("%B %d, %Y"),
            coverage_start=start.strftime("%B %d, %Y"),
            coverage_end=end.strftime("%B %d, %Y"),
            first_year=first_year,
            last_year=last_year,
            full_year_start=first_year + 1,
            full_year_end=last_year - 1,
            hv_early=f"{hv_early:,}",
            hv_early_year=hv_early_year,
            hv_late=f"{hv_late:,}",
            hv_late_year=hv_late_year,
            hv_multiple=round(hv_late / max(hv_early, 1)),
        )

    # ---------- tool execution ----------

    def _tool_search(self, args: dict, trace: RetrievalTrace) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required."
        k = max(1, min(int(args.get("k") or 8), 20))
        hits = self.ix.search_hybrid(query, k)
        if not hits:
            return f"No properties matched {query!r}."
        trace.hits.extend(hits)
        for hit in hits:
            trace.add_citations(hit.citations())
        retrievers = {h.retriever for h in hits}
        header = (
            f"{len(hits)} properties (retrieval: {'+'.join(sorted(retrievers))}; "
            f"dense {'on' if self.ix.dense_enabled else 'OFF — BM25 only'})"
        )
        body = "\n\n".join(f"[{i + 1}] {h.card_text}" for i, h in enumerate(hits))
        return f"{header}\n\n{body}"

    def _tool_lookup(self, args: dict, trace: RetrievalTrace) -> str:
        address = (args.get("address") or "").strip()
        if not address:
            return "Error: 'address' is required."
        hits = self.ix.lookup_property(address)
        if not hits:
            return (
                f"No property matching {address!r} in RentSmart. Either the address has "
                "never been the subject of a report, or it is spelled differently in the "
                "City's records. Absence of records is not evidence of good condition."
            )
        trace.hits.extend(hits)
        for hit in hits:
            trace.add_citations(hit.citations())
        return "\n\n".join(f"[{i + 1}] {h.card_text}" for i, h in enumerate(hits))

    def _tool_sql(self, args: dict, trace: RetrievalTrace) -> str:
        sql = (args.get("sql") or "").strip()
        if not sql:
            return "Error: 'sql' is required."
        try:
            result = self.ix.run_sql(sql)
        except SqlNotAllowed as exc:
            return f"Query rejected: {exc}. Rewrite it as a single read-only SELECT."
        except Exception as exc:  # noqa: BLE001 - hand the error back to the model
            return f"SQL error: {exc}. Check column names against the schema and retry."

        trace.sql_queries.append(result.sql)
        self._locations_from_sql(result, trace)
        # A SQL aggregate is grounded in the table, not in one row, so cite the dataset.
        trace.add_citations(
            [
                SourceRef(
                    dataset_key=RENTSMART.key,
                    record_id=None,
                    label=args.get("purpose") or "aggregate query",
                    detail=result.sql[:200],
                )
            ]
        )
        return result.to_markdown()

    def _tool_owner(self, args: dict, trace: RetrievalTrace) -> str:
        owner = (args.get("owner") or "").strip()
        if not owner:
            return "Error: 'owner' is required."
        k = max(1, min(int(args.get("k") or 15), 50))
        rows = self.ix.owner_portfolio(owner, k)
        if not rows:
            return f"No owner matching {owner!r} in the RentSmart owner field."
        trace.add_citations(
            [
                SourceRef(
                    dataset_key=RENTSMART.key,
                    record_id=None,
                    label=f"owner portfolio: {owner}",
                    detail=f"{len(rows)} properties",
                )
            ]
        )
        lines = [
            f"- {r['address']} ({r['neighborhood']}): {r['total_records']} records, "
            f"{r['housing_violations']} housing violations, "
            f"{r['open_violations'] or 0} open STR violations"
            for r in rows
        ]
        total = sum(r["total_records"] for r in rows)
        return (
            f"Owner match: {rows[0]['owner']}\n"
            f"{len(rows)} properties in the data, {total} RentSmart records combined.\n"
            + "\n".join(lines)
        )

    def _dispatch(self, call: ToolCall, trace: RetrievalTrace) -> str:
        handlers = {
            "search_properties": self._tool_search,
            "lookup_property": self._tool_lookup,
            "query_database": self._tool_sql,
            "owner_portfolio": self._tool_owner,
        }
        handler = handlers.get(call.name)
        if handler is None:
            return f"Unknown tool {call.name!r}."
        started = time.time()
        try:
            output = handler(call.arguments, trace)
        except Exception as exc:  # noqa: BLE001 - a tool crash should not kill the turn
            log.exception("tool %s failed", call.name)
            output = f"Tool error: {exc}"
        trace.tool_calls.append(
            {
                "tool": call.name,
                "arguments": call.arguments,
                "ms": round(1000 * (time.time() - started)),
                "output_chars": len(output),
            }
        )
        return output

    # ---------- map locations ----------

    def _locations_from_sql(self, result, trace: RetrievalTrace) -> None:
        """Pull map pins out of a SQL result when it selected coordinates."""
        cols = {c.lower(): i for i, c in enumerate(result.columns)}
        lat_i, lon_i = cols.get("latitude"), cols.get("longitude")
        addr_i = cols.get("address")
        if lat_i is None or lon_i is None or addr_i is None:
            return
        label_i = cols.get("description") or cols.get("violation_type")
        hood_i = cols.get("neighborhood")
        for row in result.rows[:MAX_LOCATIONS]:
            try:
                lat, lon = float(row[lat_i]), float(row[lon_i])
            except (TypeError, ValueError):
                continue
            trace.pins.append(
                {
                    "address": str(row[addr_i]),
                    "neighborhood": str(row[hood_i]) if hood_i is not None else "",
                    "latitude": lat,
                    "longitude": lon,
                    "label": str(row[label_i]) if label_i is not None else "RentSmart record",
                    "details": "",
                }
            )

    def _build_locations(self, trace: RetrievalTrace) -> list[dict]:
        """Merge retrieved properties and SQL-derived pins into the map payload."""
        out: list[dict] = []
        seen: set[str] = set()

        for hit in trace.hits:
            if hit.latitude is None or hit.longitude is None or not hit.address:
                continue
            if hit.address in seen:
                continue
            seen.add(hit.address)
            details = []
            if hit.total_records:
                details.append(f"{hit.total_records} RentSmart records")
            if hit.property_type:
                details.append(hit.property_type)
            if hit.year_built:
                details.append(f"built {hit.year_built}")
            out.append(
                {
                    "address": hit.address,
                    "neighborhood": hit.neighborhood or "",
                    "latitude": hit.latitude,
                    "longitude": hit.longitude,
                    "label": hit.top_issue or "RentSmart record",
                    "details": " · ".join(details),
                }
            )

        for pin in trace.pins:
            if pin["address"] in seen:
                continue
            seen.add(pin["address"])
            out.append(pin)

        return out[:MAX_LOCATIONS]

    # ---------- the loop ----------

    def ask(self, question: str) -> AgentResult:
        trace = RetrievalTrace()
        started = time.time()
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0}

        messages = self.backend.start(self._system, question, TOOLS)
        answer = ""
        turns = 0

        for turns in range(1, MAX_TURNS + 1):
            turn = self.backend.step(self._system, messages, TOOLS)
            for key, value in turn.usage.items():
                usage[key] = usage.get(key, 0) + value

            if not turn.wants_tools:
                answer = turn.text.strip()
                break

            self.backend.append_turn(messages, turn)
            results = [(call, self._dispatch(call, trace)) for call in turn.tool_calls]
            self.backend.append_results(messages, results)
        else:
            # Ran out of turns while still calling tools. Report what we have rather
            # than pretending the answer is complete.
            answer = (
                "I gathered data but ran out of reasoning steps before finishing. "
                "The retrieved records are cited below."
            )

        if not answer:
            answer = self._fallback_answer(question, trace)

        return AgentResult(
            answer=answer,
            locations=self._build_locations(trace),
            sources=[c.to_line() for c in trace.citations],
            citations=[c.to_dict() for c in trace.citations],
            trace={
                "backend": self.backend.name,
                "turns": turns,
                "tool_calls": trace.tool_calls,
                "sql": trace.sql_queries,
                "dense_enabled": self.ix.dense_enabled,
                "elapsed_ms": round(1000 * (time.time() - started)),
                "usage": usage,
            },
        )

    def _fallback_answer(self, question: str, trace: RetrievalTrace) -> str:
        """Retrieval-only answer, used when no model is configured.

        Deliberately does not attempt to sound like an analysis -- it reports what was
        retrieved and says why it is not a real answer.
        """
        hits = self.ix.search_hybrid(question, 5)
        trace.hits.extend(hits)
        for hit in hits:
            trace.add_citations(hit.citations())
        if not hits:
            return (
                "No language model is configured, so I can only run retrieval — and "
                "retrieval found nothing for this question. Set ANTHROPIC_API_KEY in "
                "backend/.env for real answers."
            )
        lines = [
            f"• {h.address} ({h.neighborhood}) — {h.total_records} records, "
            f"issues include {h.top_issue or 'n/a'}"
            for h in hits
        ]
        return (
            "No language model is configured, so this is raw retrieval output rather "
            "than a written answer. The most relevant properties were:\n\n"
            + "\n".join(lines)
            + "\n\nSet ANTHROPIC_API_KEY in backend/.env (or RENTWISE_LLM=local) to get "
            "grounded prose answers."
        )
