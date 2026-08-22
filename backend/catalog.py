"""The Knowledge Catalog: a machine-readable description of what RentWise knows.

Most RAG systems paste their schema into the system prompt and hope the model reasons
its way to the right query. That costs tokens on every single turn and, on a 4B local
model generating at ~26 tok/s, it costs *seconds*. Worse, the model still has to spend
a whole turn deciding which tool to call before any data is touched.

The catalog inverts that. It is a structured index of:

  * every dataset, its fields, grain, refresh cadence and provenance;
  * the **value vocabularies** actually present in the data -- the six violation types,
    the fifteen neighborhoods, the real `description` strings -- loaded from the
    database rather than hardcoded, so they cannot drift;
  * a synonym layer mapping how residents speak ("rats", "freezing", "falling apart")
    onto those real values.

Two payoffs:

1. **Deterministic routing.** `route()` matches a question against the catalog and, when
   the shape is unambiguous, names the tool and its arguments outright. That skips the
   model's tool-selection turn entirely -- roughly half the end-to-end latency.
2. **Cheap grounding.** Resolved catalog entities become literal SQL/search terms, so
   the model never has to guess that "rats" is stored as "Rodent Activity".

The catalog is also introspectable, which is what makes progressive disclosure honest:
the detail tier is assembled from catalog-typed facts, not from model prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property

import duckdb

from .sources import DATASETS, INCOME_RESTRICTED, RENTSMART, STR_ELIGIBILITY

# How residents phrase things -> the literal substrings that appear in the data.
# Values are matched with ILIKE against rentsmart.description.
ISSUE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "rodent": ("rodent", "rat", "mice", "mouse", "pest", "vermin", "infestation"),
    "heat": ("heat", "heating", "furnace", "boiler", "no hot water"),
    "unsafe": (
        "unsafe", "dangerous", "structur", "collapse", "egress", "falling apart",
        "crumbling", "disrepair", "derelict", "hazard",
    ),
    "trash": ("trash", "rubbish", "garbage", "dumping", "barrel", "recycl", "litter"),
    "plumbing": ("plumbing", "water", "leak", "sewer", "drain"),
    "electrical": ("electric", "wiring", "outlet"),
    "mold": ("mold", "mildew", "damp"),
    "noise": ("noise", "loud"),
    "parking": ("vehicle", "parking", "abandoned vehicle"),
    "weeds": ("weed", "overgrown", "grass"),
    "snow": ("snow", "ice", "sidewalk"),
}

# Question shapes that need aggregation rather than retrieval.
AGGREGATE_CUES = (
    "how many", "how much", "which neighborhood", "what neighborhood", "worst",
    "best", "most", "least", "highest", "lowest", "average", "count", "total",
    "compare", "rank", "top ", "trend", "over time", "by year", "increase",
    "decrease", "getting better", "getting worse", "more than", "percent",
)

# Question shapes about a landlord rather than a building.
OWNER_CUES = (
    "landlord", "owner", "own ", "owns", "portfolio", "company", "llc",
    "repeat offender", "who owns", "management",
)

# Question shapes about short-term rental legality.
STR_CUES = (
    "airbnb", "short-term", "short term", "sublet", "sublease", "vrbo",
    "home-share", "home share", "list my", "rent out", "str ", "eligible",
)

ADDRESS_RE = re.compile(
    r"\b\d{1,5}[a-z]?(?:\s*-\s*\d{1,5}[a-z]?)?\s+"
    r"[a-z0-9'.\- ]{2,40}?\s"
    r"(st|street|ave|avenue|rd|road|dr|drive|ln|lane|blvd|boulevard|ct|court|"
    r"pl|place|ter|terrace|way|sq|square|row|park|hwy|circle|cir)\b\.?",
    re.IGNORECASE,
)


@dataclass
class FieldSpec:
    name: str
    type: str
    description: str = ""


@dataclass
class DatasetSpec:
    key: str
    table: str
    grain: str
    purpose: str
    fields: list[FieldSpec] = field(default_factory=list)

    @property
    def title(self) -> str:
        return DATASETS[self.key].title if self.key in DATASETS else self.key


@dataclass
class Route:
    """A routing decision made without calling the model."""

    tool: str
    arguments: dict
    reason: str
    confident: bool

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "reason": self.reason,
            "confident": self.confident,
        }


class KnowledgeCatalog:
    """Catalog of datasets, fields and the value vocabularies present in the data."""

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con = con

    # ---------- dataset layer ----------

    @cached_property
    def datasets(self) -> list[DatasetSpec]:
        specs = [
            DatasetSpec(
                key=RENTSMART.key,
                table="rentsmart",
                grain="one row per reported issue",
                purpose="violation, complaint and sanitation history for a property",
            ),
            DatasetSpec(
                key=STR_ELIGIBILITY.key,
                table="str_eligibility",
                grain="one row per SAM address",
                purpose="whether a unit may be listed short-term, plus open violation counts",
            ),
            DatasetSpec(
                key=INCOME_RESTRICTED.key,
                table="income_restricted",
                grain="one row per income-restricted project",
                purpose="affordability context",
            ),
        ]
        for spec in specs:
            spec.fields = [
                FieldSpec(name=c, type=t)
                for c, t in self.con.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = ? ORDER BY ordinal_position",
                    [spec.table],
                ).fetchall()
            ]
        return [s for s in specs if s.fields]

    # ---------- value vocabularies, read from the data ----------

    @cached_property
    def violation_types(self) -> list[tuple[str, int]]:
        return self.con.execute(
            "SELECT violation_type, count(*) c FROM rentsmart "
            "WHERE violation_type IS NOT NULL GROUP BY 1 ORDER BY c DESC"
        ).fetchall()

    @cached_property
    def neighborhoods(self) -> list[tuple[str, int]]:
        return self.con.execute(
            "SELECT neighborhood, count(*) c FROM rentsmart "
            "WHERE neighborhood IS NOT NULL GROUP BY 1 ORDER BY c DESC"
        ).fetchall()

    @cached_property
    def descriptions(self) -> list[tuple[str, int]]:
        return self.con.execute(
            "SELECT description, count(*) c FROM rentsmart "
            "WHERE description IS NOT NULL GROUP BY 1 ORDER BY c DESC LIMIT 60"
        ).fetchall()

    @cached_property
    def coverage(self) -> dict:
        start, end = self.con.execute(
            "SELECT min(occurred_at), max(occurred_at) FROM rentsmart"
        ).fetchone()
        return {"start": start, "end": end}

    # ---------- entity resolution ----------

    def find_neighborhood(self, question: str) -> str | None:
        """Longest catalog neighborhood name contained in the question."""
        low = question.lower()
        best = None
        for name, _ in self.neighborhoods:
            if name and name.lower() in low:
                if best is None or len(name) > len(best):
                    best = name
        return best

    def find_address(self, question: str) -> str | None:
        match = ADDRESS_RE.search(question)
        return match.group(0).strip() if match else None

    def find_issue_terms(self, question: str) -> list[str]:
        """Map the resident's phrasing onto substrings that exist in `description`."""
        low = question.lower()
        terms: list[str] = []
        for _, variants in ISSUE_SYNONYMS.items():
            if any(v in low for v in variants):
                terms.extend(variants)
        # De-duplicate, preserving order.
        return list(dict.fromkeys(terms))

    def find_owner(self, question: str) -> str | None:
        """Pull a probable owner/company name out of the question.

        Token-walking rather than one regex: a lazy pattern anchored on the corporate
        suffix starts at the earliest capital in the sentence, so "What else does GBM
        Portfolio Owner LLC own?" yields the whole question rather than the company.
        Instead, find the suffix and walk left across capitalised tokens only, stopping
        at the first lowercase word or question word.
        """
        # Question and function words only. "Owner", "Company" and "Trust" are
        # deliberately absent: they are part of real registered names in this data
        # ("GBM PORTFOLIO OWNER LLC"), so stopping on them truncates the match.
        stop = {
            "what", "who", "which", "does", "do", "is", "are", "was", "else", "other",
            "own", "owns", "owned", "the", "my", "a", "an", "about", "tell", "me",
            "show", "and", "any", "all", "more", "also", "still", "have", "has",
        }
        suffixes = {"llc", "l.l.c.", "lp", "inc", "inc.", "corp", "corp.", "trust", "company", "co", "co."}

        tokens = re.findall(r"[A-Za-z0-9&'.\-]+", question)
        for i, tok in enumerate(tokens):
            if tok.lower().strip(".,") in suffixes:
                start = i
                while start > 0:
                    prev = tokens[start - 1]
                    if prev.lower() in stop or not prev[:1].isupper():
                        break
                    start -= 1
                if start < i:
                    return " ".join(tokens[start : i + 1])

        # No corporate suffix: take capitalised tokens following an ownership phrase.
        if m := re.search(r"(?:owned by|belongs to|who owns)\s+(.{2,60})", question, re.IGNORECASE):
            tail = re.findall(r"[A-Za-z0-9&'.\-]+", m.group(1))
            name = [t for t in tail if t[:1].isupper() and t.lower() not in stop]
            if name:
                return " ".join(name)
        return None

    # ---------- deterministic routing ----------

    def route(self, question: str) -> Route:
        """Choose a tool from the question's shape, without calling the model.

        `confident=False` means the agent should let the model decide instead. Being
        wrong here is worse than being slow, so anything ambiguous defers.
        """
        low = question.lower()
        address = self.find_address(question)
        neighborhood = self.find_neighborhood(question)
        issues = self.find_issue_terms(question)
        aggregate = any(cue in low for cue in AGGREGATE_CUES)
        owner_ish = any(cue in low for cue in OWNER_CUES)
        str_ish = any(cue in low for cue in STR_CUES)

        owner = self.find_owner(question) if owner_ish else None

        # A named company with no address is a portfolio question: "what else does
        # GBM Portfolio Owner LLC own". Requires the corporate suffix, so it does not
        # fire on "who owns 1 Rosa St".
        if owner and not address and not aggregate:
            return Route(
                "owner_portfolio",
                {"owner": owner},
                f"named owner with no address ({owner!r})",
                True,
            )

        # A concrete address dominates everything else, including "who owns X" -- the
        # property card already carries the owner, so the lookup answers it directly.
        if address and not aggregate:
            return Route(
                "lookup_property",
                {"address": address},
                f"question contains a street address ({address!r})",
                True,
            )

        # Counting, ranking and trends must go to SQL -- retrieval cannot count.
        if aggregate:
            return Route(
                "query_database",
                {},
                "aggregate or comparative phrasing; needs SQL, not retrieval",
                False,  # the model still has to write the SQL
            )

        # A described problem, optionally scoped to a neighborhood.
        if issues or neighborhood:
            query = " ".join(filter(None, [neighborhood, *issues[:6]])) or question
            return Route(
                "search_properties",
                {"query": query, "k": 6},
                "described problem resolved through the catalog vocabulary",
                True,
            )

        return Route("", {}, "no confident catalog match", False)

    # ---------- prompt rendering ----------

    def compact_prompt(self) -> str:
        """A deliberately small catalog summary for the system prompt.

        Full field lists live in the `describe_data` tool instead: on a local model the
        schema costs prefill time on every turn, and most questions never need it.
        """
        types = ", ".join(f"{t} ({c:,})" for t, c in self.violation_types)
        hoods = ", ".join(n for n, _ in self.neighborhoods)
        cov = self.coverage
        return (
            f"Coverage: {cov['start']:%b %Y} to {cov['end']:%b %Y}.\n"
            f"violation_type values: {types}.\n"
            f"neighborhood values: {hoods}."
        )

    def schema_prompt(self) -> str:
        """Full field listing -- returned by the describe_data tool on demand."""
        lines = []
        for spec in self.datasets:
            cols = ", ".join(f"{f.name} {f.type}" for f in spec.fields)
            lines.append(f"{spec.table} -- {spec.grain}; {spec.purpose}\n  {cols}")
        return "\n".join(lines)

    def vocabulary_prompt(self) -> str:
        """The most common `description` values, for writing precise SQL filters."""
        return ", ".join(f"{d} ({c:,})" for d, c in self.descriptions[:40])
