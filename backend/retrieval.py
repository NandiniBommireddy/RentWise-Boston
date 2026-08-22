"""Retrieval over the RentWise database.

Three surfaces, because one retriever cannot answer every kind of housing question:

  * `search_hybrid`  -- BM25 (DuckDB FTS) fused with dense vectors (local ONNX
    embeddings) via Reciprocal Rank Fusion. Sparse catches exact tokens a renter
    types (street names, "Rodent Activity", owner LLCs); dense catches paraphrase
    ("is it falling apart" -> "Unsafe Dangerous Conditions"). Neither alone is
    sufficient, which the eval suite quantifies.
  * `lookup_property` -- exact address resolution, for "tell me about 44 Portsmouth St".
  * `run_sql`         -- read-only DuckDB for aggregate and temporal questions that
    retrieval fundamentally cannot answer ("which neighborhood has the most rats?").

Dense retrieval is optional. If `fastembed` or the embedding file is missing, hybrid
degrades to BM25-only and says so via `dense_enabled`, rather than failing the demo.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from .sources import RENTSMART, STR_ELIGIBILITY, SourceRef

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "rentwise.duckdb"
EMBED_PATH = ROOT / "data" / "property_embeddings.npy"
EMBED_KEYS_PATH = ROOT / "data" / "property_embedding_keys.txt"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Tables the text-to-SQL path is allowed to touch.
ALLOWED_TABLES = {
    "rentsmart",
    "str_eligibility",
    "income_restricted",
    "property_cards",
    "property_docs",
    "neighborhood_stats",
}

_WRITE_TOKENS = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|install|load|"
    r"pragma|export|import|set|call|truncate|replace|grant|revoke|vacuum)\b",
    re.IGNORECASE,
)


class SqlNotAllowed(ValueError):
    """Raised when a generated query is not a plain read."""


@dataclass
class PropertyHit:
    addr_key: str
    address: str
    neighborhood: str | None
    card_text: str
    score: float
    retriever: str = ""
    latitude: float | None = None
    longitude: float | None = None
    rentsmart_record_id: int | None = None
    str_record_id: int | None = None
    total_records: int = 0
    top_issue: str | None = None
    property_type: str | None = None
    owner: str | None = None
    year_built: int | None = None

    def citations(self) -> list[SourceRef]:
        """Every card is grounded in two datasets; cite both."""
        refs = [
            SourceRef(
                dataset_key=RENTSMART.key,
                record_id=self.rentsmart_record_id,
                label=f"{self.address} — {self.total_records} records",
                detail=self.top_issue or "",
            )
        ]
        if self.str_record_id is not None:
            refs.append(
                SourceRef(
                    dataset_key=STR_ELIGIBILITY.key,
                    record_id=self.str_record_id,
                    label=f"{self.address} — rental eligibility",
                )
            )
        return refs


@dataclass
class SqlResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False

    def to_markdown(self, limit: int = 40) -> str:
        if not self.rows:
            return "(no rows)"
        head = " | ".join(self.columns)
        sep = " | ".join("---" for _ in self.columns)
        body = "\n".join(
            " | ".join("" if v is None else str(v) for v in row) for row in self.rows[:limit]
        )
        note = f"\n({len(self.rows)} rows, showing {min(limit, len(self.rows))})" if len(self.rows) > limit else ""
        return f"{head}\n{sep}\n{body}{note}"


@dataclass
class RetrievalTrace:
    """What the agent actually did -- surfaced to the UI and asserted on in evals."""

    tool_calls: list[dict] = field(default_factory=list)
    sql_queries: list[str] = field(default_factory=list)
    hits: list[PropertyHit] = field(default_factory=list)
    citations: list[SourceRef] = field(default_factory=list)
    # Map pins that did not come from a PropertyHit -- e.g. rows returned by a SQL
    # aggregate that selected coordinates.
    pins: list[dict] = field(default_factory=list)

    def add_citations(self, refs: list[SourceRef]) -> None:
        seen = {(c.dataset_key, c.record_id, c.label) for c in self.citations}
        for ref in refs:
            key = (ref.dataset_key, ref.record_id, ref.label)
            if key not in seen:
                seen.add(key)
                self.citations.append(ref)


def normalize_address(text: str) -> str:
    """Mirror of the SQL normalizer in ingest.py, for Python-side address matching."""
    no_zip = re.sub(r",\s*0\d{4}\s*$", "", text or "")
    alnum = re.sub(r"[^a-z0-9 ]", " ", no_zip.strip().lower())
    return re.sub(r"\s+", " ", alnum).strip()


def guard_sql(sql: str) -> str:
    """Allow exactly one read-only SELECT/WITH statement over known tables.

    The connection is opened read-only, so this is defence in depth rather than the
    only barrier -- but it turns a confusing DuckDB permission error into a message
    the model can actually recover from on the next turn.
    """
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise SqlNotAllowed("empty query")
    if ";" in cleaned:
        raise SqlNotAllowed("only a single statement is allowed")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise SqlNotAllowed("query must start with SELECT or WITH")
    if match := _WRITE_TOKENS.search(cleaned):
        raise SqlNotAllowed(f"'{match.group(0)}' is not allowed in a read-only query")
    return cleaned


class RentWiseIndex:
    def __init__(self, db_path: Path | str = DB_PATH, enable_dense: bool = True) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"{self.db_path} not found. Run: python -m backend.download && python -m backend.ingest"
            )
        self.con = duckdb.connect(str(self.db_path), read_only=True)
        self.con.execute("LOAD fts")
        self._embeddings = None
        self._embed_keys: list[str] = []
        self._embedder = None
        if enable_dense:
            self._load_dense()

    # ---------- dense ----------

    def _load_dense(self) -> None:
        if not EMBED_PATH.exists() or not EMBED_KEYS_PATH.exists():
            log.info("dense retrieval disabled: run `python -m backend.build_embeddings`")
            return
        try:
            import numpy as np
            from fastembed import TextEmbedding
        except ImportError as exc:
            log.info("dense retrieval disabled: %s", exc)
            return
        try:
            self._embeddings = np.load(EMBED_PATH)
            self._embed_keys = EMBED_KEYS_PATH.read_text().splitlines()
            if len(self._embed_keys) != self._embeddings.shape[0]:
                log.warning("embedding/key length mismatch; disabling dense retrieval")
                self._embeddings = None
                return
            self._embedder = TextEmbedding(model_name=EMBED_MODEL)
            log.info("dense retrieval ready: %s vectors", f"{self._embeddings.shape[0]:,}")
        except Exception as exc:  # noqa: BLE001 - never let this break the demo
            log.warning("dense retrieval unavailable: %s", exc)
            self._embeddings = None

    @property
    def dense_enabled(self) -> bool:
        return self._embeddings is not None and self._embedder is not None

    @property
    def dense_count(self) -> int:
        return 0 if self._embeddings is None else int(self._embeddings.shape[0])

    def search_dense(self, query: str, k: int = 8) -> list[PropertyHit]:
        if not self.dense_enabled:
            return []
        import numpy as np

        vec = next(iter(self._embedder.query_embed(query)))
        vec = np.asarray(vec, dtype="float32")
        vec /= (np.linalg.norm(vec) or 1.0)
        # Embeddings are stored L2-normalized, so a dot product is cosine similarity.
        scores = self._embeddings @ vec
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        keys = [self._embed_keys[i] for i in top]
        hits = self._hydrate(keys)
        by_key = {h.addr_key: h for h in hits}
        out = []
        for i, key in zip(top, keys):
            if hit := by_key.get(key):
                hit.score = float(scores[i])
                hit.retriever = "dense"
                out.append(hit)
        return out

    # ---------- sparse ----------

    def search_bm25(self, query: str, k: int = 8) -> list[PropertyHit]:
        rows = self.con.execute(
            """
            SELECT addr_key, score FROM (
                SELECT addr_key, fts_main_property_docs.match_bm25(addr_key, ?) AS score
                FROM property_docs
            ) WHERE score IS NOT NULL
            ORDER BY score DESC LIMIT ?
            """,
            [query, k],
        ).fetchall()
        if not rows:
            return []
        scores = dict(rows)
        hits = self._hydrate(list(scores))
        for hit in hits:
            hit.score = float(scores.get(hit.addr_key, 0.0))
            hit.retriever = "bm25"
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    # ---------- fusion ----------

    def search_hybrid(self, query: str, k: int = 8, rrf_k: int = 60) -> list[PropertyHit]:
        """Reciprocal Rank Fusion of BM25 and dense results.

        RRF is used rather than score normalization because BM25 scores and cosine
        similarities are not on comparable scales, and RRF needs no tuning per query.
        """
        pools = [self.search_bm25(query, k * 2)]
        if self.dense_enabled:
            pools.append(self.search_dense(query, k * 2))

        fused: dict[str, float] = {}
        source: dict[str, set[str]] = {}
        keep: dict[str, PropertyHit] = {}
        for pool in pools:
            for rank, hit in enumerate(pool):
                fused[hit.addr_key] = fused.get(hit.addr_key, 0.0) + 1.0 / (rrf_k + rank + 1)
                source.setdefault(hit.addr_key, set()).add(hit.retriever)
                keep.setdefault(hit.addr_key, hit)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = []
        for addr_key, score in ordered:
            hit = keep[addr_key]
            hit.score = score
            hit.retriever = "+".join(sorted(source[addr_key]))
            out.append(hit)
        return out

    # ---------- exact lookup ----------

    def lookup_property(self, address: str, limit: int = 5) -> list[PropertyHit]:
        """Resolve a typed address. Exact normalized match first, prefix match second."""
        key = normalize_address(address)
        if not key:
            return []
        rows = self.con.execute(
            "SELECT addr_key FROM property_cards WHERE addr_key = ? LIMIT ?", [key, limit]
        ).fetchall()
        if not rows:
            rows = self.con.execute(
                """
                SELECT addr_key FROM property_cards
                WHERE addr_key LIKE ? || '%' OR addr_key LIKE '%' || ? || '%'
                ORDER BY total_records DESC LIMIT ?
                """,
                [key, key, limit],
            ).fetchall()
        hits = self._hydrate([r[0] for r in rows])
        for hit in hits:
            hit.retriever = "address_lookup"
            hit.score = 1.0
        hits.sort(key=lambda h: h.total_records, reverse=True)
        return hits

    # ---------- owner graph ----------

    def owner_portfolio(self, owner: str, limit: int = 15) -> list[dict]:
        """Traverse owner -> properties -> violations.

        The `owner` field is free text typed by City staff, so the same landlord appears
        as "GBM PORTFOLIO OWNER LLC", "GBM Portfolio Owner, LLC" and worse. Matching is
        therefore a case-insensitive containment search on a punctuation-stripped form
        rather than equality -- imperfect, but it surfaces portfolios that exact
        matching misses entirely.
        """
        needle = re.sub(r"[^a-z0-9 ]", " ", (owner or "").lower())
        needle = re.sub(r"\s+", " ", needle).strip()
        if not needle:
            return []
        rows = self.con.execute(
            """
            WITH normalized AS (
                SELECT *,
                       regexp_replace(regexp_replace(lower(owner), '[^a-z0-9 ]', ' ', 'g'),
                                      ' +', ' ', 'g') AS owner_norm
                FROM property_cards
                WHERE owner IS NOT NULL
            )
            SELECT owner, address, neighborhood, total_records, housing_violations,
                   building_violations, housing_complaints, sanitation_requests,
                   str_open_violations, problem_property_owner, latitude, longitude,
                   rentsmart_record_id
            FROM normalized
            WHERE owner_norm LIKE '%' || ? || '%'
            ORDER BY total_records DESC
            LIMIT ?
            """,
            [needle, limit],
        ).fetchall()
        cols = [
            "owner", "address", "neighborhood", "total_records", "housing_violations",
            "building_violations", "housing_complaints", "sanitation_requests",
            "open_violations", "problem_property_owner", "latitude", "longitude",
            "rentsmart_record_id",
        ]
        return [dict(zip(cols, r)) for r in rows]

    # ---------- sql ----------

    def run_sql(self, sql: str, max_rows: int = 200) -> SqlResult:
        cleaned = guard_sql(sql)
        cur = self.con.execute(cleaned)
        rows = cur.fetchmany(max_rows + 1)
        columns = [d[0] for d in cur.description]
        truncated = len(rows) > max_rows
        return SqlResult(cleaned, columns, rows[:max_rows], truncated)

    def schema_prompt(self) -> str:
        """Compact schema description for the text-to-SQL tool."""
        parts = []
        for table in sorted(ALLOWED_TABLES):
            cols = self.con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
            if cols:
                rendered = ", ".join(f"{c} {t}" for c, t in cols)
                parts.append(f"{table}({rendered})")
        return "\n".join(parts)

    # ---------- internals ----------

    def _hydrate(self, addr_keys: list[str]) -> list[PropertyHit]:
        if not addr_keys:
            return []
        placeholders = ", ".join("?" for _ in addr_keys)
        rows = self.con.execute(
            f"""
            SELECT c.addr_key, c.address, c.neighborhood, d.card_text,
                   c.latitude, c.longitude, c.rentsmart_record_id, c.str_record_id,
                   c.total_records, c.issue_list, c.property_type, c.owner, c.year_built
            FROM property_cards c
            JOIN property_docs d USING (addr_key)
            WHERE c.addr_key IN ({placeholders})
            """,
            addr_keys,
        ).fetchall()
        return [
            PropertyHit(
                addr_key=r[0],
                address=r[1],
                neighborhood=r[2],
                card_text=r[3],
                score=0.0,
                latitude=r[4],
                longitude=r[5],
                rentsmart_record_id=r[6],
                str_record_id=r[7],
                total_records=r[8] or 0,
                top_issue=(r[9][0] if r[9] else None),
                property_type=r[10],
                owner=r[11],
                year_built=r[12],
            )
            for r in rows
        ]

    def close(self) -> None:
        self.con.close()
