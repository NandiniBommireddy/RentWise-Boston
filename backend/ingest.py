"""Build the RentWise DuckDB database from the raw Analyze Boston CSVs.

Design note — why "property cards" instead of chunked rows
----------------------------------------------------------
The naive RAG move here is to treat each of the 790k CSV rows as a document, embed
them, and hope cosine similarity surfaces the right ones. That fails badly on this
data for three reasons:

  1. A single row ("Rodent Activity, 44 Portsmouth St, 2026-08-21") is nearly
     identical in embedding space to 38k other sanitation rows. Retrieval collapses.
  2. The question a renter actually asks is about a *property*, not a row. "Is this
     building safe?" needs the whole violation history at once.
  3. Counting questions ("which neighborhood is worst for rats?") cannot be answered
     by retrieval at all -- they need aggregation over the full table.

So we build two retrieval surfaces from the same source:

  * `property_cards` -- one row per distinct address (~86k), pre-aggregating that
    property's entire violation history plus its short-term-rental eligibility
    flags into a single dense text document. This is the unit of *semantic*
    retrieval, and it is joined across both datasets at ingest time.
  * the raw `rentsmart` / `str_eligibility` tables -- kept intact for the
    text-to-SQL path, which handles aggregate and temporal questions exactly
    rather than approximately.

Run:  python -m backend.ingest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

from .sources import DATASETS, INCOME_RESTRICTED, RENTSMART, REQUIRED, STR_ELIGIBILITY

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "data" / "downloads"
DB_PATH = ROOT / "data" / "rentwise.duckdb"

# These CSVs contain unescaped quotes inside owner names, which defeats DuckDB's
# dialect sniffer. Pin the dialect explicitly and tolerate the bad rows.
READ_OPTS = (
    "all_varchar=true, header=true, delim=',', quote='\"', escape='\"', "
    "strict_mode=false, ignore_errors=true, null_padding=true"
)


def read_csv(path: Path) -> str:
    return f"read_csv('{path.as_posix()}', {READ_OPTS})"


def norm_sql(col: str) -> str:
    """Normalize a Boston street address into a join key.

    RentSmart writes "44 Portsmouth St, 02135"; the SAM-derived STR file writes
    "44 Portsmouth St". Dropping the zip tail and squashing punctuation makes the two
    line up -- measured at 100% coverage of distinct RentSmart addresses, because both
    datasets are ultimately keyed on the City's SAM address registry.
    """
    no_zip = f"regexp_replace({col}, ',[ ]*0[0-9][0-9][0-9][0-9][ ]*$', '')"
    alnum = f"regexp_replace(trim(lower({no_zip})), '[^a-z0-9 ]', ' ', 'g')"
    return f"regexp_replace({alnum}, ' +', ' ', 'g')"


def log(msg: str) -> None:
    print(f"[ingest] {msg}", flush=True)


def check_inputs() -> None:
    missing = [d.csv for d in REQUIRED if not (DOWNLOADS / d.csv).exists()]
    if missing:
        sys.exit(
            f"Missing required CSV(s) in {DOWNLOADS}: {', '.join(missing)}\n"
            "Run: python -m backend.download"
        )


def build_rentsmart(con: duckdb.DuckDBPyConnection) -> None:
    src = read_csv(DOWNLOADS / RENTSMART.csv)
    con.execute(f"""
        CREATE OR REPLACE TABLE rentsmart AS
        SELECT
            CAST(_id AS BIGINT)                       AS record_id,
            TRY_CAST(date AS TIMESTAMP)               AS occurred_at,
            violation_type,
            description,
            address,
            {norm_sql('address')}                     AS addr_key,
            neighborhood,
            zip_code,
            parcel,
            owner,
            TRY_CAST(year_built AS INTEGER)           AS year_built,
            TRY_CAST(year_remodeled AS INTEGER)       AS year_remodeled,
            property_type,
            TRY_CAST(latitude AS DOUBLE)              AS latitude,
            TRY_CAST(longitude AS DOUBLE)             AS longitude
        FROM {src}
        WHERE address IS NOT NULL
    """)
    n = con.execute("SELECT count(*) FROM rentsmart").fetchone()[0]
    log(f"rentsmart: {n:,} rows")


def build_str(con: duckdb.DuckDBPyConnection) -> None:
    """Load STR eligibility, collapsing the ~3k duplicate address rows.

    Duplicates are aggregated conservatively rather than arbitrarily picked: a
    property counts as a problem property if *any* of its rows says so, and we keep
    the highest open-violation count. Silently taking one row would understate risk.
    """
    src = read_csv(DOWNLOADS / STR_ELIGIBILITY.csv)
    con.execute(f"""
        CREATE OR REPLACE TABLE str_eligibility AS
        SELECT
            {norm_sql('sam_address')}                                  AS addr_key,
            min(CAST(_id AS BIGINT))                                   AS record_id,
            any_value(sam_address)                                     AS sam_address,
            max(TRY_CAST("open violation count" AS INTEGER))           AS open_violations,
            max(TRY_CAST("violations in the last 6 months" AS INTEGER)) AS violations_6mo,
            max(TRY_CAST("units in building" AS INTEGER))              AS units_in_building,
            bool_or(upper("problem property") = 'Y')                   AS problem_property,
            bool_or(upper("problem property owner") = 'Y')             AS problem_property_owner,
            bool_or(upper("income restricted") = 'Y')                  AS income_restricted,
            bool_or(upper("legally restricted") = 'Y')                 AS legally_restricted,
            bool_or(upper("home-share eligible") = 'Y')                AS home_share_eligible,
            bool_or(upper("limited-share eligible") = 'Y')             AS limited_share_eligible,
            bool_or(upper("owner-adjacent eligible") = 'Y')            AS owner_adjacent_eligible,
            bool_or(upper("unit owner-occupied") = 'Y')                AS unit_owner_occupied,
            bool_or(upper("building owner-occupied") = 'Y')            AS building_owner_occupied,
            bool_or(upper(issued_registration) = 'Y')                  AS issued_registration
        FROM {src}
        WHERE sam_address IS NOT NULL
        GROUP BY 1
    """)
    n = con.execute("SELECT count(*) FROM str_eligibility").fetchone()[0]
    log(f"str_eligibility: {n:,} unique addresses")


def build_income_restricted(con: duckdb.DuckDBPyConnection) -> None:
    path = DOWNLOADS / INCOME_RESTRICTED.csv
    if not path.exists():
        log("income_restricted: CSV absent, skipping")
        return
    src = read_csv(path)
    con.execute(f"""
        CREATE OR REPLACE TABLE income_restricted AS
        SELECT
            CAST(_id AS BIGINT)                                  AS record_id,
            "Project_Name"                                       AS project_name,
            "Neighborhood"                                       AS neighborhood,
            "Zip Code"                                           AS zip_code,
            TRY_CAST("TtlProjUnits" AS INTEGER)                  AS total_units,
            TRY_CAST("Total Income-Restricted" AS INTEGER)       AS income_restricted_units,
            TRY_CAST("Income-Restricted Rental" AS INTEGER)      AS income_restricted_rental,
            "Tenure"                                             AS tenure,
            "Public/ Private"                                    AS public_private,
            "Includes Senior Units?"                             AS senior_units,
            "Section 8"                                          AS section_8
        FROM {src}
    """)
    n = con.execute("SELECT count(*) FROM income_restricted").fetchone()[0]
    log(f"income_restricted: {n:,} projects")


def build_property_cards(con: duckdb.DuckDBPyConnection) -> None:
    """One document per property: full violation history + STR eligibility, joined."""
    con.execute("""
        CREATE OR REPLACE TABLE property_agg AS
        SELECT
            addr_key,
            -- mode() picks the most common spelling/geocode across a property's rows,
            -- which is more robust than any_value() when a few rows are malformed.
            mode(address)                                                AS address,
            mode(neighborhood)                                           AS neighborhood,
            mode(zip_code)                                               AS zip_code,
            mode(parcel)                                                 AS parcel,
            mode(owner)                                                  AS owner,
            mode(property_type)                                          AS property_type,
            max(year_built)                                              AS year_built,
            max(year_remodeled)                                          AS year_remodeled,
            median(latitude)                                             AS latitude,
            median(longitude)                                            AS longitude,
            count(*)                                                     AS total_records,
            min(occurred_at)                                             AS first_seen,
            max(occurred_at)                                             AS last_seen,
            count(*) FILTER (occurred_at >= now() - INTERVAL 12 MONTH)   AS records_12mo,
            count(*) FILTER (violation_type = 'Housing Violations')      AS housing_violations,
            count(*) FILTER (violation_type = 'Building Violations')     AS building_violations,
            count(*) FILTER (violation_type = 'Enforcement Violations')  AS enforcement_violations,
            count(*) FILTER (violation_type = 'Housing Complaints')      AS housing_complaints,
            count(*) FILTER (violation_type = 'Sanitation Requests')     AS sanitation_requests,
            count(*) FILTER (violation_type = 'Civic Maintenance Requests') AS civic_maintenance,
            -- The distinct issue vocabulary for this property, most frequent first.
            list(DISTINCT description)                                   AS issue_list
        FROM rentsmart
        GROUP BY addr_key
    """)

    con.execute("""
        CREATE OR REPLACE TABLE property_cards AS
        SELECT
            p.*,
            s.record_id              AS str_record_id,
            s.open_violations        AS str_open_violations,
            s.violations_6mo         AS str_violations_6mo,
            s.units_in_building,
            s.problem_property,
            s.problem_property_owner,
            s.income_restricted,
            s.home_share_eligible,
            s.limited_share_eligible,
            s.owner_adjacent_eligible,
            s.unit_owner_occupied,
            s.building_owner_occupied,
            s.issued_registration,
            (SELECT min(record_id) FROM rentsmart r WHERE r.addr_key = p.addr_key)
                                     AS rentsmart_record_id
        FROM property_agg p
        LEFT JOIN str_eligibility s USING (addr_key)
    """)

    # The searchable rendering. Written as prose rather than key=value pairs so that
    # BM25 and the embedding model both have natural language to work with, and so a
    # retrieved card can be dropped straight into the prompt as evidence.
    # Empty CASE branches must yield NULL, not '' -- concat_ws skips NULL but happily
    # concatenates empty strings, which is what produced runs of stray spaces. The
    # outer regexp pass is a belt-and-braces cleanup for spacing before punctuation.
    con.execute("""
        CREATE OR REPLACE TABLE property_docs AS
        SELECT
            addr_key,
            address,
            neighborhood,
            regexp_replace(
              regexp_replace(
                concat_ws(' ',
                    concat(address, ' in ', neighborhood, ' ', coalesce(zip_code, '')),
                    concat('. Property type: ', coalesce(property_type, 'unknown')),
                    CASE WHEN year_built IS NOT NULL
                         THEN concat(', built ', year_built) END,
                    CASE WHEN year_remodeled IS NOT NULL AND year_remodeled > 0
                         THEN concat(', remodeled ', year_remodeled) END,
                    concat('. Owner: ', coalesce(owner, 'unknown')),
                    concat('. ', total_records, ' RentSmart records total, ',
                           records_12mo, ' in the last 12 months: '),
                    concat_ws(', ',
                        nullif(concat(housing_violations, ' housing violations'), '0 housing violations'),
                        nullif(concat(building_violations, ' building violations'), '0 building violations'),
                        nullif(concat(enforcement_violations, ' enforcement violations'), '0 enforcement violations'),
                        nullif(concat(housing_complaints, ' housing complaints'), '0 housing complaints'),
                        nullif(concat(sanitation_requests, ' sanitation requests'), '0 sanitation requests'),
                        nullif(concat(civic_maintenance, ' civic maintenance requests'), '0 civic maintenance requests')
                    ),
                    concat('. Reported issues: ', array_to_string(issue_list[1:25], '; ')),
                    '. Short-term rental status:',
                    CASE WHEN str_open_violations IS NULL
                         THEN 'not found in the eligibility file.'
                         ELSE concat_ws(' ',
                            concat(str_open_violations, ' open violations, ',
                                   coalesce(str_violations_6mo, 0), ' in the last 6 months.'),
                            CASE WHEN problem_property THEN 'Flagged as a problem property.' END,
                            CASE WHEN problem_property_owner THEN 'Owner flagged as a problem property owner.' END,
                            CASE WHEN income_restricted THEN 'Income restricted.' END,
                            CASE WHEN home_share_eligible THEN 'Home-share eligible.'
                                 ELSE 'Not home-share eligible.' END,
                            CASE WHEN limited_share_eligible THEN 'Limited-share eligible.' END,
                            CASE WHEN owner_adjacent_eligible THEN 'Owner-adjacent eligible.' END,
                            CASE WHEN building_owner_occupied THEN 'Building is owner-occupied.' END,
                            CASE WHEN units_in_building IS NOT NULL
                                 THEN concat(units_in_building, ' units in building.') END)
                    END
                ),
              ' +([.,;:])', '\\1', 'g'),
            ' +', ' ', 'g') AS card_text
        FROM property_cards
    """)

    n = con.execute("SELECT count(*) FROM property_cards").fetchone()[0]
    matched = con.execute(
        "SELECT count(*) FROM property_cards WHERE str_open_violations IS NOT NULL"
    ).fetchone()[0]
    log(f"property_cards: {n:,} properties ({matched:,} = {100 * matched / n:.1f}% joined to STR)")


def build_fts(con: duckdb.DuckDBPyConnection) -> None:
    """BM25 index over the card text -- the sparse half of hybrid retrieval."""
    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    con.execute("PRAGMA create_fts_index('property_docs', 'addr_key', 'card_text', overwrite=1)")
    log("BM25 index built over property_docs.card_text")


def build_neighborhood_rollup(con: duckdb.DuckDBPyConnection) -> None:
    """Small precomputed table so neighborhood comparisons are instant in the demo."""
    con.execute("""
        CREATE OR REPLACE TABLE neighborhood_stats AS
        SELECT
            neighborhood,
            count(*)                                                    AS records,
            count(DISTINCT addr_key)                                    AS properties,
            count(*) FILTER (occurred_at >= now() - INTERVAL 12 MONTH)  AS records_12mo,
            count(*) FILTER (description ILIKE '%rodent%'
                          OR description ILIKE '%mice%'
                          OR description ILIKE '%pest%')                AS pest_reports,
            count(*) FILTER (description ILIKE '%heat%')                AS heat_reports,
            count(*) FILTER (description ILIKE '%unsafe%')              AS unsafe_reports,
            round(count(*)::DOUBLE / nullif(count(DISTINCT addr_key), 0), 2)
                                                                        AS records_per_property
        FROM rentsmart
        WHERE neighborhood IS NOT NULL
        GROUP BY neighborhood
        ORDER BY records DESC
    """)
    n = con.execute("SELECT count(*) FROM neighborhood_stats").fetchone()[0]
    log(f"neighborhood_stats: {n} neighborhoods")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RentWise DuckDB database.")
    parser.add_argument("--db", default=str(DB_PATH), help="output .duckdb path")
    args = parser.parse_args()

    check_inputs()
    started = time.time()

    out = Path(args.db)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    con = duckdb.connect(str(out))
    try:
        build_rentsmart(con)
        build_str(con)
        build_income_restricted(con)
        build_property_cards(con)
        build_neighborhood_rollup(con)
        build_fts(con)
        con.execute("CHECKPOINT")
    finally:
        con.close()

    size_mb = out.stat().st_size / 1_000_000
    log(f"wrote {out} ({size_mb:.0f} MB) in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
