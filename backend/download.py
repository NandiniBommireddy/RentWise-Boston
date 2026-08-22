"""Fetch the raw CSVs from Analyze Boston's CKAN datastore.

Kept as a script rather than a manual step so the whole pipeline is reproducible:
    python -m backend.download && python -m backend.ingest

Both primary datasets refresh daily, so re-running this picks up new records.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from .sources import DATASETS

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "data" / "downloads"

DUMP_URL = "https://data.boston.gov/datastore/dump/{resource_id}"


def fetch(key: str, force: bool) -> None:
    ds = DATASETS[key]
    dest = DOWNLOADS / ds.csv
    if dest.exists() and not force:
        print(f"[download] {ds.csv} already present ({dest.stat().st_size / 1e6:.1f} MB), skipping")
        return

    url = DUMP_URL.format(resource_id=ds.resource_id)
    print(f"[download] {ds.title} -> {ds.csv}")
    tmp = dest.with_suffix(".csv.part")
    try:
        with urllib.request.urlopen(url, timeout=300) as resp, tmp.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
    except Exception as exc:  # noqa: BLE001 - surface the URL that failed
        tmp.unlink(missing_ok=True)
        sys.exit(f"[download] failed for {ds.title} ({url}): {exc}")

    tmp.replace(dest)
    print(f"[download] wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Analyze Boston CSVs.")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--only", choices=sorted(DATASETS), help="fetch a single dataset")
    args = parser.parse_args()

    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    keys = [args.only] if args.only else list(DATASETS)
    for key in keys:
        fetch(key, args.force)


if __name__ == "__main__":
    main()
