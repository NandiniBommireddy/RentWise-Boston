"""Embed every property card locally, for the dense half of hybrid retrieval.

Uses fastembed (ONNX Runtime) with BAAI/bge-small-en-v1.5 -- no cloud embedding API
and no PyTorch dependency, so this runs offline on a laptop. Vectors are stored
L2-normalized in a flat .npy; at 86k x 384 a brute-force matmul is sub-millisecond,
so there is no reason to add a vector index.

Run:  python -m backend.build_embeddings
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from .retrieval import DB_PATH, EMBED_KEYS_PATH, EMBED_MODEL, EMBED_PATH


# Embedding all 86k cards needs more memory than an 8 GB machine has spare; it swaps
# and takes over an hour. So the default is a bounded slice of the busiest properties
# -- the ones questions are actually about -- and the full run is opt-in via --all.
DEFAULT_LIMIT = 20_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local embeddings for property cards.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"embed only the N busiest properties (default {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="embed every property card. Slow and memory-hungry: expect 1h+ and heavy "
        "swapping on 8 GB. Prefer the default.",
    )
    args = parser.parse_args()

    import numpy as np
    from fastembed import TextEmbedding

    con = duckdb.connect(str(DB_PATH), read_only=True)
    total_cards = con.execute("SELECT count(*) FROM property_docs").fetchone()[0]
    # Order by history depth so a bounded run covers the properties most likely to be
    # asked about, rather than an arbitrary slice.
    sql = """
        SELECT d.addr_key, d.card_text
        FROM property_docs d JOIN property_cards c USING (addr_key)
        ORDER BY c.total_records DESC
    """
    if not args.all:
        sql += f" LIMIT {args.limit}"
    rows = con.execute(sql).fetchall()
    con.close()

    if args.all:
        print(f"[embed] --all: embedding every one of {total_cards:,} cards. This is slow.")
    else:
        print(
            f"[embed] {len(rows):,} of {total_cards:,} cards (busiest first). "
            f"Use --all for everything, or --limit N to change this."
        )

    keys = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    print(f"[embed] {len(texts):,} property cards with {EMBED_MODEL}")

    model = TextEmbedding(model_name=EMBED_MODEL)
    started = time.time()
    vectors = []
    for i, vec in enumerate(model.embed(texts, batch_size=args.batch_size)):
        vectors.append(vec)
        if (i + 1) % 5000 == 0:
            rate = (i + 1) / (time.time() - started)
            eta = (len(texts) - i - 1) / rate
            print(f"[embed] {i + 1:,}/{len(texts):,}  {rate:.0f}/s  eta {eta / 60:.1f}m", flush=True)

    matrix = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.where(norms == 0, 1.0, norms)

    Path(EMBED_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBED_PATH, matrix)
    EMBED_KEYS_PATH.write_text("\n".join(keys))
    print(
        f"[embed] wrote {EMBED_PATH} {matrix.shape} "
        f"({matrix.nbytes / 1e6:.0f} MB) in {(time.time() - started) / 60:.1f}m"
    )


if __name__ == "__main__":
    main()
