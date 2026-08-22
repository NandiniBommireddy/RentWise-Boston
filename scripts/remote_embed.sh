#!/usr/bin/env bash
# Build the dense-retrieval embeddings on a remote GPU machine (e.g. a RunPod pod)
# and install the artifacts locally, end to end:
#
#   1. export property cards from data/rentwise.duckdb -> cards.jsonl.gz
#   2. copy the export + embed script to the remote over ssh
#   3. remote: venv + pip install fastembed-gpu, embed all cards on the GPU
#   4. copy property_embeddings.npy + property_embedding_keys.txt back into data/
#
# Usage:
#   scripts/remote_embed.sh user@host            # standard ssh
#   scripts/remote_embed.sh root@1.2.3.4 -p 12345  # RunPod exposed TCP port
#
# Everything after the host is passed to ssh/scp verbatim (-p port, -i key, ...).
# Requires: the DuckDB build done locally first (python -m backend.ingest), and
# python3 + pip available on the remote (any recent CUDA image has both).
# After it finishes, restart the backend server to pick up dense retrieval.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 user@host [extra ssh args...]" >&2
  exit 1
fi

HOST="$1"; shift
SSH_ARGS=("$@")
# scp uses -P for port where ssh uses -p
SCP_ARGS=()
for arg in "${SSH_ARGS[@]+"${SSH_ARGS[@]}"}"; do
  if [ "$arg" = "-p" ]; then SCP_ARGS+=("-P"); else SCP_ARGS+=("$arg"); fi
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
VENV_PY="$ROOT/.venv/bin/python"
REMOTE_DIR="rentwise-embed"
EXPORT="$DATA/cards.jsonl.gz"

[ -f "$DATA/rentwise.duckdb" ] || { echo "no $DATA/rentwise.duckdb — run: python -m backend.ingest" >&2; exit 1; }
[ -x "$VENV_PY" ] || { echo "no .venv — run: python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt" >&2; exit 1; }

echo "[1/4] exporting property cards -> $EXPORT"
"$VENV_PY" - "$DATA/rentwise.duckdb" "$EXPORT" <<'PY'
import gzip, json, sys
import duckdb

db, out = sys.argv[1], sys.argv[2]
con = duckdb.connect(db, read_only=True)
# Same query and order as backend/build_embeddings.py — row order is the
# contract between the keys file and the matrix.
rows = con.execute("""
    SELECT d.addr_key, d.card_text
    FROM property_docs d JOIN property_cards c USING (addr_key)
    ORDER BY c.total_records DESC
""").fetchall()
con.close()
with gzip.open(out, "wt") as fh:
    for addr_key, card_text in rows:
        fh.write(json.dumps({"addr_key": addr_key, "card_text": card_text}) + "\n")
print(f"    {len(rows):,} cards")
PY

echo "[2/4] copying export + scripts to $HOST:$REMOTE_DIR/"
ssh "${SSH_ARGS[@]+"${SSH_ARGS[@]}"}" "$HOST" "mkdir -p $REMOTE_DIR"
scp "${SCP_ARGS[@]+"${SCP_ARGS[@]}"}" "$EXPORT" \
  "$ROOT/scripts/embed_cards_remote.py" "$ROOT/scripts/pod_setup.sh" "$HOST:$REMOTE_DIR/"

echo "[3/4] embedding on $HOST (GPU if available)"
# pod_setup.sh installs the CUDA userspace libs (cuBLAS, cuDNN) as pip wheels,
# so this works regardless of which CUDA toolkit the container image ships —
# only the host driver matters.
ssh "${SSH_ARGS[@]+"${SSH_ARGS[@]}"}" "$HOST" "bash $REMOTE_DIR/pod_setup.sh"

echo "[4/4] fetching artifacts into $DATA/"
scp "${SCP_ARGS[@]+"${SCP_ARGS[@]}"}" \
  "$HOST:$REMOTE_DIR/out/property_embeddings.npy" \
  "$HOST:$REMOTE_DIR/out/property_embedding_keys.txt" \
  "$DATA/"

"$VENV_PY" - "$DATA" <<'PY'
import sys
from pathlib import Path
import numpy as np

data = Path(sys.argv[1])
m = np.load(data / "property_embeddings.npy")
keys = (data / "property_embedding_keys.txt").read_text().splitlines()
assert m.ndim == 2 and m.shape[1] == 384, f"bad shape {m.shape}"
assert m.shape[0] == len(keys), f"{m.shape[0]} vectors vs {len(keys)} keys"
print(f"    verified: {m.shape[0]:,} vectors x {m.shape[1]}, keys aligned")
PY

echo "done — restart the backend to enable dense retrieval:"
echo "  .venv/bin/uvicorn backend.app:app --port 8000"
