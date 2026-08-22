"""Embed property cards from a JSONL export — designed to run on a remote GPU box.

Self-contained on purpose: no imports from the backend package, so the remote
machine needs only this file, the cards export, and `pip install fastembed-gpu`
(or plain `fastembed` for CPU). Produces exactly the artifacts the backend's
retrieval layer loads:

    property_embeddings.npy        float32 (N, 384), L2-normalized
    property_embedding_keys.txt    addr_key per row, same order as the matrix

Input format: gzipped JSONL, one {"addr_key": ..., "card_text": ...} per line,
in retrieval-priority order (produced by scripts/remote_embed.sh). Row order is
preserved end-to-end — the keys file must align with the matrix.

Run:  python3 embed_cards_remote.py cards.jsonl.gz --out-dir out/
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # must match backend/retrieval.py


def preload_cuda_libs() -> None:
    """Load CUDA userspace libs (cuBLAS/cuDNN) from nvidia-* pip wheels into the
    process, so onnxruntime's CUDA provider resolves them no matter which CUDA
    toolkit the container image ships — only the host driver has to match."""
    try:
        import onnxruntime

        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls()
            return
    except Exception:
        pass
    try:  # fallback: dlopen every bundled lib with RTLD_GLOBAL
        import ctypes
        from pathlib import Path as P

        import nvidia

        for lib in sorted(P(nvidia.__file__).parent.glob("*/lib/lib*.so*")):
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    except ImportError:
        pass


def _active_providers(model) -> list[str]:
    """Dig the ONNX InferenceSession out of fastembed's wrappers and ask it which
    execution providers are actually active. Best-effort — returns [] if the
    internal attribute chain changes between fastembed versions."""
    obj, hops = model, 0
    while obj is not None and hops < 6:
        if hasattr(obj, "get_providers"):
            try:
                return list(obj.get_providers())
            except Exception:
                return []
        obj = getattr(obj, "model", None)
        hops += 1
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cards", type=Path, help="gzipped JSONL of {addr_key, card_text}")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true", help="skip the CUDA provider")
    args = parser.parse_args()

    import numpy as np
    from fastembed import TextEmbedding

    keys, texts = [], []
    with gzip.open(args.cards, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            keys.append(row["addr_key"])
            texts.append(row["card_text"])
    print(f"[embed] {len(texts):,} property cards with {EMBED_MODEL}")

    model = None
    if not args.cpu:
        preload_cuda_libs()
        try:  # fastembed-gpu + onnxruntime-gpu; falls back below if CUDA is absent
            model = TextEmbedding(model_name=EMBED_MODEL,
                                  providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            print("[embed] CUDA provider requested", flush=True)
        except Exception as exc:
            print(f"[embed] CUDA unavailable ({type(exc).__name__}: {exc}) — using CPU")
    if model is None:
        model = TextEmbedding(model_name=EMBED_MODEL)

    print("[embed] model loaded — first batch initializes the ONNX session...", flush=True)
    started = time.time()
    vectors = []
    for i, vec in enumerate(model.embed(texts, batch_size=args.batch_size)):
        vectors.append(vec)
        if i == 0:
            # The session exists only after the first batch; report what actually
            # runs the matmuls — the definitive GPU-vs-CPU answer.
            providers = _active_providers(model)
            label = ", ".join(providers) if providers else "unknown (fastembed internals changed)"
            print(f"[embed] active providers: {label}", flush=True)
            if providers and providers[0] != "CUDAExecutionProvider" and not args.cpu:
                print("[embed] *** NOT on GPU — expect ~minutes instead of ~1 minute ***", flush=True)
        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.time() - started)
            eta = (len(texts) - i - 1) / rate
            pct = 100 * (i + 1) // len(texts)
            print(f"[embed] {i + 1:,}/{len(texts):,} ({pct}%)  {rate:.0f}/s  eta {eta / 60:.1f}m", flush=True)

    matrix = np.asarray(vectors, dtype="float32")
    if matrix.shape[0] != len(keys):
        raise SystemExit(f"[embed] row mismatch: {matrix.shape[0]} vectors for {len(keys)} keys")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.where(norms == 0, 1.0, norms)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "property_embeddings.npy", matrix)
    (args.out_dir / "property_embedding_keys.txt").write_text("\n".join(keys))
    print(f"[embed] wrote {args.out_dir}/property_embeddings.npy {matrix.shape} "
          f"({matrix.nbytes / 1e6:.0f} MB) in {(time.time() - started) / 60:.1f}m")


if __name__ == "__main__":
    main()
