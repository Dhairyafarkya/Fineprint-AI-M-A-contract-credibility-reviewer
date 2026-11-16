"""Clean, chunk, embed and index a folder of .txt agreements.

Usage:
    python scripts/ingest_dataset.py data/raw --name my_dataset
    python scripts/ingest_dataset.py data/raw --name my_dataset --embeddings tfidf

Files that are numbered pieces of one document (contract_0.txt_1.txt, contract_0.txt_2.txt, ...)
are stitched back into one document first. The index is written to data/indexes/<name>/.
The dashboard does not need this step: it reads data/raw directly. Use it to pre-build
an index for scripts or very large datasets. No API key is needed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ma_engine.config import get_settings  # noqa: E402
from ma_engine.ingestion.loader import load_folder  # noqa: E402
from ma_engine.retrieval.indexer import build_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="Folder containing .txt documents (searched recursively)")
    ap.add_argument("--name", default="dataset", help="Index name (folder under data/indexes)")
    ap.add_argument("--embeddings", choices=["auto", "sentence-transformers", "tfidf"], default=None)
    ap.add_argument("--chunk-words", type=int, default=None)
    ap.add_argument("--overlap-words", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    if args.embeddings:
        settings.retrieval = replace(settings.retrieval, embedding_backend=args.embeddings)
    if args.chunk_words or args.overlap_words:
        settings.chunking = replace(
            settings.chunking,
            chunk_words=args.chunk_words or settings.chunking.chunk_words,
            overlap_words=args.overlap_words or settings.chunking.overlap_words,
        )

    docs = load_folder(args.folder)
    if not docs:
        print(f"No .txt files found in {args.folder}")
        return 1
    t0 = time.time()
    index = build_index(docs, settings)
    out = index.save(settings.retrieval.index_dir / args.name)
    print(f"\nIndexed {len(index.documents)} documents into {len(index)} chunks "
          f"({index.embedder.name}) in {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
