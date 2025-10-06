# ingest.py —— 司令塔（差し替え）
import argparse
from utilis.ingest_utils import build_index, DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", type=str, required=True)
    p.add_argument("--include_text", action="store_true")
    p.add_argument("--include_pdf",  action="store_true")
    p.add_argument("--infer_category", action="store_true")
    p.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument("--persist_dir", type=str, default=DEFAULT_PERSIST_DIR)
    p.add_argument("--collection",  type=str, default=DEFAULT_COLLECTION)
    args = p.parse_args()

    # そのままエンジンへ委譲（フォールバックはエンジン側で実施）
    build_index(
        source_dir=args.source_dir,
        include_text=args.include_text,
        include_pdf=args.include_pdf,
        infer_category=args.infer_category,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_dir=args.persist_dir,
        collection=args.collection,
    )
    print("[OK] Indexed to Chroma")
