# tests/smoke_retriever.py
from __future__ import annotations
from config import VECTOR_PERSIST_DIR, VECTOR_COLLECTION
from utilis.retriever_utils import (
    init_retriever,
    diag_categories,
)
from langchain_core.documents import Document

QUESTIONS = [
    "睡眠の推奨時間は？",
    "飲酒の健康リスクについて教えて",
    "BMI 25 以上の判定基準は？",
    "厚労省の資料で推奨される身体活動は？",
    "ヘッドホンの難聴リスクは？",
]

def _print_block(title: str, rows):
    print(f"\n=== {title} ===")
    for q, docs in rows:
        head = (docs[0].page_content if docs else "")[:80].replace("\n", " ").replace("\r", " ")
        print(f"- Q: {q} | hits={len(docs)} | head='{head}'")

def run_case(title: str, *, use_mmr: bool, score_threshold=None, k=5, fetch_k=20):
    retriever = init_retriever(
        persist_dir=VECTOR_PERSIST_DIR,
        collection=VECTOR_COLLECTION,
        use_mmr=use_mmr,
        k=k,
        fetch_k=fetch_k,
        score_threshold=score_threshold,  # MMR時は内部で無視/安全化される想定
    )
    rows = []
    for q in QUESTIONS:
        try:
            docs = retriever.invoke(q) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(q)
        except Exception as e:
            # 万一ここで例外が出た場合でもテストは続行
            print(f"[ERROR] {title} - '{q}': {e!r}")
            docs = []
        rows.append((q, docs))
    _print_block(title, rows)
    return rows

def main():
    print("categories:", diag_categories(VECTOR_PERSIST_DIR, VECTOR_COLLECTION, sample=500))

    # 1) MMR（多様性重視）
    run_case("MMR (k=5, fetch_k=20)", use_mmr=True, k=5, fetch_k=20)

    # 2) similarity（しきい値なし：一点突破）
    run_case("similarity (no threshold)", use_mmr=False, k=5)

    # 3) similarity + しきい値（下限品質を担保）
    run_case("similarity + threshold=0.35", use_mmr=False, score_threshold=0.35, k=5, fetch_k=30)

    print("\nSummary: 3ブロックとも例外なし＆各質問で hits>0 が複数あれば合格目安です。")

if __name__ == "__main__":
    main()
