# utilis/retriever_utils.py  —— 追加/置換パッチ

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
# from utilis.llm_utils import init_llm_emb   # ← 未使用ならコメントアウト
from chromadb import PersistentClient
from config import VECTOR_PERSIST_DIR, VECTOR_COLLECTION, EMBED_MODEL
from apisecret import get_secret
from collections import Counter
from pathlib import Path
from config import CATEGORY_CHAT_ONLY  # "その他・雑談" を想定

# -----------------------------
# 1) VectorStore 初期化（既存）
# -----------------------------
def init_chroma_vectorstore(
    persist_dir: str = VECTOR_PERSIST_DIR,
    collection: str = VECTOR_COLLECTION,
    embedding_model: str = EMBED_MODEL,
):
    emb = OpenAIEmbeddings(model=embedding_model, api_key=get_secret("OPENAI_API_KEY"))
    vs = Chroma(
        persist_directory=persist_dir,
        collection_name=collection,
        embedding_function=emb,
    )
    return vs

# ---------------------------------
# 2) Retriever 初期化を拡張（置換）
# ---------------------------------
def init_retriever(
    persist_dir: str = VECTOR_PERSIST_DIR,
    collection: str = VECTOR_COLLECTION,
    embedding_model: str = EMBED_MODEL,
    k: int = 5,
    fetch_k: int = 20,
    use_mmr: bool = False,
    score_threshold: Optional[float] = None,
    filter_category: Optional[str] = None,
):
    vs = init_chroma_vectorstore(persist_dir, collection, embedding_model)
    filter_kw = {"filter": {"category": filter_category}} if filter_category else {}

    
    if use_mmr:
        # ✅ MMR：fetch_k は使う。threshold は絶対に渡さない
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(fetch_k or 0, k), **filter_kw},
        )

    if score_threshold is None:
        # ✅ similarity：しきい値なし（k と filter だけ）
        return vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, **filter_kw},
        )

    def _get_docs(q: str) -> list[Document]:
        # 広めに取得してからスコアで足切り（最低限の品質ライン）
        _fetch = max(fetch_k or 0, k * 4, 20)
        pairs = vs.similarity_search_with_relevance_scores(q, k=_fetch, **filter_kw)
        kept = [d for (d, s) in pairs if s is not None and s >= score_threshold]
        # 足りなければ上位から充当して k 件に揃える
        if len(kept) < k:
            for d, s in pairs:
                if d not in kept:
                    kept.append(d)
                if len(kept) >= k:
                    break
        return kept[:k]

    class _ManualThresholdRetriever:
        def get_relevant_documents(self, q: str):
            return _get_docs(q)
        def invoke(self, q: str):
            return _get_docs(q)

    return _ManualThresholdRetriever()

# --------------------------------------------------
# 3) Router→Retriever 連携の薄いヘルパ（新規追加）
# --------------------------------------------------
def build_router_aware_retriever(
    route_category: Optional[str],
    persist_dir: str = VECTOR_PERSIST_DIR,
    collection: str = VECTOR_COLLECTION,
    embedding_model: str = EMBED_MODEL,
    k: int = 5,
    fetch_k: int = 20,
    use_mmr: bool = True,
    score_threshold: Optional[float] = None,
):
    """
    Router 決定（例: 'health', 'money', 'career', ...）をそのままメタフィルタに橋渡し。
    route_category が None/空ならフィルタ無しで広く検索。
    """
    filter_cat = route_category
    return init_retriever(
        persist_dir=persist_dir,
        collection=collection,
        embedding_model=embedding_model,
        k=k,
        fetch_k=fetch_k,
        use_mmr=use_mmr,
        score_threshold=score_threshold,
        filter_category=filter_cat,
    )

# ------------------------------------------
# 4) Runnable 化（既存そのまま使用OK）
# ------------------------------------------
def make_retrieve_runnable(retriever):
    """router_utils から .invoke(question) で使えるようにする薄いラッパ。"""
    return RunnableLambda(lambda q: retriever.get_relevant_documents(q))

# ------------------------------------------------
# 5) 参考表示の体裁を整えるフォーマッタ（新規）
# ------------------------------------------------
def format_reference(md: Dict[str, Any]) -> str:
    """
    UI の参考欄で使う統一フォーマット: 「タイトル（p.X） – source」
    """
    title = md.get("title") or ""
    page = md.get("page")
    page_s = f"（p.{page}）" if page is not None else ""
    src = md.get("source") or ""
    return f"{title}{page_s} – {src}".strip(" –")

# ----------------------------------------------
# 6) 単発クエリ用のユーティリティ（置換）
# ----------------------------------------------
def retrieve_texts(
    query: str,
    persist_dir: str = VECTOR_PERSIST_DIR,
    collection: str = VECTOR_COLLECTION,
    embedding_model: str = EMBED_MODEL,
    k: int = 5,
    fetch_k: int = 20,
    use_mmr: bool = True,
    score_threshold: Optional[float] = 0.35,
    filter_category: Optional[str] = None,
) -> List[Document]:
    retriever = init_retriever(
        persist_dir=persist_dir,
        collection=collection,
        embedding_model=embedding_model,
        k=k,
        fetch_k=fetch_k,
        use_mmr=use_mmr,
        score_threshold=score_threshold,
        filter_category=filter_category,
    )
    return retriever.invoke(query)

# ---------------------------------------------------
# 7) 診断: 複数クエリでの当たり具合を可視化（置換）
# ---------------------------------------------------
def probe_retriever(
    queries: Optional[List[str]] = None,
    persist_dir: str = VECTOR_PERSIST_DIR,
    collection: str = VECTOR_COLLECTION,
    embedding_model: str = EMBED_MODEL,
    k: int = 5,
    fetch_k: int = 20,
    use_mmr: bool = True,
    score_threshold: Optional[float] = 0.35,
    filter_category: Optional[str] = None,
) -> List[Tuple[str, int, str]]:
    """
    1問ごとのヒット件数と先頭スニペットを返す簡易診断。
    """
    if queries is None:
        queries = [
            "ベータ版のリリース日は？",
            "会話ログはどこに保存されてどの形式？",
            "UI のタイトルは？",
        ]
    results: List[Tuple[str, int, str]] = []

    retriever = init_retriever(
        persist_dir=persist_dir,
        collection=collection,
        embedding_model=embedding_model,
        k=k,
        fetch_k=fetch_k,
        use_mmr=use_mmr,
        score_threshold=score_threshold,
        filter_category=filter_category,
    )

    for q in queries:
        try:
            docs = retriever.invoke(q)
            head = docs[0].page_content[:120] if docs else ""
            results.append((q, len(docs), head))
        except Exception as e:
            results.append((q, -1, f"ERROR: {e!r}"))
    return results

def diag_categories(
    persist_dir: str,
    collection: str,
    sample: int = 200
) -> Dict[str, int]:
    """
    コレクション内ドキュメントの metadata['category'] 分布をざっくり確認。
    例: {'health': 123, 'money': 45, None: 67}
    """
    client = PersistentClient(path=persist_dir)
    coll = client.get_collection(collection)
    got = coll.get(include=["metadatas"], limit=sample)
    cats = [(m or {}).get("category", None) for m in (got.get("metadatas") or [])]
    return dict(Counter(cats))

# -----------------------------------------------
# 8) コレクション一覧（既存、そのままでOK）
# -----------------------------------------------
def diag_list_collections(
    persist_dir: str = VECTOR_PERSIST_DIR,
) -> List[Tuple[str, int]]:
    client = PersistentClient(path=persist_dir)
    items: List[Tuple[str, int]] = []
    for coll in client.list_collections():
        try:
            count = coll.count()
        except Exception:
            count = -1
        items.append((coll.name, count))
    return items

def diag_filetypes(persist_dir: str, collection: str, sample: int = 500):
    cl = PersistentClient(path=persist_dir).get_collection(collection)
    got = cl.get(include=["metadatas"], limit=sample)

    # filetype カウント
    fts = [(m or {}).get("filetype", None) for m in (got.get("metadatas") or [])]

    # sourceファイルのstemをざっくり集計
    srcs = []
    for m in (got.get("metadatas") or []):
        src = (m or {}).get("source", "")
        if src:
            srcs.append(Path(src).stem)

    return {
        "filetype": dict(Counter(fts)),
        "top_sources": dict(Counter(srcs).most_common(10))
    }

def assert_collection_exists(persist_dir: str, collection: str):
    names = [c.name for c in PersistentClient(path=persist_dir).list_collections()]
    if collection not in names:
        raise RuntimeError(f"Chroma collection '{collection}' not found in '{persist_dir}'. Existing={names}")


def _search(query: str, category: Optional[str]) -> List[Any]:
    if category == CATEGORY_CHAT_ONLY:
        return []
