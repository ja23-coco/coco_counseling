# utilis/retriever_utils.py — Shared PersistentClient + MMRフラグ + 安全フォールバック
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import os
from collections import Counter
from pathlib import Path

from chromadb.errors import InternalError
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from apisecret import get_secret
from config import VECTOR_PERSIST_DIR, VECTOR_COLLECTION, EMBED_MODEL, CATEGORY_CHAT_ONLY

# ─────────────────────────────────────────────────────────────
# PersistentClient共有ヘルパー（utilis/chroma_client が無くても安全に動作）
# ─────────────────────────────────────────────────────────────
try:
    from utilis.chroma_client import get_client  # 推奨：共通クライアント
except Exception:
    _client_cache: Dict[str, Any] = {}
    def get_client(persist_dir: Optional[str] = None):
        from chromadb import PersistentClient
        persist_dir = persist_dir or os.getenv("VECTOR_PERSIST_DIR", "data/chroma")
        cli = _client_cache.get(persist_dir)
        if cli is None:
            cli = PersistentClient(path=persist_dir)  # Settingsは渡さない（全箇所同一）
            _client_cache[persist_dir] = cli
        return cli

# -----------------------------
# 1) VectorStore 初期化（共有Client）
# -----------------------------
def init_chroma_vectorstore(
    persist_dir: str = VECTOR_PERSIST_DIR,
    collection: str = VECTOR_COLLECTION,
    embedding_model: str = EMBED_MODEL,
):
    emb = OpenAIEmbeddings(model=embedding_model, api_key=get_secret("OPENAI_API_KEY"))
    client = get_client(persist_dir)  # ★ UI と同一インスタンスを共有
    vs = Chroma(
        collection_name=collection,
        embedding_function=emb,
        client=client,                 # ← persist_directory や client_settings は渡さない
    )
    return vs

# ---------------------------------
# 2) Retriever 初期化（MMR/類似度の切替 + 安全フォールバック）
# ---------------------------------
def init_retriever(
    persist_dir: str = VECTOR_PERSIST_DIR,
    collection: str = VECTOR_COLLECTION,
    embedding_model: str = EMBED_MODEL,
    k: int = 6,
    fetch_k: int = 40,
    use_mmr: Optional[bool] = None,        # ← env 既定
    score_threshold: Optional[float] = None,
    filter_category: Optional[str] = None,
):
    """
    use_mmr の既定は環境変数 RAG_USE_MMR（true/false）。MMR経路で InternalError
    （"Nothing found on disk" 等）が出たら自動で similarity にフォールバックします。
    """
    if use_mmr is None:
        use_mmr = os.getenv("RAG_USE_MMR", "false").lower() == "true"

    vs = init_chroma_vectorstore(persist_dir, collection, embedding_model)
    filter_kw = {"filter": {"category": filter_category}} if filter_category else {}

    def _sim_retr():
        return vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, **filter_kw},
        )
    
    class _TaggedSimilarity:
        def __init__(self):
            self._inner = _sim_retr()
            self.last_mode = "sim"
        def invoke(self, q: str):
            self.last_mode = "sim"
            return self._inner.invoke(q)
        # LangChain互換
        get_relevant_documents = invoke

    if use_mmr:
        mmr = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(fetch_k or 0, k), **filter_kw},
        )

        class _SafeMMR:
            def invoke(self, q: str):
                try:
                    docs = mmr.invoke(q)
                    # 命中0や異常時は similarity にフォールバック
                    if not docs:
                        return _sim_retr().invoke(q)
                    return docs
                except InternalError:
                    # HNSW破損等 → similarityへフォールバック
                    fb = _sim_retr()
                    self.last_mode = "sim_fb"
                    return _sim_retr().invoke(q)

            # LangChain互換
            get_relevant_documents = invoke

        return _SafeMMR()

    # use_mmr=False は最初から similarity
    if score_threshold is None:
        return _TaggedSimilarity

    # ✅ similarity + 手動しきい値
    def _get_docs(q: str) -> List[Document]:
        _fetch = max(fetch_k or 0, k * 4, 20)
        pairs = vs.similarity_search_with_relevance_scores(q, k=_fetch, **filter_kw)
        kept = [d for (d, s) in pairs if s is not None and s >= score_threshold]
        if len(kept) < k:
            for d, s in pairs:
                if d not in kept:
                    kept.append(d)
                if len(kept) >= k:
                    break
        return kept[:k]

    class _ManualThresholdRetriever:
        def __init__(self):
            self.last_mode = "sim"
        def get_relevant_documents(self, q: str):
            return _get_docs(q)
        def invoke(self, q: str):
            return _get_docs(q)

    return _ManualThresholdRetriever()

# --------------------------------------------------
# 3) Router→Retriever 連携の薄いヘルパ
# --------------------------------------------------
def build_router_aware_retriever(
    route_category: Optional[str],
    persist_dir: str = VECTOR_PERSIST_DIR,
    collection: str = VECTOR_COLLECTION,
    embedding_model: str = EMBED_MODEL,
    k: int = 6,
    fetch_k: int = 40,
    use_mmr: Optional[bool] = None,       # ← 追加（env既定を尊重）
    score_threshold: Optional[float] = None,
):
    """
    Router 決定（例: 'health', 'money', 'career', ...）をそのままメタフィルタに橋渡し。
    route_category が None/空ならフィルタ無しで広く検索。
    """
    return init_retriever(
        persist_dir=persist_dir,
        collection=collection,
        embedding_model=embedding_model,
        k=k,
        fetch_k=fetch_k,
        use_mmr=use_mmr,                   # ← そのまま渡す
        score_threshold=score_threshold,
        filter_category=route_category,
    )

# ------------------------------------------
# 4) Runnable 化（router_utils から利用）
# ------------------------------------------
def make_retrieve_runnable(retriever):
    """router_utils から .invoke(question) で使えるようにする薄いラッパ。"""
    return RunnableLambda(lambda q: retriever.get_relevant_documents(q))

# ------------------------------------------------
# 5) 参考表示フォーマッタ（UI側“折りたたみ”に載せやすい表記）
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
# 6) 単発クエリ用のユーティリティ
# ----------------------------------------------
def retrieve_texts(
    query: str,
    persist_dir: str = VECTOR_PERSIST_DIR,
    collection: str = VECTOR_COLLECTION,
    embedding_model: str = EMBED_MODEL,
    k: int = 6,
    fetch_k: int = 40,
    use_mmr: Optional[bool] = None,
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
# 7) 診断: 複数クエリでの当たり具合を可視化
# ---------------------------------------------------
def probe_retriever(
    queries: Optional[List[str]] = None,
    persist_dir: str = VECTOR_PERSIST_DIR,
    collection: str = VECTOR_COLLECTION,
    embedding_model: str = EMBED_MODEL,
    k: int = 6,
    fetch_k: int = 40,
    use_mmr: Optional[bool] = None,
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

# -----------------------------------------------
# 8) コレクション診断系（共有Clientで統一）
# -----------------------------------------------
def diag_categories(
    persist_dir: str,
    collection: str,
    sample: int = 200
) -> Dict[str, int]:
    """
    コレクション内ドキュメントの metadata['category'] 分布をざっくり確認。
    例: {'health': 123, 'money': 45, None: 67}
    """
    client = get_client(persist_dir)
    coll = client.get_collection(collection)
    got = coll.get(include=["metadatas"], limit=sample)
    cats = [(m or {}).get("category", None) for m in (got.get("metadatas") or [])]
    return dict(Counter(cats))

def diag_list_collections(
    persist_dir: str = VECTOR_PERSIST_DIR,
) -> List[Tuple[str, int]]:
    client = get_client(persist_dir)
    items: List[Tuple[str, int]] = []
    for coll in client.list_collections():
        try:
            count = coll.count()
        except Exception:
            count = -1
        items.append((coll.name, count))
    return items

def diag_filetypes(persist_dir: str, collection: str, sample: int = 500):
    cl = get_client(persist_dir).get_collection(collection)
    got = cl.get(include=["metadatas"], limit=sample)

    fts = [(m or {}).get("filetype", None) for m in (got.get("metadatas") or [])]

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
    names = [c.name for c in get_client(persist_dir).list_collections()]
    if collection not in names:
        raise RuntimeError(f"Chroma collection '{collection}' not found in '{persist_dir}'. Existing={names}")

# （安全な最小スタブ：他所から参照されてもエラーにならないよう定義）
def _search(query: str, category: Optional[str]) -> List[Any]:
    if category == CATEGORY_CHAT_ONLY:
        return []
    retr = init_retriever()
    return retr.invoke(query)
