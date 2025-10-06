# ingest_utils.py — docs/直下フォルダをカテゴリに・PDFはPDFPlumberLoader一本化
from config import VECTOR_PERSIST_DIR as CFG_PERSIST_DIR, VECTOR_COLLECTION as CFG_COLLECTION, Category
import os, hashlib, re, csv, statistics, logging, warnings, config
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from contextlib import contextmanager
from collections import Counter
from apisecret import get_secret
load_dotenv()
KNOWN_LABELS = {c.value for c in Category}

# ──────────────────────────────────────────────────────────────────────────────
# 設定/レポート
# ──────────────────────────────────────────────────────────────────────────────
REPORTS_DIR = Path("data/reports")
REPORT_CSV  = REPORTS_DIR / "ingest_report.csv"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_PERSIST_DIR = CFG_PERSIST_DIR
DEFAULT_COLLECTION  = CFG_COLLECTION
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# ──────────────────────────────────────────────────────────────────────────────
# ログ/ユーティリティ
# ──────────────────────────────────────────────────────────────────────────────
def _event_log_ingest(path: Path, reason: str, extra: Dict[str, Any] | None = None):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not REPORT_CSV.exists()
    with open(REPORT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "reason", "file", "extra"])
        w.writerow([datetime.now().isoformat(), reason, str(path), (extra or {})])

def _normalize_text(s: str) -> str:
    s = re.sub(r'\n{2,}', '\n', s)
    s = re.sub(r'^[ \t]*\d{1,3}[ \t]*$', '', s, flags=re.MULTILINE)  # 行単体の数字を削除
    return s

def _quality_ok(pages: List, min_chars=300) -> bool:
    text = "\n".join((getattr(d, "page_content", "") or "") for d in pages)
    if len(text) < min_chars:
        return False
    bad = sum(ord(ch) < 32 and ch not in ("\n", "\r", "\t") for ch in text)
    if bad / max(len(text), 1) > 0.01:
        return False
    tofu = text.count("□") + text.count("�")
    if tofu / max(len(text), 1) > 0.01:
        return False
    lines = [len(l) for l in text.splitlines() if l.strip()]
    if lines and statistics.pstdev(lines) < 8 and statistics.mean(lines) < 20:
        # ほぼ表罫・ノイズのみと推定
        return False
    return True

def _doc_id(d, idx: int) -> str:
    src = str((d.metadata or {}).get("source", ""))
    page = str((d.metadata or {}).get("page", ""))
    head = (d.page_content or "")[:200]
    h = hashlib.sha1(head.encode("utf-8", errors="ignore")).hexdigest()
    return f"{src}|{page}|{idx}|{h}"

def _batched(seq: List, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def _embeddings():
    key = get_secret("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY が未設定です")
    embed_model = get_secret("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    return OpenAIEmbeddings(model=embed_model, api_key=key)

def _silence_noisy_logs():
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("pdfplumber").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message=r".*invalid float value.*", module=r".*pdfminer.*")
    warnings.filterwarnings("ignore", message=r".*alpha value out of range.*", module=r".*pdfminer.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*pdfminer.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r".*pdfminer.*")
_silence_noisy_logs()

@contextmanager
def _quiet_pdfminer():
    logger = logging.getLogger("pdfminer")
    prev_level = logger.level
    try:
        logger.setLevel(logging.ERROR)
        yield
    finally:
        logger.setLevel(prev_level)

# ──────────────────────────────────────────────────────────────────────────────
# ファイル探索
# ──────────────────────────────────────────────────────────────────────────────
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES  = {".pdf"}

def _iter_files(root, suffixes):
    suffixes_lc = tuple(s.lower() for s in suffixes)
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes_lc:
            yield p
# ──────────────────────────────────────────────────────────────────────────────
# ローダー
# ──────────────────────────────────────────────────────────────────────────────
FALLBACK_ENCODINGS = ("utf-8", "cp932", "shift_jis", "euc_jp")

def _load_text_files(paths) -> List:
    out: List[Document] = []
    for p in paths:
        # 1) 自動判定で試す
        try:
            docs = TextLoader(str(p), autodetect_encoding=True).load()
            for d in docs:
                if d.page_content:
                    md = d.metadata or {}
                    md.update({"source": str(p), "filetype": "text"})
                    d.metadata = md
                    out.append(d)
            # 成功したら次のファイルへ
            if docs:
                continue
        except Exception as e1:
            print(f"[TEXT][AUTO-ERR] {p} -> {e1!r}")

        # 2) 代表的な日本語系エンコーディングで再試行
        tried_ok = False
        for enc in FALLBACK_ENCODINGS:
            try:
                docs = TextLoader(str(p), encoding=enc).load()
                for d in docs:
                    if d.page_content:
                        md = d.metadata or {}
                        md.update({"source": str(p), "filetype": "text", "encoding": enc})
                        d.metadata = md
                        out.append(d)
                tried_ok = True
                print(f"[TEXT][OK] {p} (encoding={enc})")
                break
            except Exception as e2:
                print(f"[TEXT][ENC-ERR] {p} ({enc}) -> {e2!r}")

        if tried_ok:
            continue

        # 3) どうしてもダメなら “ignore” でサルベージ（最終手段）
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            content = content.strip()
            if content:
                out.append(Document(
                    page_content=content,
                    metadata={"source": str(p), "filetype": "text", "encoding": "utf-8(ignore)"}
                ))
                print(f"[TEXT][RESCUED] {p} via errors=ignore")
            else:
                print(f"[TEXT][EMPTY] {p}")
        except Exception as e3:
            print(f"[TEXT][FAIL] {p} -> {e3!r}")
    return out


def _load_pdf_files(paths: Iterable[Path]) -> List:
    docs = []
    with _quiet_pdfminer():
        for p in paths:
            try:
                pages = PDFPlumberLoader(str(p)).load()  # 1ページ=1ドキュメント
            except Exception as e:
                _event_log_ingest(p, "pdfplumber_error", {"error": repr(e)})
                continue

            if not _quality_ok(pages, min_chars=300):
                total_chars = sum(len((d.page_content or "")) for d in pages)
                _event_log_ingest(p, "quality_gate_fail", {"pages": len(pages), "chars": total_chars})
                continue

            for d in pages:
                meta = d.metadata or {}
                meta["filetype"] = "pdf"
                meta["title"] = p.stem
                # ページ番号の正規化
                if "page" not in meta and "page_number" in meta:
                    meta["page"] = meta["page_number"]
                d.page_content = _normalize_text(d.page_content or "")
                meta["source"] = str(p)
                d.metadata = meta

            docs.extend(pages)
    return docs

# ──────────────────────────────────────────────────────────────────────────────
# カテゴリ付与
# ──────────────────────────────────────────────────────────────────────────────
def _infer_category_from_root(docs: List, root: str | Path) -> List:
    """    パスのどこかに既知カテゴリ（日本語ラベル）があれば category に設定。
      例:
        docs/お金・ライフプラン/xxx.pdf     → お金・ライフプラン
        docs/web_scraped/健康/yyy.md         → 健康
        docs/外部/人間関係/zzz.pdf           → 人間関係
    見つからなければ従来通り、root直下の最初のフォルダ名をフォールバック。
    """
    rootp = Path(root).resolve()
    for d in docs:
        md = d.metadata or {}
        src = (d.metadata or {}).get("source")
        if not src:
            continue
        try:
            rel = Path(src).resolve().relative_to(rootp)
        except Exception:
            continue
        
        parts = list(rel.parts)  # ["web_scraped","健康","yyy.md"] など
        cat = next((p for p in parts if p in KNOWN_LABELS), None)

        if cat:
            md["category"] = cat
            print(f"[TAG] {cat} <- {rel}")
        else:
            print(f"[WARN] no category match <- {rel}")
           

    return docs

# ──────────────────────────────────────────────────────────────────────────────
# 分割 & ベクタ格納
# ──────────────────────────────────────────────────────────────────────────────
def split_docs(docs: List, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    return splitter.split_documents(docs)

def upsert_to_chroma(chunks: List, persist_dir: Optional[str] = None,
                     collection: Optional[str] = None, batch_docs: int = 64):
    persist_dir = persist_dir or DEFAULT_PERSIST_DIR
    os.makedirs(persist_dir, exist_ok=True)
    print(f"[UPSERT] chunks={len(chunks)} -> dir={persist_dir}, collection={collection}")
    collection  = collection  or DEFAULT_COLLECTION
    vs = Chroma(collection_name=collection, persist_directory=persist_dir,
                embedding_function=_embeddings())
    for i in range(0, len(chunks), batch_docs):
        part = chunks[i:i+batch_docs]
        ids = [_doc_id(d, i+j) for j, d in enumerate(part)]
        vs.add_documents(part, ids=ids)
    getattr(vs, "persist", lambda: None)()
    try:
        # langchain-chroma の VectorStore は close 明示APIがないため参照を切る
        del vs
    except Exception:
        pass
    return True # ← VectorStoreオブジェクト返却ではなく成功boolで十分

# ──────────────────────────────────────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────────────────────────────────────
def build_index(
    source_dir: str = str(config.RAG_DIR),
    include_text: bool = True,
    include_pdf: bool = True,
    infer_category: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    persist_dir: Optional[str] = None,
    collection: Optional[str] = None,
):
    root = Path(source_dir)
    print(f"[INGEST][CONF] root={root.resolve()}")
    if not root.exists():
        print(f"[ERROR] RAG root not found: {root.resolve()}")
        return None

    text_docs: List = []
    pdf_docs:  List = []

    text_paths = list(_iter_files(root, TEXT_SUFFIXES)) if include_text else []
    pdf_paths  = list(_iter_files(root, PDF_SUFFIXES))  if include_pdf  else []
    print(f"[INGEST][FOUND] text_files={len(text_paths)} pdf_files={len(pdf_paths)}")
    cats = Counter()
    for p in text_paths + pdf_paths:
        cats[ str(p.parent) ] += 1
    print("[INGEST][FOUND_BY_DIR]", cats)

    text_docs = _load_text_files(text_paths) if text_paths else []
    pdf_docs  = _load_pdf_files(pdf_paths)   if pdf_paths  else []
    docs = text_docs + pdf_docs
    
    if not docs:
        print(f"[WARN] No documents loaded")
        return None

    if infer_category:
        _infer_category_from_root(docs, root)
        cats = [(d.metadata or {}).get("category") for d in docs if (d.metadata or {}).get("category")]
        print(f"[INGEST][META] categories={Counter(cats)}")

    chunks = split_docs(docs, chunk_size, chunk_overlap)
    print(f"[INGEST][SPLIT] loaded_docs={len(docs)} -> chunks={len(chunks)}")

    persist_dir = persist_dir or os.getenv("VECTOR_PERSIST_DIR", "data/chroma")
    collection  = collection  or os.getenv("VECTOR_COLLECTION", "default_collection")
    print(f"[INGEST][WRITE] persist_dir={persist_dir} collection={collection}")

    res = upsert_to_chroma(chunks, persist_dir, collection)
    print("[INGEST][DONE] persisted.")
    return res
