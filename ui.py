# ui.py — 白画面完全回避版（超安全起動）/ Shared PersistentClient / 参照折りたたみ
import os, io, base64, re, sys, subprocess, logging, traceback, time
from uuid import uuid4
from typing import Optional, Tuple, List

import streamlit as st
st.set_page_config(page_title="ココさんのお悩み相談室", page_icon="🤖", layout="centered")
logger = logging.getLogger("streamlit")

# ─────────────────────────────────────────────────────────────
# まずは “画面を出すために必要最小限” だけ読み込む。以降は安全にロード。
# ─────────────────────────────────────────────────────────────
def _fatal_panel(title: str, err: Exception, tb: str):
    st.error(f"{title}: {type(err).__name__}: {err}")
    with st.expander("スタックトレース（クリックで開く）", expanded=False):
        st.code(tb)
    st.stop()

# 画像系は無くても動くようにする
try:
    from PIL import Image
except Exception as e:
    Image = None  # 画像は後でスキップ

# `config` は無いと先に進めないので “可視化して止める”
try:
    import config
except Exception as e:
    _fatal_panel("config の読み込みでエラー", e, traceback.format_exc())

# `apisecret` は無い場合に備えてフォールバックを用意
try:
    from apisecret import catch_errors
except Exception:
    def catch_errors():
        def _wrap(fn):
            def _in(*a, **kw):
                try:
                    return fn(*a, **kw)
                except Exception:
                    st.error("実行時エラーが発生しました。ログを確認してください。")
                    st.code(traceback.format_exc())
                    raise
            return _in
        return _wrap

# ─────────────────────────────────────────────────────────────
# PersistentClient 共有（utilis/chroma_client が無くても動くフォールバック）
# ─────────────────────────────────────────────────────────────
try:
    from utilis.chroma_client import get_client  # 推奨の共通クライアント
except Exception:
    def get_client(persist_dir: Optional[str] = None):
        try:
            from chromadb import PersistentClient
        except Exception as e:
            _fatal_panel("Chroma の読み込みでエラー", e, traceback.format_exc())
        persist_dir = persist_dir or os.getenv("VECTOR_PERSIST_DIR", "data/chroma")
        if not hasattr(get_client, "_cache"):
            get_client._cache = {}
        cli = get_client._cache.get(persist_dir)
        if cli is None:
            cli = PersistentClient(path=persist_dir)  # Settings は渡さない（全箇所同一）
            get_client._cache[persist_dir] = cli
        return cli

# ─────────────────────────────────────────────────────────────
# Router / Web連携など “重い依存” は遅延インポートして可視化
# ─────────────────────────────────────────────────────────────
def _safe_imports():
    try:
        from utilis.router_utils import init_router_components, route_answer
        from utilis.memory_utils import reset_session_history
        try:
            from utilis.web_live_chain import make_web_chain
        except Exception:
            make_web_chain = None  # Web無しでも動く
        return {"ok": True,
                "init_router_components": init_router_components,
                "route_answer": route_answer,
                "reset_session_history": reset_session_history,
                "make_web_chain": make_web_chain}
    except Exception as e:
        return {"ok": False, "err": e, "tb": traceback.format_exc()}

imports = _safe_imports()

# ===== CSS（軽量） =====
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#fff; }
.block-container {
  max-width: 720px; margin: 0 auto;
  border:4px solid #15b15b; border-radius:20px; padding:12px 14px !important;
}
.stChatMessage { margin: 10px 0; }
.assistant-bubble {
  background:#15b15b; color:#fff; padding:.6rem .9rem; border-radius:16px;
  display:inline-block; max-width:38rem; line-height:1.7;
}
.hero-wrap { display:flex; justify-content:center; margin:8px 0 6px; }
.hero-img  { width:132px; height:132px; object-fit:cover; border-radius:50%; }
h1 { text-align:center !important; font-weight:800; margin:.4rem 0 .6rem; }
@media (max-width: 480px) { .hero-img { width:112px; height:112px; } h1 { font-size:22px !important; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* タイトルの折返しとクリッピング防止 */
h1, .stMarkdown h1 {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
  line-height: 1.25 !important;
  margin-top: 0.75rem !important;   /* 上に余白を少し追加 */
}

/* 上端クリップ対策でメインコンテナに余白 */
.block-container {
  padding-top: 1.0rem !important;
}

/* カードや枠の上ボーダーが h1 に重ならないよう適度な余白 */
header, [data-testid="stHeader"] {
  z-index: 0 !important;
}

/* 参照折りたたみの中身が長いときのはみ出し防止 */
details, summary {
  overflow-wrap: anywhere !important;
}
</style>
""", unsafe_allow_html=True)

# ===== タイトルは “必ず” ここまでに出す =====
st.title("ココさんのお悩み相談室")
st.caption("なんでも相談してね。ルータでカテゴリ分岐 & RAG つき。")

# ===== 遅延インポート失敗を可視化（白画面回避の要） =====
if not imports["ok"]:
    _fatal_panel("モジュールの読み込みでエラー", imports["err"], imports["tb"])

# 以降は安全に参照
init_router_components = imports["init_router_components"]
route_answer = imports["route_answer"]
reset_session_history = imports["reset_session_history"]
make_web_chain = imports["make_web_chain"]

# ===== Chroma 診断（共有 PersistentClient 利用） =====
with st.sidebar.expander("🔍 Chroma診断", expanded=False):
    persist_dir = os.getenv("VECTOR_PERSIST_DIR", "data/chroma")
    collection  = os.getenv("VECTOR_COLLECTION", "kokosan")
    st.caption(f"dir: `{persist_dir}` / collection: `{collection}`")
    try:
        client = get_client(persist_dir)
        coll = client.get_or_create_collection(collection)
        st.success(f"✅ Collection: {coll.name}")
        st.write("📄 Docs:", coll.count())
    except Exception as e:
        st.error(f"❌ {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
    with st.expander("詳細診断（コレクション一覧/メタ）", expanded=False):
        try:
            cols = get_client(persist_dir).list_collections()
            st.write("collections:", [c.name for c in cols])
            try:
                got = get_client(persist_dir).get_or_create_collection(collection).get(include=["metadatas"], limit=3)
                st.write("sample metadatas:", got.get("metadatas", []) )
            except Exception as e2:
                st.warning(f"meta read error: {e2}")
                st.code(traceback.format_exc())
        except Exception as e0:
            st.warning(f"list_collections error: {e0}")

# ===== ingest（Cloudボタン） =====
with st.sidebar:
    if st.button("🔧 Cloudで初回セットアップ（ingest）"):
        cmd = [
            sys.executable, "ingest.py",
            "--source_dir", "docs",
            "--include_text", "--include_pdf",
            "--infer_category",
            "--persist_dir", config.VECTOR_PERSIST_DIR,
            "--collection",  config.VECTOR_COLLECTION,
        ]
        try:
            with st.spinner("ベクトルDBを作成中…"):
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=os.getcwd())
                logger.info("[INGEST][STDOUT]\n%s", result.stdout)
                if result.stderr: logger.info("[INGEST][STDERR]\n%s", result.stderr)
            time.sleep(0.5)
            try:
                coll = get_client(config.VECTOR_PERSIST_DIR).get_or_create_collection(config.VECTOR_COLLECTION)
                _ = coll.count()
                st.success("ingest 完了 & 再初期化OK。再読み込みします。")
                st.rerun()
            except Exception as e:
                st.error(f"ingest後の再初期化エラー: {type(e).__name__}: {e}")
                st.code(traceback.format_exc())
        except subprocess.CalledProcessError as e:
            st.error(f"ingestでエラー: {e.returncode}")
            logger.exception("[INGEST][ERROR] %s", e.stderr or e.stdout)

# ===== クリーン再構築 =====
with st.sidebar.expander("🧹 クリーン再構築", expanded=False):
    persist_dir = os.getenv("VECTOR_PERSIST_DIR", "data/chroma")
    if st.button("既存DBを削除して再ingest"):
        import shutil
        try:
            shutil.rmtree(persist_dir, ignore_errors=True)
            st.info("既存DBを削除しました。続けて ingest を実行してください。")
        except Exception as e:
            st.error(f"削除でエラー: {e}")
            st.code(traceback.format_exc())

# ===== 画像ユーティリティ（Pillow無くても動く） =====
@st.cache_data(show_spinner=False)
def load_image(path: str):
    if Image is None or not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None

def to_b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

AVATAR_PATH = os.path.join("assets", "coco_264.png")
img = load_image(AVATAR_PATH)
if img is not None:
    try:
        st.markdown(
            f"""<div class="hero-wrap">
                   <img class="hero-img" src="data:image/png;base64,{to_b64(img)}" alt="bot avatar"/>
                </div>""",
            unsafe_allow_html=True
        )
    except Exception:
        pass

# ===== セッション初期化 =====
ss = st.session_state
if "router_bundle" not in ss: ss.router_bundle = init_router_components()
if "messages" not in ss:      ss.messages = []
if "debug" not in ss:         ss.debug = False
if "session_id" not in ss:    ss.session_id = str(uuid4())
router, dest_chains, default_chain, retriever = ss.router_bundle

# ===== 操作ヘッダー =====
left, right = st.columns([1,1])
with left:
    if st.button("会話をクリア", use_container_width=True):
        ss.messages.clear(); ss.pop("last_meta", None)
        reset_session_history(ss.session_id)
        ss.router_bundle = init_router_components()
        ss.session_id = str(uuid4())
        st.rerun()
with right:
    ss.debug = st.toggle("デバッグ表示", value=ss.debug)

# ===== 入力エリア =====
if "is_sending" not in ss: ss.is_sending = False
if "web_chain" not in ss:
    ss.web_chain = imports["make_web_chain"]() if imports["make_web_chain"] else None

use_live_web = st.sidebar.toggle("リアルタイムWeb検索（カテゴリ⑦）", value=True)

@catch_errors()
def handle_user_input(user_text: str):
    ss.is_sending = True
    ss.messages.append({"role": "user", "content": user_text})
    try:
        with st.spinner("考え中…"):
            router, dest_chains, default_chain, retriever = ss.router_bundle
            answer = route_answer(
                user_text, router, dest_chains, default_chain, retriever,
                web_chain=(ss.web_chain if (use_live_web and ss.web_chain) else None),
                web_allowed=bool(use_live_web and ss.web_chain),
                session_id=ss.session_id
            )
        ss.messages.append({"role": "assistant", "content": answer})
        ss.is_sending = False
        return True
    except Exception:
        ss.is_sending = False
        st.toast("エラーが発生しました", icon="⚠️")
        st.code(traceback.format_exc())
        raise

text = st.chat_input("なんでも相談してね", disabled=ss.is_sending)
if text:
    if handle_user_input(text):
        st.rerun()

# ===== メッセージ描画（参考の折りたたみ） =====
REF_HEAD_RE = re.compile(
    r'^\s*(?:#\s*参照資料|参考資料|参考文献|参考|References)\s*[:：]?\s*$',
    re.MULTILINE
)
BULLET_RE = re.compile(r'^\s*(?:[-*・]|[0-9０-９]+\.)\s+.+$', re.MULTILINE)

def split_body_and_refs(text: str) -> Tuple[str, List[str]]:
    """
    本文と参照を分離する。
    1) 「# 参照資料」「参考資料」「参考」「参考文献」「References」いずれかを見出しとみなす
    2) 見出しが無い場合、末尾の連続した箇条書きブロックを参照として抽出（ヒューリスティック）
    3) 箇条書きの重複は除去
    """
    if not text:
        return text, []
    
    m = REF_HEAD_RE.search(text)
    refs: List[str] = []
    body = text
    
    if m:
        split_idx = m.start()
        body = text[:split_idx].rstrip()
        refs_block = text[m.end():].strip()
        # 見出し直下の箇条書きだけを抽出
        bullets = BULLET_RE.findall(refs_block)
        if bullets:
            # findall は行全体ではないケースがあるので、行ごと抽出で再取得
            lines = [ln.strip() for ln in refs_block.splitlines() if BULLET_RE.match(ln)]
            refs = lines
        else:
            # 箇条書きでなくても行単位で格納
            refs = [ln.strip() for ln in refs_block.splitlines() if ln.strip()]
    else:
        # 2) 見出しがない場合：末尾の連続した箇条書きブロックを抽出
        # 末尾から走査して「箇条書きが連続している範囲」を切り出す
        lines = text.rstrip().splitlines()
        tail = []
        for ln in reversed(lines):
            if BULLET_RE.match(ln):
                tail.append(ln.strip())
                continue
            # 箇条書きが一度でも始まった後で非箇条行に当たったら終了
            if tail:
                break        
        if tail:
            tail_block = list(reversed(tail))
            body = text[: text.rfind("\n".join(tail_block))].rstrip()
            refs = tail_block

    seen = set()
    uniq_refs = []
    for r in refs:
        if r not in seen:
            uniq_refs.append(r)
            seen.add(r)

    return body, uniq_refs

assistant_avatar_path = AVATAR_PATH if img is not None else None
for msg in ss.messages:
    role = msg["role"]
    body, refs = split_body_and_refs(msg["content"]) if role == "assistant" else (msg["content"], [])
    with st.chat_message(role, avatar=(assistant_avatar_path if role=="assistant" else None)):
        if role == "assistant":
            st.markdown(f'<div class="assistant-bubble">{body}</div>', unsafe_allow_html=True)
            if refs:
                with st.expander("参考（クリックで表示）", expanded=False):
                    for i, r in enumerate(refs, 1):
                        st.markdown(f"{i}. `{r}`")
        else:
            st.markdown(body)
