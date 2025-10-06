# ui.py — 安全起動版（遅延インポート＋白画面回避）/ Shared PersistentClient / 参照折りたたみ対応
import os, io, base64, re, sys, subprocess, logging, traceback, time
from uuid import uuid4
from typing import Optional

import streamlit as st
from PIL import Image

import config
from apisecret import catch_errors

# ─────────────────────────────────────────────────────────────
# PersistentClient共有ヘルパー（utilis/chroma_client が無くても安全に動作）
# ─────────────────────────────────────────────────────────────
try:
    from utilis.chroma_client import get_client  # 推奨：共通クライアント
except Exception:
    def get_client(persist_dir: Optional[str] = None):
        from chromadb import PersistentClient
        persist_dir = persist_dir or os.getenv("VECTOR_PERSIST_DIR", "data/chroma")
        if not hasattr(get_client, "_cache"):
            get_client._cache = {}
        cli = get_client._cache.get(persist_dir)
        if cli is None:
            cli = PersistentClient(path=persist_dir)  # Settingsは渡さない（全箇所同一）
            get_client._cache[persist_dir] = cli
        return cli

# ─────────────────────────────────────────────────────────────
# 安全な遅延インポート（失敗しても画面を出す）
# ─────────────────────────────────────────────────────────────
def _safe_imports():
    try:
        from utilis.router_utils import init_router_components, route_answer
        from utilis.memory_utils import reset_session_history
        from utilis.web_live_chain import make_web_chain
        return {"ok": True,
                "init_router_components": init_router_components,
                "route_answer": route_answer,
                "reset_session_history": reset_session_history,
                "make_web_chain": make_web_chain}
    except Exception as e:
        return {"ok": False, "err": e, "tb": traceback.format_exc()}

imports = _safe_imports()

# ─────────────────────────────────────────────────────────────
# ページ設定・ロガー
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="ココさんのお悩み相談室", page_icon="🤖", layout="centered")
logger = logging.getLogger("streamlit")

# ===== CSS =====
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#fff; }
.block-container {
  max-width: 720px;
  margin-left: auto;
  margin-right: auto;
  border:4px solid #15b15b;
  border-radius:20px;
  padding:12px 14px !important;
}
.stChatMessage { margin-top: 10px; margin-bottom: 10px; }
.assistant-bubble {
  background:#15b15b; color:#fff;
  padding:.6rem .9rem; border-radius:16px;
  display:inline-block; max-width:86%;
  line-height: 1.7; max-width: 38rem;
}
.assistant-bubble p { margin: .6rem 0; }
.hero-wrap { display:flex; justify-content:center; margin:8px 0 6px; }
.hero-img  { width:132px; height:132px; object-fit:cover; border-radius:50%; }
h1 { text-align:center !important; font-weight:800; margin:.4rem 0 .6rem; }
[data-testid="stCaptionContainer"] { margin-top: .1rem; margin-bottom: .8rem; }
@media (max-width: 480px) {
  .hero-img { width:112px; height:112px; }
  h1 { font-size:22px !important; line-height:1.2; white-space:nowrap; }
  .assistant-bubble { max-width: 32rem; }
}
@media (min-width: 481px) {
  h1 { font-size:36px !important; line-height:1.2; }
}
footer { margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ===== 画面ヘッダー（タイトル先に描画） =====
st.title("ココさんのお悩み相談室")
st.caption("なんでも相談してね。ルータでカテゴリ分岐 & RAG つき。")

# ===== 遅延インポートの結果をここで検査（白画面回避） =====
if not imports["ok"]:
    st.error(f"モジュールの読み込みでエラー: {type(imports['err']).__name__}: {imports['err']}")
    with st.expander("スタックトレース（クリックで開く）", expanded=False):
        st.code(imports["tb"])
    st.stop()

# 以降は安全に参照
init_router_components = imports["init_router_components"]
route_answer = imports["route_answer"]
reset_session_history = imports["reset_session_history"]
make_web_chain = imports["make_web_chain"]

# ===== Chroma 診断 =====
with st.sidebar.expander("🔍 Chroma診断", expanded=False):
    import traceback
    persist_dir = os.getenv("VECTOR_PERSIST_DIR", "data/chroma")
    collection  = os.getenv("VECTOR_COLLECTION", "kokosan")
    st.caption(f"dir: `{persist_dir}` / collection: `{collection}`")
    try:
        client = get_client(persist_dir)
        coll = client.get_or_create_collection(collection)
        cnt = coll.count()
        st.success(f"✅ Collection: {coll.name}")
        st.write("📄 Docs:", cnt)
    except Exception as e:
        st.error(f"❌ {type(e).__name__}: {e}")
        st.code("".join(traceback.format_exc()), language="text")

    with st.expander("詳細診断（コレクション一覧/メタ）", expanded=False):
        try:
            client = get_client(persist_dir)
            cols = client.list_collections()
            st.write("collections:", [c.name for c in cols])
            try:
                coll = client.get_or_create_collection(collection)
                got = coll.get(include=["metadatas"], limit=3)
                st.write("sample metadatas:", got.get("metadatas", []) )
            except Exception as e2:
                st.warning(f"meta read error: {e2}")
                st.code("".join(traceback.format_exc()), language="text")
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
                result = subprocess.run(
                    cmd, check=True, capture_output=True, text=True, cwd=os.getcwd()
                )
                logger.info("[INGEST][STDOUT]\n%s", result.stdout)
                if result.stderr:
                    logger.info("[INGEST][STDERR]\n%s", result.stderr)
            # ingest直後の再初期化（ロック/未反映対策）
            time.sleep(0.5)
            try:
                client = get_client(config.VECTOR_PERSIST_DIR)
                coll = client.get_or_create_collection(config.VECTOR_COLLECTION)
                _ = coll.count()
                st.success("ingest 完了 & 再初期化OK。再読み込みします。")
                st.rerun()
            except Exception as e:
                st.error(f"ingest後の再初期化エラー: {type(e).__name__}: {e}")
                st.code("".join(traceback.format_exc()), language="text")
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
            st.code("".join(traceback.format_exc()), language="text")

# ===== 画像ユーティリティ =====
@st.cache_data(show_spinner=False)
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")

def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

AVATAR_PATH = os.path.join("assets", "coco_264.png")

# ===== セッション初期化 =====
ss = st.session_state
if "router_bundle" not in ss:
    ss.router_bundle = init_router_components()
if "messages" not in ss:
    ss.messages = []
if "debug" not in ss:
    ss.debug = False
if "session_id" not in ss:
    ss.session_id = str(uuid4())
router, dest_chains, default_chain, retriever = ss.router_bundle

# ===== アバター（任意） =====
if os.path.exists(AVATAR_PATH):
    try:
        img_b64 = to_b64(load_image(AVATAR_PATH))
        st.markdown(
            f"""<div class="hero-wrap">
                   <img class="hero-img" src="data:image/png;base64,{img_b64}" alt="bot avatar"/>
                </div>""",
            unsafe_allow_html=True
        )
    except Exception:
        pass  # 画像が壊れていてもUIは落とさない

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
if "is_sending" not in ss:
    ss.is_sending = False

if "web_chain" not in ss:
    try:
        ss.web_chain = make_web_chain()
    except Exception:
        ss.web_chain = None

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
    except Exception as e:
        ss.is_sending = False
        st.toast("エラーが発生しました", icon="⚠️")
        raise e

text = st.chat_input("なんでも相談してね", disabled=ss.is_sending)
if text:
    if handle_user_input(text):
        st.rerun()

# ===== メッセージ描画（参考の折りたたみ） =====
def split_body_and_refs(text: str):
    s = text.strip()
    body = s
    refs = []
    m = re.search(r"\n+#\s*参照資料\s*\n(.+)$", s, flags=re.S)
    if m:
        body = s[:m.start()].rstrip()
        refs_block = m.group(1).strip()
        lines = [ln.strip() for ln in refs_block.splitlines() if ln.strip()]
        refs = [ln[2:].strip() if ln.startswith("- ") else ln for ln in lines]
    return body, refs

assistant_avatar_path = AVATAR_PATH if os.path.exists(AVATAR_PATH) else None
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
