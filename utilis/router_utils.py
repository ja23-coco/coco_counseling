# utilis/router_utils.py
# --- 完成版（回答はプレーン出力、ルータのみ structured）---

from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, Callable, List

import os, re, logging, time
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

# プロジェクト内ユーティリティ
import config
from config import Category, get_category_prompt, CATEGORY_CHAT_ONLY, FT_MODEL
from utilis.retriever_utils import build_router_aware_retriever
from utilis.memory_utils import get_session_history
from apisecret import get_secret

# =========================
# ルータ用の構造体（structured）
# =========================

class RouteDecision(BaseModel):
    """Router が出す構造化結果（カテゴリ/安全/外部検索の要否）"""
    category: Category = Field(..., description="判定カテゴリ（7分類）")
    safety: str = Field("ok", description="有害/リスク/OK などの安全判定")
    need_rag: bool = Field(True, description="RAG が有効か（参照資料を使うか）。7) その他・雑談以外は必ず True")
    need_live_web: bool = Field(False, description="ライブWeb検索が必要か、それとも雑談として自分の意見を表明するか判断。7) その他・雑談で使う")


# =========================
# System文の合成（厳守→目安）
# =========================

try:
    from counseling import COUNSELING_SYSTEM, OUTPUT_STYLE as COUNSELING_OUTPUT_STYLE
except Exception:
    COUNSELING_SYSTEM = ""
    COUNSELING_OUTPUT_STYLE = """
【出力スタイル（目安）】
- 相談の核心では必要に応じて要点の箇条書き（最大3）や最後の一問を添えてください。
- 挨拶や短いやり取りでは、自然な一言で簡潔に。
"""

def _compose_system_text(system_rules: str, category_sys: str) -> str:
    parts: List[str] = []
    if COUNSELING_SYSTEM.strip():
        parts.append(COUNSELING_SYSTEM.strip())
    if system_rules.strip():
        parts.append(system_rules.strip())
    if category_sys.strip():
        parts.append(category_sys.strip())
    # “厳守”ではなく“目安”のスタイルを最後に
    if COUNSELING_OUTPUT_STYLE.strip():
        parts.append(COUNSELING_OUTPUT_STYLE.strip())
    return "\n\n".join(parts)


# =========================
# 回答チェーン（プレーン・文字列出力）
# =========================

def build_free_answer_chain(system_text: str,
                            temperature: float = 0.4) -> Runnable:
    """
    回答用のプレーンなチェーン（常に str を返す）。履歴は RunnableWithMessageHistory で注入。
    """
    llm = ChatOpenAI(model= FT_MODEL,
                     temperature=temperature,
                     api_key=get_secret("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder("history"),
        ("human", "{question}{context_block}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def build_category_chains(system_rules: str) -> Dict[Category, Runnable]:
    """
    カテゴリ別のプレーン回答チェーンを構築（キーは Enum のまま）。
    """
    chains: Dict[Category, Runnable] = {}
    for cat in Category:
        cat_prompt = get_category_prompt(cat)  # Enumを渡す前提
        sys_text = _compose_system_text(system_rules, cat_prompt)
        chains[cat] = build_free_answer_chain(
            system_text=sys_text,
            temperature=0.4,
        )
    return chains


# =========================
# ルータ作成（with_structured_output）
# =========================

def _router_llm() -> ChatOpenAI:
    api_key = get_secret("OPENAI_API_KEY")
    model_name = get_secret("FT_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0.0, api_key=api_key)
    
def _build_router_prompt() -> ChatPromptTemplate:
    """
    ルータ専用のプロンプト。出力は RouteDecision で厳密に返す。
    """
    rules = """
あなたはユーザー入力を次の7カテゴリにルーティングし、フラグを判定するアシスタントです。

カテゴリ一覧（厳密一致で返すこと）:
- 仕事・職場
- お金・ライフプラン
- キャリア・将来不安
- 人間関係全般
- 恋愛・結婚
- 生活習慣・健康
- その他・雑談

ルール:
- ⑦その他・雑談は、1〜6のどれにも当てはまらない場合にのみ選ぶ（最後の手段）。
- ⑦では {{"need_rag"}} = false とし、必要に応じて {{"need_live_web"}} を true にできる。
- 必要に応じた判断の基準として次の例示を使ってください。:
  * ユーザーが「あなたはどう思う？」など意見を求める → need_live_web = false
  * ユーザーが最新情報や事実を尋ねる（今日のニュースは？何の日？意味は？など） → need_live_web = true
  * 単なる挨拶や軽い雑談 → need_live_web = false
- ⑦以外（1〜6）のカテゴリでは {{"need_rag"}} = true とする。
- {{"safety"}} は "ok" または "danger" のいずれかのみ。
- 出力は JSON 形式で、次のキーのみを含めること:
  {{"category"}}, {{"need_rag"}}, {{"need_live_web"}}, {{"safety"}}
- 余計な文章や説明は出力しない。true/false は小文字で返す。

出力例:
入力: 「休日に長時間寝てしまう」
出力: {{"category"}}: "生活習慣・健康", {{"need_rag"}}: true, {{"need_live_web"}}: false, {{"safety"}}: "ok"

入力: 「今日は何の日？」
出力: {{"category"}}: "その他・雑談", {{"need_rag"}}: false, {{"need_live_web"}}: true, {{"safety"}}: "ok"

入力: 「投資信託の積立のはじめ方を知りたい」
出力: {{"category"}}: "お金・ライフプラン", {{"need_rag"}}: true, {{"need_live_web"}}: false, {{"safety"}}: "ok"
"""
    return ChatPromptTemplate.from_messages([
        ("system", rules),
        ("human", "{input}")
    ])


# =========================
# 初期化：router / dest_chains / default_chain
# =========================

def init_router_components() -> Tuple[Runnable, Dict[Category, Runnable], Runnable, Any]:
    """
    ルータ（structured）とカテゴリ別の回答チェーン（プレーン）を初期化。
    RAG 検索は route_answer 内で、router のカテゴリ判定後に build_router_aware_retriever() を用いて実行する。
    """
    # Router
    router_prompt = _build_router_prompt()
    router: Runnable = router_prompt | _router_llm().with_structured_output(RouteDecision)

    # 回答チェーン：System共通ルール（出典方針など）
    system_rules = """
- 参照資料が無い時は出典を匂わせない
- 参照資料がある時は、回答本文の最後に「# 参照資料」を付けて箇条書きで提示
- 過度な断定は避け、相談者の文脈に合わせて簡潔に
"""

    # カテゴリ別チェーン（Enumキー）
    dest_chains: Dict[Category, Runnable] = build_category_chains(system_rules)

    # デフォルトチェーン（プレーン）
    default_sys = _compose_system_text(system_rules, "")
    default_chain = build_free_answer_chain(
        system_text=default_sys,
        temperature=0.4,
    )

    # retriever はここでは返さない（route_answer でカテゴリ確定後に生成）
    return router, dest_chains, default_chain, None


# =========================
# テキスト整形ユーティリティ
# =========================

def _strip_guidance(text: str) -> str:
    # ガイド文やデバッグ行が混入した場合の軽いクリーニング
    return text.replace("【出力スタイル（目安）】", "").strip()

def _sanitize_if_no_rag(text: str) -> str:
    # 参照資料が無いのに“参考/参照”が紛れた場合の軽い除去
    return text.replace("# 参照資料", "").strip()

def _remove_memory_apology(text: str, has_history: bool) -> str:
    # 「前回の会話を覚えていません」のような不要な詫びを抑制（簡易）
    if has_history:
        return text.replace("前回の会話を保存していません", "").strip()
    return text


# =========================
# ココの独自意見の検知
# =========================
OPINION_RE = re.compile(
    r"(?:あなた|君|きみ|ココ(?:さん)?)\s*(?:は|って)?\s*(?:どう思|どう考|意見|賛成|反対|好き|嫌い)|"
    r"(?:どう思う|あなたはどう|どっちが|どれが好き)",
    re.IGNORECASE
)
def _is_opinion_query(text: str) -> bool:
    return bool(OPINION_RE.search(text))

SMALLTALK_RE = re.compile(r"(?:こんにちは|こんばんは|おはよう|元気|調子どう|暇|天気)", re.IGNORECASE)
def _is_smalltalk_query(text: str) -> bool:
    return bool(SMALLTALK_RE.search(text))

# =========================
# 本体：route_answer
# =========================

import logging
from types import SimpleNamespace
logger = logging.getLogger("streamlit")

# =========================
# 本体：route_answer（差し替え版）
# =========================
def route_answer(
    user_text: str,
    router: Runnable,
    dest_chains: Dict[Category, Runnable],
    default_chain: Runnable,
    retriever: Any = None,  # 互換のため残す（未使用）
    *,
    web_chain: Optional[Callable[[str], str]] = None,
    web_allowed: bool = True,
    session_id: Optional[str] = None,
) -> str:
    logger = logging.getLogger("streamlit")

    # --- 初期化 ---
    rag_docs: List[Any] = []
    used_rag: bool = False
    context_block: str = ""
    category_val: str = CATEGORY_CHAT_ONLY  # 既定=「その他・雑談」
    parsed = SimpleNamespace(need_rag=False, need_live_web=False, safety="ok", category=category_val)

    # --- Router ---
    try:
        res = router.invoke({"input": user_text}, config={"configurable": {"session_id": session_id}})
        parsed = res if isinstance(res, RouteDecision) else RouteDecision(**res)
        category_val = getattr(parsed.category, "value", str(parsed.category)) or CATEGORY_CHAT_ONLY
    except Exception:
        logger.exception("[ROUTER_ERR]")

    is_chat_only = (category_val == CATEGORY_CHAT_ONLY)

    # --- 雑談系ヒント（意見/小雑談） ---
    opinion_hint = ""
    if is_chat_only:
        if _is_opinion_query(user_text):
            setattr(parsed, "need_live_web", False)
            opinion_hint = (
                "\n\n[OPINION_MODE] ユーザーはあなた自身の考えを求めています。"
                "Web検索や外部参照は使わず、第一人称（私は）で短く・具体例を交えて答えてください。"
                "断定ではなく『一つの考え方として…』『私なら…』のトーンで。"
            )
        elif _is_smalltalk_query(user_text):
            setattr(parsed, "need_live_web", False)
    
    # route_answer 内、Retrieverを作る直前に追加
    use_mmr_env = os.getenv("RAG_USE_MMR", "false").lower() == "true"

    # --- RAG 実行（単一路線に統一） ---
    rag_on    = os.getenv("ENABLE_RAG", "false").lower() == "true"
    force_rag = os.getenv("FORCE_RAG", "false").lower() == "true"
    auto_fb   = os.getenv("RAG_FALLBACK", "true").lower() == "true"

    try:
        if rag_on and ((not is_chat_only and getattr(parsed, "need_rag", False)) or force_rag):
            eff_category = None if force_rag else category_val

            k_val = int(os.getenv("RAG_K", "6"))
            fk_val = int(os.getenv("RAG_FETCH_K", "60")) 

            t0 = time.time()
            retr = build_router_aware_retriever(
                route_category=eff_category,
                persist_dir=config.VECTOR_PERSIST_DIR,
                collection=config.VECTOR_COLLECTION,
                embedding_model=config.EMBED_MODEL,
                k=k_val,
                fetch_k=fk_val,
                use_mmr=use_mmr_env,
                score_threshold=None,
            )
            rag_docs = retr.invoke(user_text)
            latency_ms = int((time.time() - t0) * 1000)
            mode = getattr(retr, "last_mode", "unknown")
            used_rag = len(rag_docs) >= int(os.getenv("RAG_AUTO_THRESHOLD", "1"))

            # 0件時フォールバック（カテゴリ解除）
            fb_info = None     
            if auto_fb and not used_rag and not force_rag:
                t1 = time.time()
                fb_retr = build_router_aware_retriever(
                    route_category=None,
                    persist_dir=config.VECTOR_PERSIST_DIR,
                    collection=config.VECTOR_COLLECTION,
                    embedding_model=config.EMBED_MODEL,
                    k=k_val,
                    fetch_k=fk_val,
                    use_mmr=True,
                    score_threshold=None,
                )
                fb_docs = fb_retr.invoke(user_text)
                fb_latency_ms = int((time.time() - t1) * 1000)
                fb_mode = getattr(fb_retr, "last_mode", "unknown")
                if fb_docs:
                    rag_docs = fb_docs
                    used_rag = True
                fb_info = {
                    "fallback_used": bool(fb_docs),
                    "fallback_docs": len(fb_docs or []),
                    "fallback_mode": fb_mode,
                    "fallback_latency_ms": fb_latency_ms,
                }

            if used_rag:
                context_block = "\n\n" + "\n\n".join(getattr(d, "page_content", str(d)) for d in rag_docs)
            
            logger.info("[RAG] %s", {
                "category": category_val,
                "forced": force_rag,
                "auto_fb": auto_fb,
                "docs": len(rag_docs),
                "used_rag": used_rag,
                "mode": mode,            # 'mmr' / 'sim' / 'sim_fb' / 'unknown'
                "k": k_val,
                "fetch_k": fk_val,
                "latency_ms": latency_ms,
                **({"fallback": fb_info} if fb_info is not None else {})
            })

    except Exception:
        logger.exception("[RAG][ERROR]")
        rag_docs, used_rag, context_block = [], False, ""

    # --- 安全プリチェック ---
    safety_val = str(getattr(parsed, "safety", "ok")).lower()
    if safety_val not in ("ok", "low"):
        out = default_chain.invoke({"question": user_text, "context_block": ""}, config={"configurable": {"session_id": session_id}})
        text = str(getattr(out, "content", out))
        note = "※差し迫った危険がある場合は、身の安全を最優先に地域の相談窓口/緊急連絡先へ連絡してください。\n\n"
        has_hist = len(get_session_history(session_id).messages) > 0
        return note + _remove_memory_apology(_sanitize_if_no_rag(_strip_guidance(text)), has_hist)

    # --- 空入力 ---
    if not user_text.strip():
        return "よかったら、いま気になっていることを一言で教えてね。"

    # --- 雑談カテゴリのみ Web 使用 ---
    if is_chat_only and web_allowed and web_chain is not None and getattr(parsed, "need_live_web", False):
        try:
            web_text = web_chain(user_text)
        except Exception:
            logger.exception("[WEB][ERROR]")
            out = default_chain.invoke({"question": user_text, "context_block": ""}, config={"configurable": {"session_id": session_id}})
            web_text = str(getattr(out, "content", out))
        has_hist = len(get_session_history(session_id).messages) > 0
        return _remove_memory_apology(_strip_guidance(_sanitize_if_no_rag(web_text)), has_hist)

    # --- 回答生成（カテゴリ別チェーン） ---
    try:
        dest_key = parsed.category if isinstance(parsed.category, Category) else Category(category_val)
    except Exception:
        dest_key = Category.OTHER

    # 意見ヒントを context_block に付与（雑談のみ）
    final_context = (context_block + opinion_hint) if opinion_hint else context_block

    out = dest_chains.get(dest_key, default_chain).invoke(
        {"question": user_text, "context_block": final_context},
        config={"configurable": {"session_id": session_id}},
    )
    answer = str(getattr(out, "content", out))

    # --- 参照資料の付与（RAG時のみ） ---
    if used_rag and rag_docs:
        def _fmt(d):
            m = getattr(d, "metadata", {}) or {}
            src = m.get("source", "(no source)")
            pg  = m.get("page")
            cat = m.get("category")
            base = os.path.basename(src)
            return f"{base}" + (f" p.{pg}" if pg is not None else "") + (f" [{cat}]" if cat else "")
        
        lines = [_fmt(d) for d in rag_docs]
        answer += "\n\n# 参照資料\n" + "\n".join(f"- {ln}" for ln in lines)

    # --- 最終クリーンアップ（案内文や“参照”の混入を抑制） ---
    has_hist = len(get_session_history(session_id).messages) > 0
    final_text = _strip_guidance(answer)
    if not used_rag:
        # 参照を使っていない時だけ、誤案内の「# 参照資料」を除去
        final_text = _sanitize_if_no_rag(final_text)

    return _remove_memory_apology(final_text, has_hist)
