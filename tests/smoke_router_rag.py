# tests/smoke_router_rag.py —— strict assertions (structured_output 前提)
import os
import sys
from textwrap import shorten

os.environ.setdefault("RAG_AUTO_THRESHOLD", "1")
sys.path.append(".")

from config import Category
from utilis.router_utils import init_router_components, route_answer

QUERIES = {
    Category.WORK:     "上司との関係を改善したい。具体的なコミュニケーション手順を教えて",
    Category.LOVE:     "結婚に踏み切るか迷っています。決断のための整理の仕方は？",
    Category.RELATION: "友人との距離感が難しいです。境界線の引き方を知りたい",
    Category.CAREER:   "将来が不安。今の仕事を続けるべきか転職すべきか判断軸をください",
    Category.HEALTH:   "最近なかなか眠れません。睡眠の質を上げる生活習慣は？",
    Category.MONEY:    "NISAを使って貯蓄から投資へ移りたい。最初の一歩は？",
    Category.OTHER:    "最近のテックニュース、面白い話題ある？",
}

HEADINGS = ["# 共感", "# 要点", "# フォロー質問"]

def hr(label: str):
    print("\n" + "=" * 14 + f" {label} " + "=" * 14)

def _non_empty(text: str, min_chars: int = 40) -> bool:
    return len(str(text).strip()) >= min_chars

def _assert_headings(resp: str):
    for h in HEADINGS:
        assert h in resp, f"出力に見出し『{h}』がありません"
    # 各見出しの直後が空になっていない（ざっくり）
    parts = {}
    for h in HEADINGS:
        idx = resp.find(h)
        assert idx >= 0
        parts[h] = resp[idx + len(h):].strip()
    assert len(parts["# 共感"].splitlines()[0].strip()) > 0, "共感セクションが空です"
    assert "- " in parts["# 要点"], "要点セクションに箇条書き（- ）が見当たりません"
    assert len(parts["# フォロー質問"].splitlines()[0].strip()) > 0, "フォロー質問が空です"

def run_case(router, dest_chains, default_chain, category: Category, text: str):
    hr(f"{category.value} / {text[:10]}...")
    resp = route_answer(
        user_text=text,
        router=router,
        dest_chains=dest_chains,
        default_chain=default_chain,
        retriever=None,      # router側でカテゴリ確定後に必要なら生成
        web_chain=None,      # ここではWeb未使用（雑談でも通電のみ）
        web_allowed=True,
        session_id=f"smoke_{category.name.lower()}",
    )

    resp_str = str(resp)
    preview = shorten(resp_str.replace("\n", " "), width=240, placeholder="…")
    print("RESP:", preview)

    # 1) 出力が空でない
    assert _non_empty(resp_str), f"{category.value}: 出力が短すぎます"

    # 2) 3見出しが必ず含まれる（structured_output→固定レンダリング前提）
    _assert_headings(resp_str)

    # 3) 雑談カテゴリではRAGの参照資料は付かない
    if category == Category.OTHER:
        assert "参照資料" not in resp_str, "OTHER（雑談）で参照資料が付いています"

def main():
    print("[SETUP] init_router_components ...")
    router, dest_chains, default_chain, _ = init_router_components()

    order = [
        Category.WORK, Category.HEALTH, Category.MONEY, Category.CAREER,
        Category.RELATION, Category.LOVE, Category.OTHER
    ]
    for cat in order:
        run_case(router, dest_chains, default_chain, cat, QUERIES[cat])

    print("\n[OK] smoke test (strict) finished.")

if __name__ == "__main__":
    main()
