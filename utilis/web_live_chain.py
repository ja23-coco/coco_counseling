import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


def _format_refs(items: List[Dict]) -> str:
    """末尾の『参考: [1] タイトル（ドメイン）, ...』を作る。URLは出さない方針。"""
    refs = []
    for i, r in enumerate(items, 1):
        url = (r.get("url") or "").strip()
        host = url.split("/")[2] if "://" in url else url
        title = (r.get("title") or host or "Web").strip()
        if host:
            refs.append(f"[{i}] {title}（{host}）")
        else:
            refs.append(f"[{i}] {title}")
    return "参考: " + ", ".join(refs) if refs else ""

def make_web_chain(model: str | None = None, max_results: int = 5):
    """
    ⑦その他・雑談 用のリアルタイムWebチェーン。
    - まず検索を確実に実行 → 上位ヒットを短く要約 → 最後に『参考: …』を付与
    - TAVILY_API_KEY 未設定なら穏やかにフォールバック
    """
    chat_model = model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=chat_model, temperature=0.2)

    # Tavilyキーがなければフォールバック関数を返す
    if not os.getenv("TAVILY_API_KEY"):
        def run_no_search(query: str) -> str:
            prompt = (
                "次の質問に、一般的な知見の範囲で簡潔に日本語で答えてください。"
                "確信が持てない点は『わからない』と述べ、最新情報は公式サイトを案内してください。\n\n"
                f"質問: {query}"
            )
            return llm.invoke(prompt).content
        return run_no_search

    search = TavilySearchResults(max_results=max_results)

    def run(query: str) -> str:
        try:
            # 1) 常に検索を実行（ツール自動判断に依存しない）
            results: List[Dict] = search.invoke({"query": query}) or []
            top = results[:max_results]

            # 2) LLMに要約させるための短いコンテキストを作成
            ctx_parts = []
            for i, r in enumerate(top, 1):
                title = (r.get("title") or "").strip()
                snippet = (r.get("content") or "").strip().replace("\n", " ")
                snippet = snippet[:700]  # 取り込みすぎない
                url = (r.get("url") or "").strip()
                ctx_parts.append(f"[{i}] {title}\n{snippet}\nURL: {url}")
            context = "\n\n".join(ctx_parts) if ctx_parts else "(ヒットなし)"

            # 3) 回答生成（日本語・簡潔・出典番号付き）
            prompt = (
                "あなたはユーザーの質問に対し、与えられた検索要約だけを根拠に、"
                "日本語で簡潔・正確に回答します。憶測は避け、不確かな点は『わからない』と述べてください。\n\n"
                f"質問: {query}\n\n"
                "# 検索要約（番号付き）\n"
                f"{context}\n\n"
                "出力要件:\n"
                "1) 箇条書きまたは短い段落で端的に答える\n"
                "2) 回答の最後に、参照した番号に対応する『参考: [1] タイトル（ドメイン）, ...』を1行で付ける\n"
                "3) 本文にはURLを直接書かない\n"
            )
            answer = llm.invoke(prompt).content
            refs = _format_refs(top)
            if refs and "参考:" not in answer:
                answer = f"{answer.rstrip()}\n\n{refs}"
            return answer

        except Exception as e:
            # 4) フォールバック（検索失敗時）
            fallback = (
                "今はリアルタイム検索にアクセスできませんでした。"
                "一般的な知見の範囲でお答えします。必要なら公式サイトもあわせて確認してください。\n\n"
                f"質問: {query}\n\n"
                f"(詳細: {e})"
            )
            return llm.invoke(fallback).content

    return run