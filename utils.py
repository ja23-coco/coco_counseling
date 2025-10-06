# utils.py
from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from config import FT_MODEL
from counseling import COUNSELING_SYSTEM

__all__ = ["init_chain_lcel"]

def init_chain_lcel(system_prompt: str):
    """
    LCEL版の最終回答チェーン（共感→要約→次アクション）。
    - retriever/memoryは外部注入方針（ここでは扱わない）
    """
    base = (
        "あなたは共感×課題解決の相談員です。"
        "1)共感ひと言→2)論点要約→3)次の具体アクション を簡潔に示してください。"
        "相手の気持ちに真っ先に寄り添う表現を最初に添えてください。例: “それはつらかったですね” “すごくわかりますよ”"
    )
    sys_msg = "\n".join([base, COUNSELING_SYSTEM, system_prompt]).strip()
    prompt = CoreChatPromptTemplate.from_messages([
        ("system", sys_msg),
        MessagesPlaceholder("history", optional=True),
        ("human", "質問:\n{question}\n\n参考文脈:\n{context}")
    ])
    llm = ChatOpenAI(model=FT_MODEL, temperature=0.3)
    return prompt | llm | StrOutputParser()

# 旧APIは使わない前提。万一どこかで呼ばれたら気づけるよう明示エラーにします。
def init_chain(*args, **kwargs):
    raise RuntimeError(
        "init_chain は廃止しました。LCELの init_chain_lcel(system_prompt) を使用してください。"
    )   
