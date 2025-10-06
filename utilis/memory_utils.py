
from __future__ import annotations
from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

__all__ = ["get_session_history", "reset_session_history", "init_memory"]

# インメモリのセッション履歴ストア（必要なら将来Redis等に置換）
_STORE: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """session_id に紐づく会話履歴（Message型）を返す。無ければ作る。"""
    if session_id not in _STORE:
        _STORE[session_id] = ChatMessageHistory()
    return _STORE[session_id]

def reset_session_history(session_id: str) -> None:
    """指定セッションの履歴を破棄。"""
    if session_id in _STORE:
        del _STORE[session_id]

def init_memory(*_args, **_kwargs):
    """
    廃止: Summary系メモリ（ConversationSummaryBufferMemory）は使用しません。
    LCELの RunnableWithMessageHistory + get_session_history(session_id) を使用してください。
    """
    raise RuntimeError(
        "init_memory は廃止しました。RunnableWithMessageHistory と "
        "get_session_history(session_id) を使用してください。"
    )
