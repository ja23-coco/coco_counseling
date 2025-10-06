"""
# API キー＆例外統一ハンドラ
"""
import os
from dotenv import load_dotenv
import logging
import functools
from typing import Callable, Any
try:
    import streamlit as st
    _secrets = st.secrets
except Exception:
    _secrets = {}

def get_secret(name: str, default: str | None = None) -> str | None:
    # 1) st.secrets → 2) env → 3) default
    val = (_secrets.get(name) if isinstance(_secrets, dict) else None) or os.getenv(name) or default
    # LangChain等が環境変数を見る想定のため、見つかったらenvへ橋渡し
    if val and not os.getenv(name):
        os.environ[name] = val
    return val

load_dotenv()  # .env を読み込む

def get_key(name: str) -> str:
    key = os.getenv(name)
    if not key:
        raise RuntimeError(f"環境変数 {name} が設定されていません。")
    return key

# エラーメッセージ収集のHook 方式
def _default_reporter(exc: Exception) -> None:
    """標準の例外レポータ: ローカルログに書くだけ"""
    logging.exception("Unhandled error", exc_info=exc)

def catch_errors(reporter: Callable[[Exception], None] | None = None):
    """
    任意の reporter を差し替え可能なデコレータ。
    例: @catch_errors(reporter=sentry_sdk.capture_exception)
    """
    reporter = reporter or _default_reporter

    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                reporter(e)   # Hook 先へ例外を渡す
                raise
        return wrapper
    return decorator
