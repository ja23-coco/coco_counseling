# tests/test_apisecret.py
from apisecret import catch_errors

# ▶ テスト用ログバッファ
error_log: list[str] = []

# ▶ Hook 替わりの reporter
def _test_reporter(exc: Exception) -> None:
    error_log.append(str(exc))

# ▶ 強制的に例外を発生させる関数
@catch_errors(reporter=_test_reporter)
def boom() -> None:
    raise ValueError("hook test error")

def test_catch_errors_hook():
    # 例外が呼び出し元に伝搬することを確認
    try:
        boom()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # reporter にエラーメッセージが渡ったか？
    assert any("hook test error" in msg for msg in error_log)
