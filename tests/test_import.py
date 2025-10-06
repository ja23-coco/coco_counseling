import traceback
try:
    import main           # ここで main.py を読む
    print("✅ main.py imported OK")
except Exception:
    traceback.print_exc()
