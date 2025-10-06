# utilis/chroma_client.py
from chromadb import PersistentClient
import os

# StreamlitでもPython単体でも使える軽量キャッシュ
_client_cache = {}

def get_client(persist_dir: str | None = None) -> PersistentClient:
    persist_dir = persist_dir or os.getenv("VECTOR_PERSIST_DIR", "data/chroma")
    key = persist_dir
    cli = _client_cache.get(key)
    if cli is None:
        cli = PersistentClient(path=persist_dir)  # Settingsは渡さない（全箇所同一）
        _client_cache[key] = cli
    return cli
