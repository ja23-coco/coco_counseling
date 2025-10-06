"""
# 共通設定（Google Drive/Chromaパスなど）
# src/config.py
"""
from pathlib import Path
from dotenv import load_dotenv
import os
from enum import Enum
from typing import Dict, List
from apisecret import get_secret

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent   # プロジェクトルート
# GOOGLE_APPLICATION_CREDENTIALS = BASE_DIR / os.getenv("GDRIVE_KEY_PATH")

VECTOR_PERSIST_DIR = os.getenv("VECTOR_PERSIST_DIR", "data/chroma")
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "default_collection")

# === パス設定（RAG 参照先をモードで切替）===

RAG_MODE    = os.getenv("RAG_MODE", "drive_desktop")
GDRIVE_ROOT = Path(os.getenv("GDRIVE_ROOT", ""))
RAG_SUBDIR  = os.getenv("RAG_SUBDIR", "")

# ===カテゴリーチェーン===
CONCISE_SYSTEM = """
あなたは、親切で有能なアシスタントです。ユーザーの質問に対し、要点を的確に捉え、簡潔に日本語で回答してください。共感の言葉を添えるのは良いですが、冗長な繰り返しや、不必要な質問で会話を引き延ばすことは避けてください。
""".strip()

def _resolve_rag_root() -> Path:
    if RAG_MODE == "local":
        # リポジトリ内 docs/ を想定（必要なら変更可）
        return BASE_DIR / "docs"
    if RAG_MODE == "drive_desktop":
        # Google Drive for desktop の “マイドライブ” 実パス
        return GDRIVE_ROOT
    if RAG_MODE == "colab":
        # Colab で drive.mount('/content/drive') 済みの標準パス
        return Path("/content/drive/MyDrive")
    # フォールバック
    return GDRIVE_ROOT

RAG_DIR = _resolve_rag_root() / RAG_SUBDIR

GDRIVE_RAG_DIR = RAG_DIR

CHROMA_DIR = Path(VECTOR_PERSIST_DIR)

# === モデル関連 ===
FT_MODEL = get_secret("FT_MODEL", "gpt-4o-mini") 
EMBED_MODEL = get_secret("EMBED_MODEL", "text-embedding-3-small")

# === カテゴリごとの System Prompt ===
class Category(str, Enum):
    WORK = "仕事・職場"
    LOVE = "恋愛・結婚"
    RELATION = "人間関係全般"
    CAREER = "キャリア・将来不安"
    HEALTH = "生活習慣・健康"
    MONEY = "お金・ライフプラン"
    OTHER = "その他・雑談"
    


CATEGORY_PROMPTS: Dict[Category, str] = {
    Category.WORK:     "あなたは共感的なキャリアカウンセラーです。",
    Category.LOVE:     "あなたは傾聴と具体策のバランスに優れた恋愛アドバイザーです。",
    Category.RELATION: "あなたは相談相手との関係性構築に秀でた人間関係アドバイザーです。",
    Category.CAREER:   "あなたは優秀で相談者に対して親身になったアドバイスをするキャリアコンサルタントです。",
    Category.HEALTH:   "あなたは多くの社会人の健康管理をしているパーソナルトレーナーです。",
    Category.MONEY:    "あなたは具体でわかりやすい説明が得意なファイナンシャルプランナーです。",
    Category.OTHER:    "あなたは一緒にいると安心感をもたらせる包容力があるカウンセラーです。",
}

ROUTE_CATEGORIES: List[str] = [c.value for c in Category]
CATEGORY_CHAT_ONLY = Category.OTHER.value

def get_category_prompt(category: Category) -> str:
    """カテゴリに対応するSystem Promptを取得（未定義時は空文字）"""
    return CATEGORY_PROMPTS.get(category, "")

# === その他 ===
MAX_HISTORY_TOKENS = 4000

# === 便利ユーティリティ（任意だが推奨） ===
def assert_paths_ok() -> None:
    """主要パスが妥当か軽くチェックして警告。致命的なら例外。"""
    # RAG 参照ディレクトリの存在チェック（Driveパス誤りの早期発見）
    if RAG_MODE == "drive_desktop" and not RAG_DIR.exists():
        print(f"[WARN] RAG_DIR not found: {RAG_DIR}  (GDRIVE_ROOT={GDRIVE_ROOT}, RAG_SUBDIR={RAG_SUBDIR})")
    # ベクトル保存ディレクトリは起動側で作成される想定だが、型だけ統一しておく
    if not isinstance(CHROMA_DIR, Path):
        raise TypeError("CHROMA_DIR must be pathlib.Path")
    
