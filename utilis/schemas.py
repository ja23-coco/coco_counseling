# utilis/schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

class SourceItem(BaseModel):
    title: Optional[str] = None
    page: Optional[int] = None
    source: Optional[str] = None  # path or url

class AnswerSchema(BaseModel):
    """回答を構造化（共感/要点/フォロー質問）"""
    empathy: str = Field(..., description="1〜2文。相談者に寄り添う共感文（日本語）")
    points: List[str] = Field(str, min_items=1, max_items=3)  # 箇条書き（最大3）
    followup: str = Field(..., description="次に進むための短い質問を1つだけ（日本語）")
    # 任意：必要なら後で使えるよう参照情報も保持可
    sources: Optional[List[SourceItem]] = None
