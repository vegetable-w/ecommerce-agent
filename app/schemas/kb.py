"""/api/kb の入出力スキーマ。

content_type と source を Literal / Enum にして pydantic に弾かせることはしない。
それをやると不正値が 422 になるが、この画面の「未登録の資料名」「対象外の content_type」は
**入力の意味の誤り**であって、一律 400 で返すと決めている(app/api/kb.py 側で検査する)。
検査場所を 2 か所に分けないこと。
"""

from pydantic import BaseModel, Field


class PreviewRequest(BaseModel):
    """text + content_type か、source(登録済み資料名)のどちらかを受ける。"""

    text: str | None = Field(default=None, description="その場で貼り付けた Markdown 本文")
    content_type: str | None = Field(default=None, description="faq / policy / manual / mined")
    source: str | None = Field(default=None, description="data/kb の登録済み資料名")


class IngestRequest(PreviewRequest):
    vectorize: bool = Field(
        default=False,
        description="MySQL へ保存したあと続けてベクトル化するか。false なら pending のまま残す",
    )


class ChunkPreview(BaseModel):
    index: int
    category: str
    section_path: str
    questions: str
    answer: str
    chars: int
    is_key_clause: bool
    is_table: bool
    # None は「既存 chunk を読めなかったので判定していない」。False(重複でない)とは違う
    is_duplicate: bool | None = None


class PreviewResponse(BaseModel):
    source: str | None
    content_type: str
    total: int
    key_clauses: int
    tables: int
    duplicates: int | None
    duplicate_check: str = Field(description="ok なら既存との突き合わせ済み、unavailable なら未判定")
    chunks: list[ChunkPreview]


class IngestResponse(BaseModel):
    source: str | None
    content_type: str
    inserted: int
    skipped: int
    ids: list[int]
    knowledge_total: int | None
    vectorized: int | None = None
    message: str


class VectorizeResponse(BaseModel):
    vectorized: int
    milvus_count: int | None
    pending: int | None
    message: str


class SearchRequest(BaseModel):
    query: str | None = Field(default=None, description="検索文")
    top_k: int | None = None
    min_score: float | None = None


class SearchHit(BaseModel):
    id: int
    score: float
    question: str
    answer: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    min_score: float
    hits: list[SearchHit]


class StagingRow(BaseModel):
    id: int
    batch_no: str
    source_ref: str | None
    question: str
    answer: str
    status: str


class StagingResponse(BaseModel):
    rows: list[StagingRow]
    stats: dict | None
