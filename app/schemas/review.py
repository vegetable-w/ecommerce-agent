"""査読(データフライホイールの人の判断)の入出力スキーマ。

09 章の閉ループの最後の一段。ここで承認された答えは 03 章の取り込み経路を通って
ナレッジベースへ戻り、次に同じ質問が来たときは検索で当たるようになる。

`review_status` は英語の識別子のまま返し(DDL の ENUM と同じ値)、日本語は
`status_label` に別立てで載せる。画面が識別子を日本語へ訳し直すと、
app/core/labels.py と画面の 2 箇所に表示名が生まれる(02 章の決定)。
"""

from pydantic import BaseModel, Field, field_validator


class ApproveRequest(BaseModel):
    """承認ボタンが送る内容。**モデルの参考回答ではなく、査読者が確定させた答え。**"""

    # min_length は OpenAPI の minLength として表出させるために残し、空白のみの値
    # (min_length を通過してしまう)はこの validator で拒否する
    # (app/schemas/actions.py と同じ規約。新しい書き方を発明しない)。
    #
    # 空の答えをナレッジベースへ書き戻すと、次に同じ質問が来たときに
    # 「根拠は引けるのに中身が無い」chunk が当たる。確信度ゲートは根拠の有無しか
    # 見ないのでそのまま通過し、空の答えが自信を持って返る。
    # bare な .strip() であることが要点で、.strip(" ") へ「明示化」すると
    # U+3000(全角スペース)が抜ける。日本語 IME がそのまま出す空入力なので実際に届く。
    approved_answer: str = Field(
        min_length=1, description="査読者が承認した回答。この文面がナレッジベースへ入る"
    )

    @field_validator("approved_answer")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("空白のみの値は許可されない")
        return v


class ReviewItemOut(BaseModel):
    """一覧の 1 行。ナレッジの穴 1 つ分。"""

    id: int = Field(description="review_queue の id")
    normalized_question: str = Field(description="正規化済みの質問。承認するとこれが chunk の questions になる")
    ai_suggested_answer: str | None = Field(description="モデルの参考回答。査読者はこれを直してから承認する")
    occurrence_count: int = Field(description="この穴に何件の生の質問がまとまったか。そのまま優先度になる")
    review_status: str = Field(description="pending / approved / rejected(DDL の ENUM と同じ英語の識別子)")
    status_label: str = Field(description="review_status の日本語表示名(app/core/labels.py が出所)")
    created_at: str | None = Field(description="穴が作られた時刻(ISO 8601)")


class ReviewListResponse(BaseModel):
    items: list[ReviewItemOut] = Field(description="occurrence_count の降順")


class RawQuestionOut(BaseModel):
    """穴へまとめられた生の質問 1 件。"""

    raw_question: str = Field(description="ユーザーが実際に打った文面(正規化前)")
    source: str = Field(description="retrieval_low_conf / self_check / user_feedback")
    reason: str | None = Field(description="プールへ積んだ理由。経路によっては空")
    created_at: str | None = Field(description="プールへ積まれた時刻(ISO 8601)")
    # 検索を通らなかった経路は null。[] ではないことに意味がある(app/db/models.py)。
    retrieved_chunks: list | None = Field(
        description="そのとき引けた chunk の写し。ナレッジが無いのか、有るのに引けていないのかを見分ける材料"
    )


class ReviewDetailOut(ReviewItemOut):
    approved_answer: str | None = Field(description="承認済みなら、実際に書き戻した答え")
    raws: list[RawQuestionOut] = Field(description="この穴へまとめられた生の質問(古い順)")


class ApproveResponse(BaseModel):
    ok: bool
    chunk_ids: list[int] = Field(description="ナレッジベースへ作られた chunk の id")


class RejectResponse(BaseModel):
    ok: bool
