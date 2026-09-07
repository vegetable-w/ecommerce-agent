"""画面のボタンから叩く action エンドポイントの入出力スキーマ。

05 章で追加した「有人対応 / チケット作成」の選択肢のうち、backend の処理があるのは
チケット作成だけ(有人対応は本章では画面上の見た目のみ。spec §6.2)。
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CreateTicketRequest(BaseModel):
    conversation_id: int = Field(description="チケットをぶら下げる会話。tickets の FK になる")
    description: str = Field(min_length=1, description="ユーザーが入力した問い合わせ内容")
    # DB の ENUM('after_sales','complaint','inquiry') と同じ英語の識別子。
    # 画面には app/core/labels.py の TICKET_TYPE で日本語を出す(日本語 → 識別子の
    # 対応表は用意しない。日本語を DB へ書く経路を作らないという 02 章の決定)。
    #
    # 検証を Literal に任せるのは、許容値の一覧が OpenAPI の enum としてそのまま
    # 表に出るため。自前の if 文で弾くと画面側は許容値を推測するしかなくなる。
    ticket_type: Literal["after_sales", "complaint", "inquiry"] = Field(
        description="問い合わせ種別。DDL の ENUM と同じ英語の識別子"
    )

    # min_length は OpenAPI の minLength として表出させるために残し、空白のみの値
    # (min_length を通過してしまう)はこの validator で拒否する
    # (app/schemas/agent.py の AgentRequest と同じ規約。新しい書き方を発明しない)。
    #
    # 空白のみを通すと、何の苦情か分からないチケットが tickets に残り、運用側は
    # 会話を遡らないと対応できない。bare な .strip() であることが要点で、
    # .strip(" ") へ「明示化」すると U+3000(全角スペース)が抜ける。日本語 IME が
    # そのまま出す空入力なので、実際に届く。
    @field_validator("description")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("空白のみの値は許可されない")
        return v


class CreateTicketResponse(BaseModel):
    ticket_no: str = Field(description="採番されたチケット番号。画面にそのまま出す")
    # 会話が「オペレーター対応」へ移ったことを画面へ伝える日本語ラベル。
    # 文言は labels から引く(app/api/actions.py 参照)。既定値をここに書くと
    # 表示名の出所が 2 つになる。
    status: str = Field(description="チケット作成後の会話状態(日本語の表示ラベル)")
