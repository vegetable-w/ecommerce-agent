"""画面のボタンから叩く action エンドポイントの入出力スキーマ。

05 章で追加した「有人対応 / チケット作成」の選択肢のうち、backend の処理があるのは
チケット作成だけ(有人対応は本章では画面上の見た目のみ。spec §6.2)。
"""

from typing import Any, Literal

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


class CreateRefundRequest(BaseModel):
    """返金申請フォームの送信内容(06 章)。

    専用のテーブルは作らず tickets を再利用するため、ここで受けるのは
    「どの会話の、どの注文を、どの理由で」の 3 点だけ。ticket_type は画面から
    受け取らない(このエンドポイントは返金以外を作らないので、外から指定できると
    別種のチケットを作る抜け道になる)。
    """

    conversation_id: int = Field(description="チケットをぶら下げる会話。tickets の FK になる")
    order_id: str = Field(min_length=1, description="返金を申請する注文番号")
    # **自由記述にしない。** 理由は画面のドロップダウンの固定分類で、後で集計と
    # オペレーションの振り分けに使う。文言そのものは DB の ENUM ではなく
    # description の中に入るだけなので日本語でよい。
    #
    # 検証を Literal に任せるのは CreateTicketRequest と同じ理由で、許容値の一覧が
    # OpenAPI の enum として表に出て、画面のドロップダウンとの契約になるため。
    reason: Literal[
        # 選択肢の文言はナレッジ側の用語に合わせる。返品ポリシーの節が
        # 「品質不良と保証」「品質不良、誤配送など」と書いているので、申請理由も
        # 同じ語にしておくと、Agent が引用する規約とユーザーが選んだ理由が
        # 同じ言葉で並ぶ(spec は「商品不良」、plan は「品質問題」と揺れていた)。
        "7日以内の自己都合返品", "品質不良", "誤配送", "不要になった", "その他"
    ] = Field(description="返金理由。画面のドロップダウンと同じ固定の選択肢")

    # min_length=1 は空白のみの注文番号を通す(CreateTicketRequest の description と
    # 同じ穴)。通すと、どの注文の申請か分からない返金チケットが残り、運用側は
    # 会話を遡らないと対応できない。bare な .strip() であることが要点で、
    # .strip(" ") へ「明示化」すると U+3000(全角スペース)が抜ける。
    @field_validator("order_id")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("空白のみの値は許可されない")
        return v


class CreateRefundResponse(BaseModel):
    ticket_no: str = Field(description="採番されたチケット番号。画面にそのまま出す")
    # create-ticket と同じく、会話が「オペレーター対応」へ移ったことを伝える
    # 日本語ラベル。文言は labels から引く(app/api/actions.py 参照)。
    status: str = Field(description="返金申請後の会話状態(日本語の表示ラベル)")


class ResumeRequest(BaseModel):
    """中断した turn の再開(06 章)。画面が注文を 1 件選んだときに送る。

    conversation_id は checkpointer の thread_id そのもので、どの中断を再開するかは
    これだけで決まる(中断は 1 会話につき 1 つしか待たない)。
    """

    conversation_id: int = Field(description="再開する会話。checkpointer の thread_id になる")
    # **型で縛らない。** 画面が一覧の要素をそのまま返して {"order_id": "1001"} で来ることも、
    # 番号だけを "1001" や 1001 で返すこともある。どれで来ても同じ注文に落とすのは
    # fetch_order の _normalize_order_id の仕事で、ここで形を決め打ちにすると
    # 画面の実装を 1 通りに縛った上、解釈できない値を 422 として弾いてしまう
    # (fetch_order は読めない値でも会話を止めずに聞き直す設計になっている)。
    #
    # 既定値を置かないので、key 自体の欠落は 422 になる(None は明示すれば通る)。
    value: Any = Field(description="ユーザーが選んだ値。注文番号または一覧の要素そのもの")
