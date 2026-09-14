"""満足度フィードバック(👍 / 👎)の入出力スキーマ。

09 章のデータフライホイールの 3 つ目の入口(spec §4)。回答を断った質問だけでなく、
**答えたが外していた**質問も改善の燃料にする。前 2 つ(retrieval_low_conf /
self_check)は agent 自身の判断で積むが、こちらは人が押したときにだけ積む。
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FeedbackRequest(BaseModel):
    conversation_id: int = Field(description="評価された回答が属する会話")
    # 検証を Literal に任せるのは、許容値が OpenAPI の enum としてそのまま表に出て、
    # 画面のボタンとの契約になるため(app/schemas/actions.py と同じ規約)。
    # 3 つ目の値を足すのは「どちらでもない」という新しい意味を作ることなので、
    # ここが増えるときは low_confidence_questions の source も一緒に考える。
    rating: Literal["up", "down"] = Field(description="👍 は up、👎 は down")
    # **ユーザーの発話そのもの**。モデルの回答ではない。プールに残るのは
    # 「何を訊かれて答えきれなかったか」であって、こちらが何と答えたかではない
    # (答えの方は会話 ID から辿れる)。
    question: str = Field(min_length=1, description="評価の対象になったターンのユーザー発話")

    # min_length は OpenAPI の minLength として表出させるために残し、空白のみの値
    # (min_length を通過してしまう)はこの validator で拒否する
    # (app/schemas/actions.py と同じ規約。新しい書き方を発明しない)。
    #
    # 空白のみを通すと、何を訊かれたのか分からない行がプールに残る。人がレビュー画面で
    # 見ても直しようがなく、Task 8 以降の正規化と重複排除にも空文字が流れ込む。
    # bare な .strip() であることが要点で、.strip(" ") へ「明示化」すると
    # U+3000(全角スペース)が抜ける。日本語 IME がそのまま出す空入力なので実際に届く。
    @field_validator("question")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("空白のみの値は許可されない")
        return v


class FeedbackResponse(BaseModel):
    ok: bool = Field(description="受け付けたかどうか。画面はこれを見て表示を変えない")
    # 画面のためではなく、**受け入れ検証と運用のため**の値。👍 は log だけで DB へ
    # 保存しないので(spec の明示的な非目標)、「押したのに増えていない」を
    # 仕様どおりだと確かめられる印がここに要る。
    pooled: bool = Field(description="低信頼プールへ積んだかどうか。👍 は常に false")
