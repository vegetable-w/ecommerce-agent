from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"
    # 4モデルのうち Conversation だけ eager_defaults=True にしている。
    # status/created_at/updated_at は server_default 任せで Python 側に値を持たないが、
    # Conversation は flush 直後に同じインスタンスの status/created_at を読み戻す使い方
    # (tests/test_models.py::test_conversation_defaults_and_autoincrement、および
    # 呼び出し元での「作成直後の会話ステータス確認」)がある。expire_on_commit=False の
    # AsyncSession でこれを素の属性アクセスとしてやると、SQLAlchemy は暗黙のリフレッシュ
    # (同期IO)を試みてしまい "sqlalchemy.exc.MissingGreenlet" になる。eager_defaults は
    # flush 時に awaited な SELECT でこれらの列を先読みし、そのIOを安全な場所に前倒しする。
    #
    # Message/Faq/Ticket には付けていない: created_at などの server_default 列を
    # flush 直後の同一インスタンスから読み返す消費者が(Task 4/11 の計画を含め)存在せず、
    # 付けると全 INSERT ごとに不要な追加 SELECT が発生するだけになる
    # (messages は1ターンあたり3〜4 INSERT走るためコストが無視できない)。
    # もし今後 Message/Faq/Ticket のどれかで flush 直後に server_default 列を読む必要が
    # 出てきたら、症状は同じ "sqlalchemy.exc.MissingGreenlet" として現れる。そのときは
    # その列だけ Python 側でも値を持たせる(例えば ticket_no のように)か、
    # 該当モデルにだけ eager_defaults=True を足すこと。全モデルに機械的に足し返さない。
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum("in_progress", "escalated", "closed"), server_default="in_progress"
    )
    # 会話の要約と、その要約がどの message まで含んでいるか。
    # スライディングウィンドウはこの次の message から原文を並べる。
    # summary は「最近の要約断片から投影した本文」で、断片そのものは
    # conversation_summaries に追記だけしていく(同じ事実を何度も圧縮し直さないため)。
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ConversationSummary(Base):
    """会話の要約の断片。**追記のみで、書き換えない。**

    1 本の要約を毎回書き直す作り方だと、5 回目の版には最初期の発話が 5 回
    圧縮された残骸として入る。どこかの回で注文番号が「重要でない」と判断されて
    落ちた場合、あとから「いつ、なぜ消えたのか」を辿れない。断片を残しておけば、
    ある事実がどの区間の圧縮で失われたかを後から特定できる。

    conversations.summary はここから最近の断片を並べて作った**投影**で、
    prompt に載せるのはそちら。こちらが原本。
    """

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger)
    seq: Mapped[int] = mapped_column(Integer)          # 1 から始まる断片の通し番号
    from_msg_id: Mapped[int] = mapped_column(BigInteger)   # この断片が覆う最初の message
    upto_msg_id: Mapped[int] = mapped_column(BigInteger)   # 最後の message(両端を含む)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool"))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    description: Mapped[str] = mapped_column(Text)
    # 06 章で refund を末尾へ追加した(sql/06-ticket-type.sql)。**順番も DDL と同じにする。**
    # tests/test_models.py が information_schema の COLUMN_TYPE と tuple で突き合わせており、
    # 並びが違うと ORM と実スキーマのずれとして落ちる。
    ticket_type: Mapped[str] = mapped_column(
        Enum("after_sales", "complaint", "inquiry", "refund")
    )
    status: Mapped[str] = mapped_column(
        Enum("pending", "resolved"), server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[int] = mapped_column(Integer, server_default="0")
    prev_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    next_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done"), server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded"), server_default="extracted"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class LowConfidenceQuestion(Base):
    """回答を断った質問のプール。09 章のデータフライホイールの入口になる。

    conversation_id は nullable。会話に紐づかない経路(評価スクリプトなど)からも
    積めるようにするため、および FK 違反で投入自体を失わせないため
    (repository.insert_low_confidence のコメントを参照)。
    """

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), nullable=True
    )
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        Enum("retrieval_low_conf", "self_check", "user_feedback")
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FaithCase(Base):
    """幻覚ケース台帳。評価の実行をまたいで 1 問 1 行で積み上がる。

    レポート JSON は実行のたびに作り直される成果物なので、そこにしか無い幻覚ケースは
    次の実行で消える。同じ問いが何度も幻覚になるのか、それとも一度きりだったのかは
    実行をまたいで数えないと分からないため、判定結果をこちらに残す。

    eval_id が UNIQUE。同じ問いが再び幻覚と判定されても行は増えず、
    repository.upsert_faith_case が最新の実行の内容で上書きして seen_count を増やす。

    citations はその実行でモデルへ渡した Top-K 根拠の**全件**。回答が実際に引用するのは
    そのうち 2〜3 件だが、判定を人が見直すときは「引用しなかった根拠に答えが載っていた」
    ことまで確かめる必要があるので、引用された部分集合ではなく全件を残す。

    server_default は DDL に合わせて宣言してあるが、repository の書き込み経路では
    seen_count / status / 2 つの時刻を Python 側で必ず埋める。commit 直後に同じ
    インスタンスからそれらを読み返すため(戻り値に載せる)、server_default 任せにすると
    暗黙のリフレッシュで "sqlalchemy.exc.MissingGreenlet" になる(Conversation の
    eager_defaults のコメントを参照)。
    """

    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), server_default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("unresolved", "resolved", "no_action_needed"), server_default="unresolved"
    )
    seen_count: Mapped[int] = mapped_column(Integer, server_default="1")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
