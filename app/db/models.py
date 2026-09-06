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
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


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
    ticket_type: Mapped[str] = mapped_column(Enum("after_sales", "complaint", "inquiry"))
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
