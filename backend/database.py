import os
from sqlalchemy import Column, String, Integer, DateTime, Enum as SAEnum
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone
from backend.models.schemas import ModelStatus
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./mlops.db")

engine = create_async_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String(1024), nullable=True)
    status = Column(
        SAEnum(ModelStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=ModelStatus.uploaded,
    )
    port = Column(Integer, nullable=True)
    container_id = Column(String(128), nullable=True)
    file_path = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()