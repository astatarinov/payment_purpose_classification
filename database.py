from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Float
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from loans_classification.config import (
    DATABASE_SCHEMA_NAME,
    DATABASE_URL,
    DATABASE_MONITORING_TABLE_NAME,
    DATABASE_NEW_MONITORING_TABLE_NAME,
    USE_DB,
    IS_TEST,
    DATABASE_URL_TEST
)

Base = declarative_base()


class Monitoring(Base):
    __tablename__ = DATABASE_MONITORING_TABLE_NAME
    __table_args__ = {"extend_existing": True, "schema": DATABASE_SCHEMA_NAME}

    id = Column(Integer, primary_key=True, autoincrement=True)
    operation = Column(String)
    text = Column(String)
    main_group = Column(String)
    subgroup = Column(String)
    okveds = Column(String)
    duration = Column(Float)
    status = Column(String)
    data = Column(DateTime)


class NewMonitoring(Base):
    __tablename__ = DATABASE_NEW_MONITORING_TABLE_NAME
    __table_args__ = {"extend_existing": True, "schema": DATABASE_SCHEMA_NAME}

    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_ref = Column(String)
    text = Column(String)
    predicted_class = Column(String)
    proba = Column(Float)
    confident = Column(Boolean)
    duration = Column(Float)
    status = Column(String)
    date = Column(DateTime)


class DataBase:
    def __init__(self) -> None:
        self.monitoring_table_name = DATABASE_MONITORING_TABLE_NAME
        self.new_monitoring_table_name = DATABASE_NEW_MONITORING_TABLE_NAME
        self.schema = DATABASE_SCHEMA_NAME

        if USE_DB:
            if IS_TEST:
                self.engine = create_async_engine(
                    DATABASE_URL_TEST,
                    future=True,
                    pool_timeout=60,
                    pool_recycle=60,
                )
            else:
                self.engine = create_async_engine(
                    DATABASE_URL,
                    future=True,
                    pool_timeout=60,
                    pool_recycle=60,
                )

            self.async_session_maker = async_sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                class_=AsyncSession,
            )

    async def async_init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def add_new_log(self, data: dict | list):
        if not USE_DB:
            return

        async with self.async_session_maker() as session, session.begin():

            if isinstance(data, dict):

                data["date"] = datetime.now()

                new_data = NewMonitoring(
                    payment_ref=data.get("payment_ref"),
                    text=data.get("text"),
                    predicted_class=data.get("predicted_class"),
                    proba=data.get("proba"),
                    confident=data.get("confident"),
                    duration=data.get("duration"),
                    status=data.get("status"),
                    date=data.get("date"),
                )
                session.add(new_data)
            else:
                for row in data:
                    row["date"] = datetime.now()
                records = [NewMonitoring(**row) for row in data]
                session.add_all(records)
