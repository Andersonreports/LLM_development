from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import settings

engine = create_engine(settings.db_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    accession: Mapped[str] = mapped_column(String(64), unique=True)
    case_type: Mapped[str] = mapped_column(String(16))           # single | trio | couple
    status: Mapped[str] = mapped_column(String(32), default="created")
    # created -> queued -> processing -> review -> approved | rejected | failed
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    rejected_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    rejected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    referring_clinician: Mapped[str | None] = mapped_column(String(128), nullable=True)
    hospital: Mapped[str | None] = mapped_column(String(256), nullable=True)
    indication: Mapped[str | None] = mapped_column(String(256), nullable=True)
    clinical_history: Mapped[str] = mapped_column(Text, default="")
    family_history: Mapped[str | None] = mapped_column(Text, nullable=True)
    consanguinity: Mapped[str | None] = mapped_column(String(16), nullable=True)  # yes | no | unknown
    manual_hpo: Mapped[list] = mapped_column(JSON, default=list)
    hpo_terms: Mapped[list] = mapped_column(JSON, default=list)
    phenotype_ai: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    seq_total_gb: Mapped[str | None] = mapped_column(String(32), nullable=True)   # "Total Read Generated"
    seq_q30: Mapped[str | None] = mapped_column(String(32), nullable=True)        # "Data >= Q30"
    coverage_path: Mapped[str | None] = mapped_column(String(512), nullable=True)  # optional gene-coverage TSV
    consent_secondary: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_carrier: Mapped[bool] = mapped_column(Boolean, default=False)

    qc: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ai_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ai_usage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    rule_flag: Mapped[str | None] = mapped_column(String(16), nullable=True)
    ai_flag: Mapped[str | None] = mapped_column(String(16), nullable=True)
    final_flag: Mapped[str | None] = mapped_column(String(16), nullable=True)
    review: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # reviewer edits

    analyst: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    report_version: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)

    samples: Mapped[list["Sample"]] = relationship(back_populates="case", cascade="all, delete-orphan",
                                                   order_by="Sample.id")
    events: Mapped[list["Event"]] = relationship(back_populates="case", cascade="all, delete-orphan",
                                                 order_by="Event.id")

    def sample(self, role: str) -> "Sample | None":
        return next((s for s in self.samples if s.role == role), None)

    @property
    def proband(self) -> "Sample | None":
        return self.sample("proband") or self.sample("partner1")


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    role: Mapped[str] = mapped_column(String(16))   # proband | father | mother | partner1 | partner2
    name: Mapped[str] = mapped_column(String(128))
    patient_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pin: Mapped[str | None] = mapped_column(String(64), nullable=True)            # lab PIN
    sample_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    received_on: Mapped[str | None] = mapped_column(String(32), nullable=True)
    age: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dob: Mapped[str | None] = mapped_column(String(32), nullable=True)
    sex: Mapped[str | None] = mapped_column(String(16), nullable=True)
    affected: Mapped[bool] = mapped_column(Boolean, default=False)
    sample_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    collected_on: Mapped[str | None] = mapped_column(String(32), nullable=True)
    vcf_path: Mapped[str] = mapped_column(String(512))
    vcf_sample: Mapped[str] = mapped_column(String(128))

    case: Mapped[Case] = relationship(back_populates="samples")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    user: Mapped[str | None] = mapped_column(String(128), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))     # stage | info | warning | error | review | report
    message: Mapped[str] = mapped_column(Text)

    case: Mapped[Case] = relationship(back_populates="events")


def init_db() -> None:
    Base.metadata.create_all(engine)
    # lightweight migration: add columns introduced after the first release
    with engine.begin() as conn:
        for table, cls in (("cases", Case), ("samples", Sample)):
            have = {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            for col in cls.__table__.columns:
                if col.name not in have:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}")
