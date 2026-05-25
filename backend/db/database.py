import os

from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "db_data"
)
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "phytoquery.sqlite")

# SQLite URL for aiosqlite
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)


# ── SQLite production pragmas ────────────────────────────────────────────────
# Per-connection settings — they must run on EVERY new connection, not just
# once at startup. SQLAlchemy's "connect" event on the underlying sync engine
# (the one aiosqlite wraps) is the right hook.
#
# journal_mode=WAL       concurrent readers don't block writers
# synchronous=NORMAL     safe with WAL, ~100x write throughput vs FULL
# foreign_keys=ON        SQLite ships with FKs disabled by default
# busy_timeout=5000      retry locked DB for up to 5s before failing
# cache_size=-20000      20 MB page cache (negative value = KB units)
# temp_store=MEMORY      temporary tables / indexes stay in RAM
@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA cache_size=-20000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
