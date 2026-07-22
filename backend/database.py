"""
database.py

Sets up the SQLite database connection using SQLAlchemy.

Why SQLite: it's a single-file, zero-configuration database that's ideal
for a lightweight assessment project (per the assessment's "Persistence"
expectations: SQLite, H2, JSON-file, or another lightweight approach).

Why SQLAlchemy: it lets us define the database table as a Python class
(see models.py) instead of writing raw SQL everywhere. It's the most
widely used Python ORM, so it's a reasonable, professional default.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# This creates (or connects to, if it already exists) a file called tasks.db
# in the backend folder. That single file IS the entire database.
DATABASE_URL = "sqlite:///./tasks.db"

# `connect_args` is SQLite-specific: by default SQLite only allows the
# connection to be used from the thread that created it. FastAPI can handle
# requests on different threads, so we relax that restriction here.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# A "session" is a temporary conversation with the database (open it,
# do some reads/writes, close it). SessionLocal is a factory that creates
# these sessions on demand.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base is the parent class that our table models (in models.py) inherit
# from. SQLAlchemy uses it to know which Python classes map to which
# database tables.
Base = declarative_base()


def get_db():
    """
    A dependency function FastAPI will call for every request that needs
    database access. It opens a session, hands it to the endpoint function,
    and guarantees the session is closed afterward (even if an error occurs).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()