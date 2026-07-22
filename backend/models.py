"""
models.py

Defines the `Task` table: one row per uploaded video.

Each video the user uploads becomes a "task" that we track through its
lifecycle: pending -> processing -> complete (or failed).
"""

from sqlalchemy import Column, String, DateTime, Text
from datetime import datetime, timezone
from database import Base


class Task(Base):
    __tablename__ = "tasks"

    # Primary key: a unique string ID for this task (we generate a UUID
    # for this -- see main.py). Using a string ID rather than an
    # auto-incrementing integer means IDs are unguessable and safe to
    # expose in URLs.
    id = Column(String, primary_key=True, index=True)

    # One of: "pending", "processing", "complete", "failed"
    status = Column(String, default="pending", nullable=False)

    # Where the uploaded video file lives on disk
    video_path = Column(String, nullable=False)

    # Once processing finishes, we store the ENTIRE result JSON here as a
    # text blob (SQLite has no native JSON type, so we store it as text
    # and parse/serialize with Python's json module when reading/writing).
    result_json = Column(Text, nullable=True)

    # If something goes wrong during processing, we store the error here
    # so it's visible via the status endpoint instead of failing silently.
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )