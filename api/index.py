"""Vercel entry point: serves the FastAPI app as a Python function; the React build is served as static files."""
import os
import sqlite3
import sys
from pathlib import Path

# Vercel's Python is built without SQLite extension loading, which sqlite-vec needs.
if not hasattr(sqlite3.Connection, "enable_load_extension"):
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
os.environ.setdefault("VAR_DIR", "/tmp/var")

from app.main import create_app  # noqa: E402

app = create_app()
