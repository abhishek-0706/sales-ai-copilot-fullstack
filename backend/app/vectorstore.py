"""Hybrid retrieval on SQLite: sqlite-vec (vector KNN) + FTS5 (BM25 keyword), fused with RRF.

Access control is part of the query: the caller's territory / record-type / sensitivity
predicate is applied inside both the KNN search and the keyword search, so records the
user cannot see are never scored, returned or placed in a prompt.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass

import numpy as np
import sqlite_vec

from .embeddings import Embedder, tokenize
from .models import SENSITIVITY_LEVEL, Evidence, User
from .security import sql_predicate

RRF_K = 60


@dataclass
class Hit:
    evidence: Evidence
    score: float
    vector_rank: int | None
    keyword_rank: int | None


class HybridStore:
    def __init__(self, embedder: Embedder, path: str = ":memory:"):
        self.embedder = embedder
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._evidence: dict[int, Evidence] = {}

    def build(self, chunks: list[Evidence]) -> None:
        dim = self.embedder.dim
        with self._lock:
            cur = self.db.cursor()
            cur.executescript("""
                DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS vec_chunks; DROP TABLE IF EXISTS fts_chunks;
                CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, source_id TEXT UNIQUE, record_id TEXT,
                    record_type TEXT, account_id TEXT, territory TEXT, sensitivity_level INTEGER, title TEXT, text TEXT);
                CREATE VIRTUAL TABLE fts_chunks USING fts5(title, text, content='chunks', content_rowid='rowid',
                    tokenize='porter unicode61');
            """)
            cur.execute(f"""CREATE VIRTUAL TABLE vec_chunks USING vec0(chunk_rowid INTEGER PRIMARY KEY,
                embedding float[{dim}] distance_metric=cosine, territory TEXT, record_type TEXT,
                sensitivity_level INTEGER, account_id TEXT)""")
            vectors = self.embedder.embed_documents([f"{c.title}. {c.text}" for c in chunks])
            for i, (c, v) in enumerate(zip(chunks, vectors), start=1):
                level = SENSITIVITY_LEVEL[c.sensitivity]
                cur.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
                            (i, c.source_id, c.record_id, c.source_type, c.account_id or "", c.territory, level,
                             c.title, c.text))
                cur.execute("INSERT INTO vec_chunks(chunk_rowid, embedding, territory, record_type, sensitivity_level,"
                            " account_id) VALUES (?,?,?,?,?,?)",
                            (i, np.asarray(v, dtype=np.float32).tobytes(), c.territory, c.source_type, level,
                             c.account_id or ""))
                self._evidence[i] = c
            cur.execute("INSERT INTO fts_chunks(fts_chunks) VALUES ('rebuild')")
            self.db.commit()

    def count(self) -> int:
        return len(self._evidence)

    def search(self, user: User, query: str, k: int = 6, account_id: str | None = None,
               candidates: int = 20, mode: str = "hybrid") -> list[Hit]:
        """mode: 'hybrid' (default), or 'vector' / 'keyword' for ablation in the evaluation."""
        where, params = sql_predicate(user)
        if account_id:
            where += " AND account_id IN (?, '')"
            params = [*params, account_id]
        qvec = np.asarray(self.embedder.embed_query(query), dtype=np.float32).tobytes()
        with self._lock:
            vec_rows = self.db.execute(
                f"SELECT chunk_rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? AND {where}"
                " ORDER BY distance", [qvec, candidates, *params]).fetchall()
            kw_rows: list[tuple[int, float]] = []
            terms = tokenize(query)
            if terms:
                fts_query = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
                kw_rows = self.db.execute(
                    f"SELECT chunks.rowid, bm25(fts_chunks) FROM fts_chunks JOIN chunks ON chunks.rowid = fts_chunks.rowid"
                    f" WHERE fts_chunks MATCH ? AND {where} ORDER BY bm25(fts_chunks) LIMIT ?",
                    [fts_query, *params, candidates]).fetchall()
        vrank = {rid: r for r, (rid, _) in enumerate(vec_rows, start=1)} if mode != "keyword" else {}
        krank = {rid: r for r, (rid, _) in enumerate(kw_rows, start=1)} if mode != "vector" else {}
        fused = {rid: sum(1.0 / (RRF_K + rk[rid]) for rk in (vrank, krank) if rid in rk)
                 for rid in {*vrank, *krank}}
        ordered = sorted(fused, key=lambda rid: (-fused[rid], rid))[:k]
        return [Hit(self._evidence[rid].model_copy(update={"score": round(fused[rid], 5)}), fused[rid],
                    vrank.get(rid), krank.get(rid)) for rid in ordered]
