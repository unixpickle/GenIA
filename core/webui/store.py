"""SQLite job history and transactional queue transitions."""
import json
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

ACTIVE = ("queued", "running", "cancelling")


class Store:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(exist_ok=True)
        self.path = self.root / "jobs.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL,
                created REAL NOT NULL, started REAL, finished REAL, error TEXT,
                assets TEXT NOT NULL DEFAULT '{}')""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, seq)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        job = dict(row)
        job["assets"] = json.loads(job["assets"])
        return job

    def get(self, job_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def enqueue(self, job_id, title, seed):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0] >= 100:
                raise ValueError("The queue is full (100 waiting jobs). Please try again later.")
            db.execute("INSERT INTO jobs (id,title,seed,status,created) VALUES (?,?,?,'queued',?)",
                       (job_id, title, seed, time.time()))
        return self.get(job_id)

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY seq LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), row[0]))
        return self.get(row[0])

    def finish(self, job_id, status, error=None, assets=None):
        with self.connect() as db:
            db.execute("""UPDATE jobs SET status=?, finished=?, error=?, assets=?
                          WHERE id=? AND status IN ('running','cancelling')""",
                       (status, time.time(), error, json.dumps(assets or {}), job_id))

    def cancel(self, job_id):
        with self.connect() as db:
            db.execute("""UPDATE jobs SET status='cancelled', finished=?
                          WHERE id=? AND status='queued'""", (time.time(), job_id))
            db.execute("UPDATE jobs SET status='cancelling' WHERE id=? AND status='running'", (job_id,))
        return self.get(job_id)

    def recover(self):
        with self.connect() as db:
            db.execute("""UPDATE jobs SET status='interrupted', finished=?,
                error='The server stopped during this generation. Submit it again to retry.'
                WHERE status IN ('running','cancelling')""", (time.time(),))

    def delete(self, job_id):
        """Remove a terminal job's files before removing its history record.

        Keep the record if cleanup fails so deletion can be retried. The write
        transaction prevents a concurrent queue claim or state transition.
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return False
            if row["status"] in ACTIVE:
                raise ValueError("Cancel this job and wait for it to stop before deleting it.")
            directory = self.jobs_dir / job_id
            if directory.is_symlink():
                directory.unlink()
            elif directory.exists():
                shutil.rmtree(directory)
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return True

    def snapshot(self, before=None, limit=12, since=None):
        with self.connect() as db:
            active = db.execute("SELECT * FROM jobs WHERE status IN ('queued','running','cancelling') ORDER BY seq").fetchall()
            rows = db.execute("""SELECT * FROM jobs WHERE status NOT IN ('queued','running','cancelling')
                AND seq < ? ORDER BY seq DESC LIMIT ?""", (before or 2**63 - 1, limit + 1)).fetchall()
            counts = dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())
            history_ids = None if since is None else [row[0] for row in db.execute(
                "SELECT id FROM jobs WHERE seq>=? AND status NOT IN ('queued','running','cancelling')",
                (since,))]
        return {"queue": [self.decode(r) for r in active],
                "history": [self.decode(r) for r in rows[:limit]], "counts": counts,
                "next": rows[limit - 1]["seq"] if len(rows) > limit else None,
                "history_ids": history_ids}
