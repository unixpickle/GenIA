"""One supervisor thread, one isolated worker process per job, no GPU in the server."""
import fcntl
import os
import signal
import subprocess
import sys
import threading

from .artifacts import collect
from .progress import write_stage


class JobQueue:
    def __init__(self, store, *, environment=None, command=None):
        self.store = store
        self.environment = environment or {}
        self.command = command or (lambda path: [sys.executable, "-u", "-m",
                                                 "genia.core.webui.worker", str(path)])
        self.condition = threading.Condition()
        self.process = None
        self.current = None
        self.stopping = False
        self.thread = None
        self.lock_file = None

    def start(self):
        # A second server must not recover or consume this server's live queue.
        self.lock_file = (self.store.root / "server.lock").open("a")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock_file.close()
            self.lock_file = None
            raise RuntimeError("Another server is already using this data directory.") from exc
        self.store.recover()
        self.thread = threading.Thread(target=self._run, name="genia-queue", daemon=True)
        self.thread.start()

    def wake(self):
        with self.condition:
            self.condition.notify_all()

    def _kill(self):
        if self.process is not None:
            try:
                # Kill descendants too. The process is not reaped until under this lock,
                # so its process-group ID cannot have been recycled for an unrelated job.
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def cancel(self, job_id):
        with self.condition:
            job = self.store.cancel(job_id)
            if self.current == job_id and job and job["status"] == "cancelling":
                self._kill()
            self.condition.notify_all()
            return job

    def stop(self):
        with self.condition:
            self.stopping = True
            self._kill()
            self.condition.notify_all()
        if self.thread:
            self.thread.join()
            self.thread = None
        if self.lock_file:
            self.lock_file.close()
            self.lock_file = None

    def _run(self):
        while True:
            with self.condition:
                if self.stopping:
                    return
                job = self.store.claim()
                if job is None:
                    self.condition.wait(timeout=2)
                    continue
                self.current = job["id"]
                directory = self.store.jobs_dir / self.current
                try:
                    with (directory / "worker.log").open("ab", buffering=0) as log:
                        self.process = subprocess.Popen(
                            self.command(directory), stdin=subprocess.PIPE, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, **self.environment, "PYTHONUNBUFFERED": "1"})
                except Exception as exc:
                    self.store.finish(self.current, "failed", f"Could not start worker: {exc}")
                    self.current = None
                    continue
            # Do not call wait/poll outside the lock: cancelling must never signal a
            # reaped child's potentially reused PID. waitid(WNOWAIT) is not on macOS.
            with self.condition:
                while self.process.poll() is None:
                    self.condition.wait(timeout=0.2)
                code = self.process.returncode
                self.process.stdin.close()
                self.process = None
                job = self.store.get(self.current)
                if job["status"] == "cancelling":
                    self.store.finish(self.current, "cancelled")
                elif self.stopping:
                    self.store.finish(self.current, "interrupted", "Server stopped. Submit again to retry.")
                elif code != 0:
                    self.store.finish(self.current, "failed", f"Worker exited with code {code}. See the log for details.")
                else:
                    try:
                        write_stage(directory, "previews")
                        self.store.finish(self.current, "succeeded", assets=collect(directory))
                    except Exception as exc:
                        self.store.finish(self.current, "failed", str(exc))
                self.current = None
