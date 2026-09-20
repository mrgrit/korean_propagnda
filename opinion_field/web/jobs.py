"""Single-worker job queue for the web UI (build / run / compare). One job at a time — a 1M-voter
simulation holds a few hundred MB and the CC backend is already parallel inside."""
from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any, Callable

MAX_LINES = 4000


class Job:
    def __init__(self, job_id: str, kind: str, params: dict[str, Any]):
        self.id = job_id
        self.kind = kind
        self.params = params
        self.status = "queued"          # queued | running | done | failed | stopped | cancelled
        self.lines: list[str] = []
        self.progress: dict[str, Any] = {}
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.created = time.time()
        self.started: float | None = None
        self.ended: float | None = None

    def log(self, *args: Any) -> None:
        line = " ".join(str(a) for a in args)
        self.lines.append(f"[{time.strftime('%H:%M:%S')}] {line}")
        if len(self.lines) > MAX_LINES:
            del self.lines[: len(self.lines) - MAX_LINES]

    def to_dict(self, since: int = 0, tail: int | None = None) -> dict[str, Any]:
        lines = self.lines[since:]
        if tail is not None:
            lines = lines[-tail:]
        return {"id": self.id, "kind": self.kind, "params": self.params, "status": self.status,
                "progress": self.progress, "error": self.error, "created": self.created,
                "started": self.started, "ended": self.ended, "n_lines": len(self.lines),
                "lines": lines, "stop_requested": self.stop_event.is_set()}


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.q: "queue.Queue[tuple[Job, Callable[[Job], None]]]" = queue.Queue()
        self.current: Job | None = None
        self._n = 0
        self._lock = threading.Lock()
        self.worker = threading.Thread(target=self._loop, daemon=True, name="of-jobs")
        self.worker.start()

    def submit(self, kind: str, params: dict[str, Any], fn: Callable[[Job], None]) -> Job:
        with self._lock:
            self._n += 1
            job = Job(f"j{self._n:04d}", kind, params)
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.q.put((job, fn))
        return job

    def _loop(self) -> None:
        while True:
            job, fn = self.q.get()
            if job.stop_event.is_set():
                job.status = "cancelled"
                job.ended = time.time()
                continue
            self.current = job
            job.status = "running"
            job.started = time.time()
            try:
                fn(job)
                job.status = "stopped" if job.stop_event.is_set() else "done"
            except Exception as e:  # noqa: BLE001 - surface everything to the UI
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.log("ERROR", job.error)
                job.log(traceback.format_exc())
            finally:
                job.ended = time.time()
                self.current = None

    def stop(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        job.stop_event.set()
        job.log("stop requested — finishing the current round, then checkpointing")
        return True

    def list(self) -> list[dict[str, Any]]:
        return [self.jobs[j].to_dict(tail=0) for j in reversed(self.order)]

    def busy(self) -> bool:
        return self.current is not None or not self.q.empty()
