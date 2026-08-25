"""Terminable process boundary for native OCR libraries.

Threads cannot reliably interrupt Paddle/ONNX native calls.  This worker keeps
one model instance alive for normal requests, but makes a hung invocation
terminable and restartable without blocking unrelated page orchestration.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import queue
import uuid
from dataclasses import dataclass
from typing import Any


class OCRTimeoutError(TimeoutError):
    failure_code = "OCR_TIMEOUT"


class OCRWorkerError(RuntimeError):
    pass


def _worker_main(module_name: str, class_name: str, kwargs: dict[str, Any], requests, results):
    provider_class = getattr(importlib.import_module(module_name), class_name)
    provider = provider_class(**kwargs)
    while True:
        request = requests.get()
        if request is None:
            return
        job_id, method_name, args = request
        try:
            value = getattr(provider, method_name)(*args)
            results.put((job_id, True, value))
        except BaseException as exc:  # child errors cross the process as data
            results.put((job_id, False, f"{type(exc).__name__}: {exc}"))


@dataclass(frozen=True)
class OCRWorkerStats:
    starts: int
    completed: int
    timeouts: int
    terminations: int


class IsolatedTextExtractor:
    """TextExtractor-compatible persistent subprocess with hard call bounds."""

    def __init__(self, module_name: str, class_name: str, *, provider_kwargs: dict | None = None,
                 timeout_seconds: float = 30.0, engine_name: str = "isolated_ocr",
                 model_name: str = "unknown", model_version: str = "unknown") -> None:
        if timeout_seconds <= 0:
            raise ValueError("ocr_timeout_seconds must be positive")
        self.module_name = module_name
        self.class_name = class_name
        self.provider_kwargs = dict(provider_kwargs or {})
        self.timeout_seconds = timeout_seconds
        self.engine_name = engine_name
        self.model_name = model_name
        self.model_version = model_version
        self._context = mp.get_context("spawn")
        self._process = None
        self._requests = None
        self._results = None
        self._starts = self._completed = self._timeouts = self._terminations = 0

    @property
    def stats(self) -> OCRWorkerStats:
        return OCRWorkerStats(self._starts, self._completed, self._timeouts, self._terminations)

    def _start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._requests = self._context.Queue()
        self._results = self._context.Queue()
        self._process = self._context.Process(
            target=_worker_main,
            args=(self.module_name, self.class_name, self.provider_kwargs,
                  self._requests, self._results),
            daemon=True,
            name=f"cdp-ocr-{self.engine_name}",
        )
        self._process.start()
        self._starts += 1

    def _terminate(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            self._terminations += 1
        self._process = self._requests = self._results = None

    def _execute(self, method_name: str, *args):
        self._start()
        job_id = uuid.uuid4().hex
        self._requests.put((job_id, method_name, args))
        try:
            returned_id, succeeded, value = self._results.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._timeouts += 1
            self._terminate()
            raise OCRTimeoutError(
                f"OCR_TIMEOUT:{self.engine_name}:{self.timeout_seconds:g}s"
            ) from exc
        if returned_id != job_id:
            self._terminate()
            raise OCRWorkerError("OCR_WORKER_RESPONSE_ID_MISMATCH")
        if not succeeded:
            raise OCRWorkerError(value)
        self._completed += 1
        return value

    def extract(self, image):
        return self._execute("extract", image)

    def extract_region(self, image, x0: int, y0: int, x1: int, y1: int):
        return self._execute("extract_region", image, x0, y0, x1, y1)

    def close(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._requests.put(None)
            self._process.join(timeout=5)
        self._terminate()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
