from __future__ import annotations

from collections import OrderedDict
from threading import RLock

from .contracts import PageObservation


class PageObservationCache:
    """Bounded process-local cache; external caches remain optional accelerators."""

    def __init__(self, max_entries: int = 256):
        self._max_entries = max_entries
        self._items: OrderedDict[str, PageObservation] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def key(page_sha256: str, ocr_model_version: str, preprocessing_version: str,
            page_id: str | None = None) -> str:
        # A content-identical page owned by another document is still distinct
        # evidence.  The identity prefix prevents cross-document provenance reuse.
        return f"{page_id or 'UNSCOPED'}:{page_sha256}:{ocr_model_version}:{preprocessing_version}"

    def get(self, key: str) -> PageObservation | None:
        with self._lock:
            item = self._items.get(key)
            if item is not None:
                self._items.move_to_end(key)
            return item

    def put(self, key: str, value: PageObservation) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)
