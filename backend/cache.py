"""Small bounded TTL cache used by the prediction endpoint."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

Value = TypeVar("Value")


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    size: int


class TTLCache(Generic[Value]):
    """Thread-safe LRU cache with expiry and a hard entry limit."""

    def __init__(self, max_entries: int = 1024, ttl_seconds: float = 300.0) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, Value]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Value | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self._misses += 1
                return None
            expires_at, value = item
            if expires_at <= now:
                del self._items[key]
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Value) -> None:
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_seconds, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                size=len(self._items),
            )
