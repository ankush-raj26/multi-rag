"""Key-value access for short-lived state: login sessions, e-mail codes, reset tokens, rate limits.

Backed by Redis in the real app. `MemoryKV` has the same four methods and is what the
tests use, so the auth service never needs a running Redis to be tested.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

from app.config import get_settings


class KV(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_seconds: int) -> None: ...
    def delete(self, key: str) -> None: ...
    def incr(self, key: str, ttl_seconds: int) -> int: ...


class MemoryKV:
    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def _alive(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            return None
        value, expires = item
        if expires < time.monotonic():
            del self._data[key]
            return None
        return value

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._alive(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        with self._lock:
            self._data[key] = (value, time.monotonic() + ttl_seconds)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def incr(self, key: str, ttl_seconds: int) -> int:
        with self._lock:
            current = int(self._alive(key) or 0) + 1
            expires = self._data[key][1] if key in self._data else time.monotonic() + ttl_seconds
            self._data[key] = (str(current), expires)
            return current


class RedisKV:
    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=3)

    def get(self, key: str) -> str | None:
        value = self._r.get(key)
        return str(value) if value is not None else None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._r.set(key, value, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def incr(self, key: str, ttl_seconds: int) -> int:
        pipe = self._r.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl_seconds, nx=True)
        count, _ = pipe.execute()
        return int(count)

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:
            return False


def get_kv(redis_url: str | None = None) -> KV:
    url = redis_url if redis_url is not None else get_settings().redis_url
    return RedisKV(url) if url else MemoryKV()
