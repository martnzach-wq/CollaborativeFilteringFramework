"""HTTP access with a response cache and retries, shared by the connectors."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests


class SourceError(RuntimeError):
    pass


class TooLargeError(SourceError):
    """The server keeps answering 413 or 'processing asynchronously'."""


class CachedClient:
    """GET/POST with responses cached under ``cache_dir/<kind>/`` by request hash.

    A cached request is never sent again. HTTP 413, 429, 5xx and bodies for which ``is_pending`` is true are
    retried with exponential back-off; other errors raise ``error``.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        offline: bool = False,
        max_retries: int = 6,
        timeout: float = 120.0,
        is_pending: Callable[[str], bool] = lambda text: False,
        error: type[SourceError] = SourceError,
    ):
        self.cache_dir = Path(cache_dir)
        self.session = session or requests.Session()
        self.sleep = sleep
        self.offline = offline
        self.max_retries = max_retries
        self.timeout = timeout
        self.is_pending = is_pending
        self.error = error

    def get(self, url: str, params: dict, kind: str, fmt: str = "json") -> str:
        return self._request("GET", url, params, None, kind, fmt)

    def post_json(self, url: str, payload: dict, kind: str) -> str:
        return self._request("POST", url, {}, payload, kind, "json")

    def _request(self, method: str, url: str, params: dict, payload: dict | None, kind: str, fmt: str) -> str:
        key = request_key(url, params, payload)
        path = self.cache_dir / kind / f"{key}.{fmt}"
        if path.exists():
            return path.read_text(encoding="utf-8")
        if self.offline:
            raise self.error(f"not cached and offline: {url} {params or payload}")
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            if method == "GET":
                r = self.session.get(url, params=encode(params), timeout=self.timeout)
            else:
                r = self.session.post(url, json=payload, timeout=self.timeout)
            pending = self.is_pending(r.text)
            if r.status_code == 200 and not pending:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(r.text, encoding="utf-8")
                return r.text
            retryable = r.status_code in (413, 429) or r.status_code >= 500 or pending
            if not retryable:
                raise self.error(f"HTTP {r.status_code} for {r.url}: {r.text[:300]}")
            if attempt < self.max_retries:
                self.sleep(delay)
                delay *= 2
        if r.status_code == 413 or pending:
            raise _too_large(self.error)(url)
        raise self.error(f"HTTP {r.status_code} after {self.max_retries} retries: {r.url}")


def _too_large(error: type[SourceError]) -> type[SourceError]:
    return type(f"{error.__name__}TooLarge", (error, TooLargeError), {})


def as_list(v: Any) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def encode(params: dict) -> list[tuple[str, str]]:
    return [(k, str(x)) for k, v in params.items() for x in as_list(v)]


def request_key(url: str, params: dict, payload: dict | None = None) -> str:
    canon = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(encode(params)))
    if payload is not None:
        canon += json.dumps(payload, sort_keys=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", url.rsplit("/", 1)[-1])[:40]
    return f"{slug}_{hashlib.sha256(canon.encode()).hexdigest()[:16]}"
