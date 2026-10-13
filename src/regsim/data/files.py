"""Download of static files into ``data/raw`` (skipped when the file exists)."""

from __future__ import annotations

from pathlib import Path

import requests


def download(url: str, path: str | Path, timeout: float = 600.0, chunk: int = 1 << 20) -> Path:
    """Download ``url`` to ``path`` unless it exists; the file appears only after a complete transfer."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk):
                f.write(block)
    tmp.replace(path)
    return path
