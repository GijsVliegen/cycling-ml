from __future__ import annotations

import hashlib
import re
import asyncio
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import httpx

BASE_URL = "https://www.procyclingstats.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
DEFAULT_TIMEOUT = 30.0
HTML_CACHE_DIR = Path("html pages")
HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_fragment(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    return sanitized or "index"


def _url_to_cache_path(url: str) -> Path:
    parsed = urlparse(url)
    domain_dir = HTML_CACHE_DIR / _sanitize_fragment(parsed.netloc.replace(".", "_"))
    raw_path = parsed.path.strip("/") or "index"
    safe_path = _sanitize_fragment(raw_path.replace("/", "_"))
    if parsed.query:
        query_hash = hashlib.md5(parsed.query.encode()).hexdigest()[:8]
        safe_path = f"{safe_path}_{query_hash}"
    if not safe_path.endswith(".html"):
        safe_path = f"{safe_path}.html"
    return domain_dir / safe_path


async def _download_into(url: str, dest: Path, client: httpx.AsyncClient) -> None:
    resp = await client.get(url, follow_redirects=True)
    if resp.status_code == 403:
        raise PermissionError(f"403 Forbidden for {url}")
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")


async def ensure_html(
    url: str,
    *,
    force: bool = False,
    client: httpx.AsyncClient | None = None,
) -> Path:
    cache_path = _url_to_cache_path(url)
    if cache_path.exists() and not force:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    close_client = client is None
    if client is None:
        client = httpx.AsyncClient(headers=HEADERS, timeout=DEFAULT_TIMEOUT)
    try:
        await _download_into(url, cache_path, client)
    finally:
        if close_client:
            await client.aclose()
    return cache_path


def read_cached_html(path: Path) -> str:
    return path.read_text(encoding="utf-8")


async def fetch_html(
    url: str,
    *,
    force: bool = False,
    client: httpx.AsyncClient | None = None,
) -> str:
    cache_path = await ensure_html(url, force=force, client=client)
    return read_cached_html(cache_path)


async def fetch_race_index_html(year: int, *, force: bool = False) -> str:
    url = f"{BASE_URL}/races.php?year={year}"
    return await fetch_html(url, force=force)


async def fetch_rider_html(
    rider_name: str,
    *,
    force: bool = False,
    client: httpx.AsyncClient | None = None,
) -> str:
    url = f"{BASE_URL}/rider/{rider_name}"
    return await fetch_html(url, force=force, client=client)


async def fetch_rider_pages(
    rider_names: List[str],
    *,
    force: bool = False,
    concurrency: int = 5,
) -> Dict[str, str]:
    pages: Dict[str, str] = {}
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _fetch_single(
        rider_name: str, client: httpx.AsyncClient
    ) -> tuple[str, str]:
        async with semaphore:
            html = await fetch_rider_html(rider_name, force=force, client=client)
        return rider_name, html

    async with httpx.AsyncClient(headers=HEADERS, timeout=DEFAULT_TIMEOUT) as client:
        tasks = [_fetch_single(name, client) for name in rider_names]
        for task in asyncio.as_completed(tasks):
            rider_name, html = await task
            pages[rider_name] = html
    return pages
