"""Fetch, dedupe and cache niche news from NewsAPI, with fixtures as a demo fallback."""
import json
import logging
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import httpx

from app.config import ROOT_DIR, settings

logger = logging.getLogger(__name__)

NEWS_API_BASE = "https://newsapi.org/v2"
CACHE_FILE = ROOT_DIR / ".cache" / "news_cache.json"
FIXTURES_DIR = ROOT_DIR / "fixtures"
CACHE_TTL_SECONDS = 3600
PAGE_SIZE = 20
THIN_RESULT_COUNT = 8
TITLE_SIMILARITY = 0.8


@dataclass(frozen=True)
class Niche:
    id: str
    label: str
    query: str
    category: str | None = None
    filter_headlines: bool = True


NICHES: dict[str, Niche] = {
    n.id: n
    for n in [
        Niche("finance", "Finance", "sensex OR nifty OR RBI", "business"),
        Niche("tech", "Tech/AI", "technology OR \"artificial intelligence\"", "technology", False),
        Niche("bollywood", "Bollywood", "bollywood", "entertainment"),
        Niche("cricket", "Cricket", "cricket", "sports"),
        Niche("fitness", "Fitness", "fitness OR workout OR nutrition", "health"),
        Niche("food", "Food", "food OR restaurant OR recipe"),
        Niche("travel", "Travel", "travel OR tourism OR airline"),
        Niche("startups", "Startups", "startup OR funding OR founder", "business"),
        Niche("fashion", "Fashion", "fashion OR couture OR \"style trends\""),
    ]
}


class NewsUnavailable(Exception):
    """Raised with a user-safe message when no news could be loaded."""


def _normalize_title(title: str) -> str:
    title = title.rsplit(" - ", 1)[0]  # NewsAPI appends " - Publisher"
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def dedupe(articles: list[dict]) -> list[dict]:
    kept: list[dict] = []
    seen: list[str] = []
    for article in articles:
        title = _normalize_title(article["title"])
        if any(SequenceMatcher(None, title, other).ratio() >= TITLE_SIMILARITY for other in seen):
            continue
        seen.append(title)
        kept.append(article)
    return kept


def _clean(raw_articles: list[dict]) -> list[dict]:
    """Keep usable articles and map them to the app's shape (ids are assigned later)."""
    articles = []
    for raw in raw_articles:
        title = (raw.get("title") or "").strip()
        url = raw.get("url") or ""
        if not title or title == "[Removed]" or not url.startswith("http"):
            continue
        articles.append({
            "title": title,
            "description": (raw.get("description") or "").strip(),
            "url": url,
            "source": (raw.get("source") or {}).get("name") or "Unknown",
            "published_at": raw.get("publishedAt") or "",
        })
    return articles


def _with_ids(articles: list[dict]) -> list[dict]:
    return [{**article, "id": f"a{i}"} for i, article in enumerate(articles[:PAGE_SIZE], start=1)]


def _matches_query(article: dict, query: str) -> bool:
    text = f"{article['title']} {article['description']}".lower()
    return any(term.strip(' "').lower() in text for term in query.split(" OR "))


async def _get(client: httpx.AsyncClient, endpoint: str, params: dict) -> list[dict]:
    response = await client.get(
        f"{NEWS_API_BASE}/{endpoint}",
        params=params,
        headers={"X-Api-Key": settings.news_api_key},
    )
    response.raise_for_status()
    return _clean(response.json().get("articles", []))


async def _fetch_live(query: str, category: str | None, filter_headlines: bool) -> list[dict]:
    articles: list[dict] = []
    async with httpx.AsyncClient(timeout=15) as client:
        if category:
            headlines = await _get(client, "top-headlines", {
                "category": category, "country": "in", "pageSize": PAGE_SIZE,
            })
            if filter_headlines:
                headlines = [a for a in headlines if _matches_query(a, query)]
            articles = headlines
        if len(articles) < THIN_RESULT_COUNT:
            articles += await _get(client, "everything", {
                "q": query, "searchIn": "title,description", "language": "en",
                "sortBy": "publishedAt", "pageSize": PAGE_SIZE,
            })
    return dedupe(articles)


def _read_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_cache(key: str, articles: list[dict]) -> None:
    cache = _read_cache()
    cache[key] = {"saved_at": time.time(), "articles": articles}
    _save_cache(cache)


def _save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(CACHE_FILE)
    except OSError as exc:
        logger.warning("Could not write news cache: %s", exc)


def cache_key(niche_id: str | None, custom_query: str | None) -> str:
    return niche_id if niche_id else f"custom:{(custom_query or '').strip().lower()}"


def cache_get(key: str) -> dict | None:
    entry = _read_cache().get(key)
    if entry and time.time() - entry["saved_at"] < CACHE_TTL_SECONDS:
        return entry["value"]
    return None


def cache_put(key: str, value: dict) -> None:
    cache = _read_cache()
    cache[key] = {"saved_at": time.time(), "value": value}
    _save_cache(cache)


def _load_fixture(niche_id: str | None) -> list[dict] | None:
    if not niche_id:
        return None
    path: Path = FIXTURES_DIR / f"news_{niche_id}.json"
    if not path.exists():
        return None
    return _clean(json.loads(path.read_text(encoding="utf-8"))["articles"])


async def get_articles(niche_id: str | None = None, custom_query: str | None = None) -> dict:
    """Return {articles, origin} where origin is live, cache, stale_cache or fixture."""
    niche = NICHES.get(niche_id) if niche_id else None
    if niche_id and not niche:
        raise NewsUnavailable("Unknown niche.")
    query = niche.query if niche else (custom_query or "").strip()
    if not query:
        raise NewsUnavailable("Pick a niche or type a topic.")

    key = cache_key(niche.id if niche else None, query)
    cached = _read_cache().get(key)
    if cached and time.time() - cached["saved_at"] < CACHE_TTL_SECONDS:
        return {"articles": cached["articles"], "origin": "cache"}

    failure = "no results"
    if settings.news_api_key:
        try:
            category = niche.category if niche else None
            articles = await _fetch_live(query, category, niche.filter_headlines if niche else False)
            if articles:
                articles = _with_ids(articles)
                _write_cache(key, articles)
                return {"articles": articles, "origin": "live"}
        except (httpx.HTTPError, ValueError) as exc:
            failure = type(exc).__name__
            status = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning("NewsAPI request failed: %s status=%s", failure, status)
    else:
        failure = "NEWS_API_KEY not set"

    if cached:
        return {"articles": cached["articles"], "origin": "stale_cache"}
    if settings.demo_mode:
        fixture = _load_fixture(niche.id if niche else None)
        if fixture:
            return {"articles": _with_ids(dedupe(fixture)), "origin": "fixture"}
        samples = ", ".join(sorted(p.stem.removeprefix("news_") for p in FIXTURES_DIR.glob("news_*.json")))
        raise NewsUnavailable(f"Live news is unavailable and there's no sample data for this topic. Try: {samples}.")
    if failure == "no results":
        raise NewsUnavailable("No recent stories found for that topic. Try something broader.")
    raise NewsUnavailable("Couldn't reach the news service right now. Please try again.")
