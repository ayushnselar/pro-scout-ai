"""
Scout Agent — real-time news retrieval and batch LLM summarization.

Step 3.1: Async DuckDuckGo search per player, ONE Groq summarization call
for all players, structured NewsResult output. Fallback to raw snippets if Groq fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Dict, List
from urllib.parse import urlparse

from graph.schemas import NewsResult

logger = logging.getLogger(__name__)

_DDG_MAX_RESULTS = 3
_GROQ_MODEL = "llama-3.1-8b-instant"
_DDG_QUERY_TIMEOUT_S = 3.0
_DDG_PLAYER_TIMEOUT_S = 6.0
_DDG_GLOBAL_TIMEOUT_S = 9.0
_GROQ_TIMEOUT_S = 10

DOMAIN_DENYLIST = {
    "wikipedia.org",
    "imdb.com",
    "behindthename.com",
    "nameberry.com",
    "babynames.com",
    "thebump.com",
    "britannica.com",
}

KEYWORD_DENYLIST = {
    "name meaning",
    "meaning of",
    "saint",
    "biblical",
    "definition",
    "etymology",
}

_DDG_CONCURRENCY = 4
_NBA_TEAM_LOOKUP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class _FilterTally:
    before: int
    after: int
    removed_domain: int
    removed_keyword: int
    removed_empty: int


def _should_add_tonight(user_query: str | None) -> bool:
    """
    If the user question implies immediate recency, bias searches to 'tonight'.
    Triggers: contains 'tonight' or 'today' or a date-like phrase.
    """
    if not user_query:
        return False
    q = user_query.lower()
    if "tonight" in q or "today" in q:
        return True
    # Simple date-like phrases: 2026-02-26, 2/26, 02/26/2026
    if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", q):
        return True
    if re.search(r"\b\d{1,2}/\d{1,2}(/\d{2,4})?\b", q):
        return True
    return False


NEGATIVE_SUFFIX = " -imdb -actor -cast -bio -wikipedia -name -meaning -saint -etymology"


def _build_ddg_queries(player: str, *, user_query: str | None, team: str | None = None) -> list[str]:
    """
    Build the 3 required DDG queries per player (NBA-context + recency).

    Default queries:
    - f"{player} NBA injury update"
    - f"{player} NBA status tonight"
    - f"{player} NBA minutes restriction"

    If recency is implied by the user query, include "tonight" in the other
    queries as well.
    """
    add_tonight = _should_add_tonight(user_query)
    context = team if team else "NBA"
    q1 = f"{player} {context} injury update"
    q2 = f"{player} {context} status tonight"
    q3 = f"{player} {context} minutes restriction"
    if add_tonight:
        q1 = f"{q1} tonight"
        q3 = f"{q3} tonight"
    return [q1 + NEGATIVE_SUFFIX, q2 + NEGATIVE_SUFFIX, q3 + NEGATIVE_SUFFIX]


def _normalize_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _get_team(player: str) -> str | None:
    """
    Resolve player's current team via nba_api. Module-level for testability.
    Returns None on API failure or missing data.
    """
    try:
        from nba_api.stats.endpoints import commonplayerinfo  # type: ignore[import]
        from nba_api.stats.static import players as nba_players  # type: ignore[import]
    except Exception:
        return None
    try:
        matches = nba_players.find_players_by_full_name(player)
        if not matches:
            return None
        player_id = int(matches[0]["id"])
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        frames = info.get_data_frames()
        if not frames:
            return None
        df = frames[0]
        if "TEAM_NAME" not in df.columns:
            return None
        name = df["TEAM_NAME"].iloc[0]
        return str(name) if isinstance(name, (str, bytes)) else None
    except Exception:
        return None


def _domain_is_denied(domain: str) -> bool:
    if not domain:
        return False
    for denied in DOMAIN_DENYLIST:
        if domain == denied or domain.endswith("." + denied):
            return True
    return False


def _text_has_denied_keyword(text: str) -> bool:
    hay = text.lower()
    return any(kw in hay for kw in KEYWORD_DENYLIST)


def _get_groq_api_key() -> str | None:
    """Groq API key from environment or Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # type: ignore[import]

        k = getattr(st, "secrets", None) and st.secrets.get("GROQ_API_KEY")
        if isinstance(k, str) and k:
            return k
    except Exception:
        pass
    return None


def _ddg_text_sync(query: str) -> List[dict[str, Any]]:
    """Run DuckDuckGo text search (sync). Returns list of {title, href, body}."""
    try:
        from ddgs import DDGS  # type: ignore[import]

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=_DDG_MAX_RESULTS))
    except Exception as e:
        logger.warning("DuckDuckGo search failed for %r: %s", query, e)
        return []


async def _ddg_text_with_timeout(query: str, semaphore: asyncio.Semaphore) -> List[dict[str, Any]]:
    """
    Async wrapper around DDG text search with per-query timeout and concurrency limit.
    Semaphore must be created in the same event loop as the caller (e.g. inside get_news).
    """
    async with semaphore:
        loop = asyncio.get_event_loop()
        start = monotonic()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _ddg_text_sync, query),
                timeout=_DDG_QUERY_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("DuckDuckGo query timed out for %r after %.2fs", query, _DDG_QUERY_TIMEOUT_S)
            return []
        except Exception as e:
            logger.warning("DuckDuckGo query failed for %r: %s", query, e)
            return []
        finally:
            elapsed = monotonic() - start
            logger.debug("DuckDuckGo query %r completed in %.2fs", query, elapsed)


def _filter_ddg_results(
    results: list[dict[str, Any]],
) -> tuple[list[str], list[str], _FilterTally, dict[str, int]]:
    """
    Filter raw DDG results before summarization to remove irrelevant content.

    Discard a result if:
    - Domain matches DOMAIN_DENYLIST (subdomain match)
    - Snippet/title contains KEYWORD_DENYLIST phrases (case-insensitive)
    - Snippet is missing/empty
    """
    before = len(results)

    kept_snippets: list[str] = []
    kept_urls: list[str] = []
    seen_urls: set[str] = set()

    removed_domain = 0
    removed_keyword = 0
    removed_empty = 0

    for item in results:
        if not isinstance(item, dict):
            continue

        title = (item.get("title") or "").strip()
        body = (item.get("body") or "").strip()
        snippet = (body or title).strip()
        if not snippet:
            removed_empty += 1
            continue

        href = item.get("href")
        domain = _normalize_domain(href) if isinstance(href, str) else ""
        if domain and _domain_is_denied(domain):
            removed_domain += 1
            continue

        if _text_has_denied_keyword(f"{title}\n{snippet}"):
            removed_keyword += 1
            continue

        kept_snippets.append(snippet)
        if isinstance(href, str) and href and href not in seen_urls:
            seen_urls.add(href)
            kept_urls.append(href)

    after = len(kept_snippets)
    tally = _FilterTally(
        before=before,
        after=after,
        removed_domain=removed_domain,
        removed_keyword=removed_keyword,
        removed_empty=removed_empty,
    )
    removed = {"domain": removed_domain, "keyword": removed_keyword, "empty": removed_empty}
    return kept_snippets, kept_urls, tally, removed


async def _fetch_snippets_for_player(
    player: str,
    *,
    user_query: str | None,
    team: str | None,
    semaphore: asyncio.Semaphore,
) -> tuple[str, List[str], List[str], _FilterTally, dict[str, int]]:
    """
    Run three DDG queries for one player in parallel and filter results.

    Returns:
    (player, filtered_snippets, filtered_urls, tally, removed_reasons)
    """
    queries = _build_ddg_queries(player, user_query=user_query, team=team)
    start = monotonic()
    try:
        results: List[List[dict]] = await asyncio.wait_for(
            asyncio.gather(*[_ddg_text_with_timeout(q, semaphore) for q in queries], return_exceptions=False),
            timeout=_DDG_PLAYER_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("DuckDuckGo per-player retrieval timed out for %r after %.2fs", player, _DDG_PLAYER_TIMEOUT_S)
        return (player, [], [], _FilterTally(0, 0, 0, 0, 0), {"domain": 0, "keyword": 0, "empty": 0})
    finally:
        elapsed = monotonic() - start
        logger.debug("DuckDuckGo per-player retrieval for %r completed in %.2fs", player, elapsed)
    flat: list[dict[str, Any]] = []
    for result_list in results:
        if isinstance(result_list, list):
            flat.extend([x for x in result_list if isinstance(x, dict)])

    snippets, urls, tally, removed = _filter_ddg_results(flat)
    return (player, snippets, urls, tally, removed)


def _groq_summarize_batch(players_snippets: dict[str, tuple[List[str], List[str]]]) -> dict[str, dict[str, str]]:
    """
    Single Groq call to summarize all player snippets. Returns dict[player_name, summary].
    Raises on missing key or API/parse errors.
    """
    api_key = _get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not configured")
    try:
        from groq import Groq
    except Exception as e:
        raise RuntimeError("groq client not available") from e

    players = list(players_snippets.keys())
    template = {p: "" for p in players}

    parts: List[str] = []
    for player, (snippets, _) in players_snippets.items():
        if snippets:
            parts.append(f"PLAYER: {player}\nSNIPPETS:\n" + "\n".join(f"- {s}" for s in snippets[:15]))
        else:
            parts.append(f"PLAYER: {player}\nSNIPPETS:\n- (none)")
    combined = "\n\n".join(parts)

    user_content = (
        "Summarize the fantasy impact and availability status for each player.\n"
        f"Players: {json.dumps(players)}\n\n"
        "Return ONLY JSON.\n"
        "Keys MUST be exactly the player names provided. Do not rename players. Do not omit keys.\n"
        "If there is no relevant news, return an empty string for that player.\n\n"
        "Return JSON exactly in this shape (same keys):\n"
        f"{json.dumps(template, ensure_ascii=False)}\n\n"
        "Data:\n"
        f"{combined}"
    )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You output only valid JSON. No markdown, no explanation."},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )
    content = (response.choices[0] if response.choices else None) and response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("Groq response content is not a string")
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("Groq response is not a JSON object")
    # Normalize to dict[player -> dict fields]
    out: dict[str, dict[str, str]] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            out[str(key)] = {
                "availability": str(val.get("availability", "")) if val.get("availability") is not None else "",
                "minutes_note": str(val.get("minutes_note", "")) if val.get("minutes_note") is not None else "",
                "role_note": str(val.get("role_note", "")) if val.get("role_note") is not None else "",
                "news_summary": str(val.get("news_summary", "")) if val.get("news_summary") is not None else "",
            }
        else:
            out[str(key)] = {
                "availability": "",
                "minutes_note": "",
                "role_note": "",
                "news_summary": str(val) if val is not None else "",
            }
    return out


async def get_news(
    players: List[str],
    *,
    user_query: str | None = None,
    debug: bool = False,
) -> List[NewsResult] | tuple[List[NewsResult], Dict[str, Any]]:
    """
    Retrieve news for all players in parallel, then one batch Groq summarization.

    - DuckDuckGo searches (injury, minutes, fantasy) per player via asyncio.gather.
    - Single Groq call summarizes all snippets; parse into NewsResult per player.
    - On Groq failure, use concatenated snippets as summary. On search failure, use empty summary/URLs.
    """
    debug_meta: Dict[str, Any] = {
        "per_player": {},
        "filtered_out_reasons": {"domain": 0, "keyword": 0, "empty": 0},
        "groq_succeeded": False,
    }

    if not players:
        if debug:
            return [], debug_meta
        return []

    logger.info("Scout get_news: %d player(s), debug=%s", len(players), debug)

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Semaphore must be created in this event loop (not at module import) to avoid
    # "bound to a different event loop" errors when Streamlit uses asyncio.run().
    ddg_semaphore = asyncio.Semaphore(_DDG_CONCURRENCY)

    # Team lookup via nba_api is blocking I/O; run in executor to avoid blocking the loop.
    def _fetch_teams_sync() -> dict[str, str | None]:
        return {p: _get_team(p) for p in players}

    loop = asyncio.get_event_loop()
    try:
        teams = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch_teams_sync),
            timeout=_NBA_TEAM_LOOKUP_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("NBA team lookup timed out after %.2fs", _NBA_TEAM_LOOKUP_TIMEOUT_S)
        teams = {p: None for p in players}

    try:
        start_global = monotonic()
        results = await asyncio.wait_for(
            asyncio.gather(
                *[
                    _fetch_snippets_for_player(p, user_query=user_query, team=teams.get(p), semaphore=ddg_semaphore)
                    for p in players
                ],
                return_exceptions=True,
            ),
            timeout=_DDG_GLOBAL_TIMEOUT_S,
        )
        logger.debug(
            "DuckDuckGo global retrieval for %d players completed in %.2fs",
            len(players),
            monotonic() - start_global,
        )
    except asyncio.TimeoutError:
        logger.warning("DuckDuckGo global gather timed out after %.2fs", _DDG_GLOBAL_TIMEOUT_S)
        player_results = [
            (p, [], [], _FilterTally(0, 0, 0, 0, 0), {"domain": 0, "keyword": 0, "empty": 0}) for p in players
        ]
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("DuckDuckGo gather failed unexpectedly: %s", exc)
        player_results = [
            (p, [], [], _FilterTally(0, 0, 0, 0, 0), {"domain": 0, "keyword": 0, "empty": 0}) for p in players
        ]
    else:
        player_results: list[tuple[str, list[str], list[str], _FilterTally, dict[str, int]]] = []
        for idx, result in enumerate(results):
            player_name = players[idx]
            if isinstance(result, Exception):
                logger.warning("DuckDuckGo fetch failed for %s: %s", player_name, result)
                player_results.append(
                    (player_name, [], [], _FilterTally(0, 0, 0, 0, 0), {"domain": 0, "keyword": 0, "empty": 0})
                )
                continue
            # Normal successful result from _fetch_snippets_for_player.
            res_player, snippets, urls, tally, removed = result
            # Prefer the externally tracked name to avoid any mismatch.
            player_results.append((player_name, snippets, urls, tally, removed))

    players_snippets: dict[str, tuple[List[str], List[str]]] = {}
    for player, snippets, urls, tally, removed in player_results:
        players_snippets[player] = (snippets, urls)
        debug_meta["per_player"][player] = {
            "ddg_results_before": tally.before,
            "ddg_results_after": tally.after,
        }
        for k in ("domain", "keyword", "empty"):
            debug_meta["filtered_out_reasons"][k] += int(removed.get(k, 0))

    # One Groq summarization for all players.
    try:
        summaries = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _groq_summarize_batch(players_snippets),
            ),
            timeout=_GROQ_TIMEOUT_S,
        )
        debug_meta["groq_succeeded"] = True
    except Exception as e:
        logger.warning("Groq batch summarization failed, using snippet fallback: %s", e)
        summaries = {}

    out: List[NewsResult] = []
    for player in players:
        snippets, urls = players_snippets.get(player, ([], []))
        structured = summaries.get(player) or {}
        availability = (structured.get("availability") or "").strip()
        minutes_note = (structured.get("minutes_note") or "").strip()
        role_note = (structured.get("role_note") or "").strip()
        summary_text = (structured.get("news_summary") or "").strip()

        # Always produce a per-player summary.
        if not summary_text:
            summary_text = "\n".join(snippets[:10]).strip() if snippets else "No relevant news found."
        if not summary_text:
            summary_text = "No relevant news found."

        bullet_parts = []
        if availability:
            bullet_parts.append(f"Availability: {availability}")
        if minutes_note:
            bullet_parts.append(f"Minutes: {minutes_note}")
        if role_note:
            bullet_parts.append(f"Role: {role_note}")
        if bullet_parts:
            combined_summary = " | ".join(bullet_parts) + f" | Impact: {summary_text}"
        else:
            combined_summary = summary_text

        out.append(
            NewsResult(
                player=player,
                news_summary=combined_summary,
                source_urls=urls[:20],
                retrieved_at_iso=now_iso,
                availability=availability or None,
                minutes_note=minutes_note or None,
                role_note=role_note or None,
                fantasy_impact=summary_text,
            )
        )
    if debug:
        return out, debug_meta
    return out
