"""
GM Agent — decision engine synthesizing news and stats.

Step 5.1:
- async make_decision(...)
- strict JSON-only Groq output matching DecisionResult
- one retry on invalid JSON / validation
- sources must be subset of provided URLs
- HOLD with low confidence on uncertainty or failure
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import List, Sequence

from graph.entity_extraction import extract_players
from graph.schemas import Action, DecisionResult, NewsResult, StatsResult
from graph.validate import validate_json_model

logger = logging.getLogger(__name__)

_GROQ_MODEL_DECISION = "llama-3.3-70b-versatile"
_GROQ_TIMEOUT_S = 12
_LOG_CONTENT_LIMIT = 500


def _truncate(text: str, limit: int = _LOG_CONTENT_LIMIT) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _salvage_first_json_object(text: str) -> str | None:
    """
    Attempt to salvage the first JSON object from a model response.

    Strategy:
    - Find the first '{'
    - Use a simple brace-matching scanner (string-aware) to find the matching '}'
    - If matching fails, fall back to substring from first '{' to last '}'
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    end = text.rfind("}")
    if end != -1 and end > start:
        return text[start : end + 1]
    return None


def _get_groq_api_key() -> str | None:
    """Groq API key from environment or Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key

    try:
        import streamlit as st  # type: ignore[import]

        secrets_key = getattr(st, "secrets", None) and st.secrets.get("GROQ_API_KEY")
        if isinstance(secrets_key, str) and secrets_key:
            return secrets_key
    except Exception:
        pass

    return None


def _collect_source_urls(news: Sequence[NewsResult]) -> list[str]:
    """Flatten and deduplicate all source URLs from news results."""
    seen: set[str] = set()
    urls: list[str] = []
    for item in news:
        for url in item.source_urls:
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def _build_decision_prompt(query: str, news: Sequence[NewsResult], stats: Sequence[StatsResult]) -> str:
    """
    Build a structured prompt for the GM decision model.

    Priority rules (from PRD):
    1. Injury or availability risk
    2. Minutes restrictions
    3. Recent performance trend
    4. Season performance
    """
    lines: list[str] = []
    lines.append("You are an expert fantasy basketball GM assistant.")
    lines.append("You MUST output ONLY valid JSON matching the exact schema below.")
    lines.append("")
    lines.append("Schema (JSON):")
    lines.append(
        json.dumps(
            {
                "action": "START | SIT | TRADE_ACCEPT | TRADE_REJECT | HOLD",
                "confidence": 0.0,
                "reasoning": "string",
                "sources": ["string"],
            },
            indent=2,
        )
    )
    lines.append("")
    lines.append("Decision rules (apply in order of importance):")
    lines.append("1. Injury or availability risk is the highest priority.")
    lines.append("2. Minutes restrictions or load management are next.")
    lines.append("3. Recent performance trend (UP/DOWN/STABLE).")
    lines.append("4. Overall season performance.")
    lines.append("")
    lines.append(f"User query: {query}")
    lines.append("")
    lines.append("News signals per player (structured):")
    if news:
        for item in news:
            lines.append(f"- Player: {item.player}")
            lines.append(f"  Availability: {item.availability or 'UNKNOWN'}")
            lines.append(f"  Minutes note: {item.minutes_note or 'UNKNOWN'}")
            lines.append(f"  Role note: {item.role_note or 'UNKNOWN'}")
            lines.append(f"  Fantasy impact: {item.fantasy_impact or item.news_summary}")
            if item.source_urls:
                lines.append(f"  Sources: {', '.join(item.source_urls)}")
    else:
        lines.append("- (no news available)")
    lines.append("")
    lines.append("Statistical signals per player:")
    if stats:
        for stat in stats:
            lines.append(f"- Player: {stat.player}")
            lines.append(
                f"  Season avg points: {stat.season_avg_points if stat.season_avg_points is not None else 'N/A'}"
            )
            lines.append(
                f"  Last 5 avg points: {stat.last5_avg_points if stat.last5_avg_points is not None else 'N/A'}"
            )
            lines.append(f"  Trend: {stat.trend}")
            lines.append(f"  Data source: {stat.data_source}")
    else:
        lines.append("- (no stats available)")
    lines.append("")
    lines.append("")
    lines.append("Rubric for combining signals:")
    lines.append(
        "- If availability or minutes restriction signals indicate "
        "OUT/DOUBTFUL/QUESTIONABLE, heavily discount the player."
    )
    lines.append("- If minutes_note indicates a restriction, weigh recent trend and role less than availability.")
    lines.append(
        "- If news is effectively just a profile (no actionable availability/minutes/role info), "
        "rely primarily on stats."
    )
    lines.append(
        "- If availability is NO_ACTIONABLE_NEWS or missing for all players, treat stats as the primary signal."
    )
    lines.append("")
    lines.append(
        "Respond with ONLY a single JSON object that matches the schema above. "
        "Do not include any extra keys. Do not include markdown or natural language outside of the JSON."
    )
    return "\n".join(lines)


def _call_groq_decision_once(
    query: str,
    news: Sequence[NewsResult],
    stats: Sequence[StatsResult],
    *,
    attempt: int,
) -> DecisionResult:
    """
    Single Groq call that returns a validated DecisionResult.

    Raises ValueError or RuntimeError on any parsing/validation/API issues.
    """
    api_key = _get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    try:
        from groq import Groq
    except Exception as exc:
        raise RuntimeError("groq client not available") from exc

    prompt = _build_decision_prompt(query, news, stats)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=_GROQ_MODEL_DECISION,
        messages=[
            {"role": "system", "content": "You output ONLY JSON that matches the requested schema."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    try:
        content = response.choices[0].message.content  # type: ignore[assignment]
    except (AttributeError, IndexError, KeyError) as exc:
        raise RuntimeError("Unexpected Groq response structure") from exc

    if not isinstance(content, str):
        raise RuntimeError("Groq response content is not a string")

    salvage_attempted = False
    raw_content = content

    # Strict JSON-only enforcement using shared validator, with one salvage attempt.
    try:
        decision = validate_json_model(DecisionResult, raw_content)
    except Exception as exc:
        salvaged = _salvage_first_json_object(raw_content)
        if salvaged and salvaged != raw_content:
            salvage_attempted = True
            try:
                decision = validate_json_model(DecisionResult, salvaged)
            except Exception as exc2:
                logger.warning(
                    "GM decision parse failed (attempt=%s, salvage=%s): %s: %s | raw=%r",
                    attempt,
                    salvage_attempted,
                    type(exc2).__name__,
                    exc2,
                    _truncate(raw_content),
                )
                raise
        else:
            logger.warning(
                "GM decision parse failed (attempt=%s, salvage=%s): %s: %s | raw=%r",
                attempt,
                salvage_attempted,
                type(exc).__name__,
                exc,
                _truncate(raw_content),
            )
            raise

    # Enforce that sources are a subset of known URLs.
    allowed_urls = set(_collect_source_urls(news))
    filtered_sources = [url for url in decision.sources if url in allowed_urls]

    return DecisionResult(
        action=decision.action,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        sources=filtered_sources,
    )


def _fallback_hold_decision(reason: str) -> DecisionResult:
    """Safe HOLD decision when uncertainty or system failure occurs."""
    return DecisionResult(
        action=Action.HOLD,
        confidence=0.15,
        reasoning=reason,
        sources=[],
    )


def _heuristic_fallback_decision(query: str, stats: Sequence[StatsResult]) -> DecisionResult:
    """
    Deterministic heuristic fallback when the LLM is unavailable after retries.

    Rules:
    - start A or B: pick higher last5_avg_points; tie-break season_avg_points; else HOLD 0.2
    - start A: START if trend UP or last5 >= season_avg, else HOLD 0.45
    - trade A for B: accept if B beats A by small margin in both last5 and season, else reject; missing -> HOLD 0.35
    """
    q = query.strip()
    ql = q.lower()

    # Build a case-insensitive stats lookup.
    stat_by_name = {s.player.lower(): s for s in stats if isinstance(s.player, str)}

    def _get_stat(name: str) -> StatsResult | None:
        return stat_by_name.get(name.lower())

    def _score(stat: StatsResult | None) -> tuple[float, float]:
        if not stat or stat.last5_avg_points is None or stat.season_avg_points is None:
            return (-1.0, -1.0)
        return (float(stat.last5_avg_points), float(stat.season_avg_points))

    players = extract_players(q)

    # Start A or B
    if "start" in ql and " or " in ql and len(players) >= 2:
        a, b = players[0], players[1]
        sa, sb = _get_stat(a), _get_stat(b)
        a_last5, a_season = _score(sa)
        b_last5, b_season = _score(sb)

        if a_last5 < 0 and b_last5 < 0:
            return DecisionResult(
                action=Action.HOLD,
                confidence=0.2,
                reasoning="LLM unavailable; insufficient stats to choose, returning HOLD.",
                sources=[],
            )

        chosen = a
        if (b_last5, b_season) > (a_last5, a_season):
            chosen = b

        return DecisionResult(
            action=Action.START,
            confidence=0.55,
            reasoning=(
                f"LLM unavailable; fallback heuristic selected {chosen} due to higher last-5 average points "
                "with season average as tie-break."
            ),
            sources=[],
        )

    # Start single player
    if "start" in ql and len(players) == 1:
        p = players[0]
        s = _get_stat(p)
        if not s or s.last5_avg_points is None or s.season_avg_points is None:
            return DecisionResult(
                action=Action.HOLD,
                confidence=0.2,
                reasoning="LLM unavailable; missing stats for the player, returning HOLD.",
                sources=[],
            )

        if s.trend.name == "UP" or float(s.last5_avg_points) >= float(s.season_avg_points):
            return DecisionResult(
                action=Action.START,
                confidence=0.55,
                reasoning="LLM unavailable; fallback heuristic recommends START based on recent production/trend.",
                sources=[],
            )

        return DecisionResult(
            action=Action.HOLD,
            confidence=0.45,
            reasoning="LLM unavailable; fallback heuristic is cautious (recent production below season), HOLD.",
            sources=[],
        )

    # Trade A for B
    if "trade" in ql and " for " in ql and len(players) >= 2:
        a, b = players[0], players[1]
        sa, sb = _get_stat(a), _get_stat(b)
        if (
            not sa
            or not sb
            or sa.last5_avg_points is None
            or sa.season_avg_points is None
            or sb.last5_avg_points is None
            or sb.season_avg_points is None
        ):
            return DecisionResult(
                action=Action.HOLD,
                confidence=0.35,
                reasoning="LLM unavailable; insufficient stats to evaluate trade, returning HOLD.",
                sources=[],
            )

        margin = 2.0
        better_last5 = float(sb.last5_avg_points) >= float(sa.last5_avg_points) + margin
        better_season = float(sb.season_avg_points) >= float(sa.season_avg_points) + margin
        if better_last5 and better_season:
            return DecisionResult(
                action=Action.TRADE_ACCEPT,
                confidence=0.55,
                reasoning="LLM unavailable; fallback heuristic ACCEPTS (target stronger in last-5 and season).",
                sources=[],
            )
        return DecisionResult(
            action=Action.TRADE_REJECT,
            confidence=0.55,
            reasoning="LLM unavailable; fallback heuristic REJECTS (target not clearly stronger).",
            sources=[],
        )

    return DecisionResult(
        action=Action.HOLD,
        confidence=0.2,
        reasoning="LLM unavailable; unable to apply heuristic confidently, returning HOLD.",
        sources=[],
    )


async def make_decision(
    query: str,
    news: List[NewsResult],
    stats: List[StatsResult],
) -> DecisionResult:
    """
    Asynchronously synthesize signals into a single fantasy decision.

    - Strict JSON-only Groq response validated into DecisionResult.
    - One retry on JSON/validation/API failure.
    - Sources filtered to subset of provided URLs.
    - On failure, returns a low-confidence HOLD decision.
    """
    loop = asyncio.get_event_loop()

    def _blocking_call(attempt: int) -> DecisionResult:
        return _call_groq_decision_once(query, news, stats, attempt=attempt)

    # First attempt
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _blocking_call(1)),
            timeout=_GROQ_TIMEOUT_S,
        )
    except Exception as exc:
        logger.warning("GM decision attempt 1 failed: %s: %s", type(exc).__name__, exc)

    # Retry once
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _blocking_call(2)),
            timeout=_GROQ_TIMEOUT_S,
        )
    except Exception as exc:
        logger.error("GM decision attempt 2 failed: %s: %s", type(exc).__name__, exc)
        fb = _heuristic_fallback_decision(query, stats)
        logger.info(
            "GM using heuristic fallback: action=%s confidence=%.2f",
            fb.action,
            fb.confidence,
        )
        return fb
