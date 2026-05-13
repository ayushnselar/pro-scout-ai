"""
Entity extraction for Pro-Scout AI.

Step 2.1:
- `extract_players` implements deterministic parsing for common fantasy phrases.
- `extract_players_llm_fallback` uses Groq (llama3-8b-8192) with strict JSON output
  and Pydantic validation, with a single retry on invalid output.
"""

from __future__ import annotations

import json
import logging
import os
import re
import string
from typing import List, Optional

from pydantic import (
    RootModel,
    ValidationError,
    constr,
    model_validator,
)

logger = logging.getLogger(__name__)


class _PlayersList(RootModel[List[constr(min_length=1)]]):  # type: ignore[type-arg]
    """Pydantic RootModel enforcing a JSON list of player name strings."""

    @model_validator(mode="after")
    def _validate_non_empty_and_non_blank(self) -> "_PlayersList":
        # Strip whitespace and ensure at least one non-blank name.
        cleaned = [name.strip() for name in self.root if isinstance(name, str) and name.strip()]
        if not cleaned:
            raise ValueError("Players list must contain at least one non-empty string")
        self.root = cleaned
        return self


# Simple alias mapping for common NBA star shorthands.
_ALIAS_MAP = {
    "lebron": "LeBron James",
    "lebron james": "LeBron James",
    "lbj": "LeBron James",
    "steph": "Stephen Curry",
    "steph curry": "Stephen Curry",
    "curry": "Stephen Curry",
    "kd": "Kevin Durant",
    "kevin durant": "Kevin Durant",
    "ad": "Anthony Davis",
    "anthony davis": "Anthony Davis",
    "tatum": "Jayson Tatum",
    "jayson tatum": "Jayson Tatum",
}


_TRADE_PATTERN = re.compile(r"trade\s+(?P<a>.+?)\s+for\s+(?P<b>.+)", re.IGNORECASE)
_START_PATTERN = re.compile(r"\bstart\s+(?P<names>.+)", re.IGNORECASE)
_SIT_PATTERN = re.compile(r"\bsit\s+(?P<names>.+)", re.IGNORECASE)

# Matchup tokens: truncate fragment at first occurrence (keep only tokens before).
_MATCHUP_TOKENS = frozenset({"vs", "v", "versus", "against", "at"})

_TRAILING_FILLER_TOKENS = {
    "tonight",
    "today",
    "now",
    "please",
    "vs",
    "versus",
    "against",
    "this",
    "week",
    "tomorrow",
}


def _normalize_player_name(raw: str) -> Optional[str]:
    """
    Normalize a raw name fragment into a canonical player name.

    - Applies alias mapping for common shorthand.
    - Falls back to simple title-casing of words.
    """
    text = raw.strip().strip(string.punctuation)
    if not text:
        return None

    key = text.lower()
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]

    # Basic heuristic: title-case each token for "First Last" style names.
    tokens = [t for t in re.split(r"\s+", text) if t]
    if not tokens:
        return None
    return " ".join(token.capitalize() for token in tokens)


def _cleanup_fragment(chunk: str) -> str:
    """
    Cleanup a raw name fragment: truncate at matchup tokens, then strip trailing
    filler words and punctuation.

    Matchup tokens: vs, v, versus, against, @, at (whole-token or @ in token).
    Examples:
    - "curry vs lakers this week" -> "curry"
    - "kd tonight?" -> "kd"
    - "LeBron now" -> "LeBron"
    """
    text = chunk.strip().strip(string.punctuation)
    if not text:
        return ""

    tokens = [t for t in re.split(r"\s+", text) if t]
    # Truncate at first matchup token (keep only tokens before it).
    kept: List[str] = []
    for t in tokens:
        clean = t.strip(string.punctuation).lower()
        if clean in _MATCHUP_TOKENS:
            break
        if "@" in t:
            before = t.split("@")[0].strip(string.punctuation)
            if before:
                kept.append(before)
            break
        kept.append(t)

    # Strip trailing filler words and punctuation.
    while kept:
        last = kept[-1].strip(string.punctuation).lower()
        if last in _TRAILING_FILLER_TOKENS:
            kept.pop()
        else:
            break

    return " ".join(kept)


def _split_candidate_names(chunk: str) -> List[str]:
    """
    Split a text chunk that may contain multiple player names.

    Supports separators like "or", "and", commas.
    """
    # Normalize common separators to comma.
    normalized = re.sub(r"\s+(or|and)\s+", ",", chunk, flags=re.IGNORECASE)
    parts = [part.strip() for part in normalized.split(",")]
    return [part for part in parts if part]


def extract_players(query: str) -> List[str]:
    """
    Deterministically extract player names from a user query.

    Handles patterns such as:
    - "Trade A for B"
    - "start A"
    - "sit A"
    - "start A or B"
    """
    candidates: List[str] = []

    # Trade pattern: "trade A for B"
    trade_match = _TRADE_PATTERN.search(query)
    if trade_match:
        a_raw = trade_match.group("a")
        b_raw = trade_match.group("b")
        for frag in (a_raw, b_raw):
            name = _normalize_player_name(frag)
            if name and name not in candidates:
                candidates.append(name)
        if candidates:
            return candidates

    # Start / sit patterns.
    for pattern in (_START_PATTERN, _SIT_PATTERN):
        match = pattern.search(query)
        if match:
            names_chunk = match.group("names")
            for frag in _split_candidate_names(names_chunk):
                cleaned = _cleanup_fragment(frag)
                if not cleaned:
                    continue
                name = _normalize_player_name(cleaned)
                if name and name not in candidates:
                    candidates.append(name)

    return candidates


def _get_groq_api_key() -> Optional[str]:
    """Retrieve Groq API key from environment or Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key

    # Optional Streamlit secrets support; avoid hard failure if not available.
    try:
        import streamlit as st  # type: ignore[import]

        secrets_key = st.secrets.get("GROQ_API_KEY")  # type: ignore[attr-defined]
        if isinstance(secrets_key, str) and secrets_key:
            return secrets_key
    except Exception:
        # Streamlit not available or secrets missing; handled by caller.
        pass

    return None


def _call_groq_players_once(query: str) -> List[str]:
    """
    Single Groq call to extract player names as a JSON list of strings.

    Raises on network / API / validation errors so that the caller can
    perform a retry and ultimately fall back safely.
    """
    api_key = _get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    try:
        from groq import Groq  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - import-time environment issue
        raise RuntimeError("groq client library is not available") from exc

    client = Groq(api_key=api_key)

    system_prompt = (
        "You extract NBA player names from fantasy basketball questions.\n"
        "Return ONLY a JSON array of full player name strings.\n"
        'Example: ["LeBron James", "Stephen Curry"]\n'
        "Do not include any explanation or extra keys, output must be raw JSON."
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
    )

    try:
        content = response.choices[0].message.content  # type: ignore[assignment]
    except (AttributeError, IndexError, KeyError) as exc:
        raise RuntimeError("Unexpected Groq response structure") from exc

    if not isinstance(content, str):
        raise RuntimeError("Groq response content is not a string")

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Groq response was not valid JSON") from exc

    try:
        model = _PlayersList.model_validate(raw)
    except ValidationError as exc:
        raise ValueError("Groq JSON failed schema validation") from exc

    return model.root


def extract_players_llm_fallback(query: str) -> List[str]:
    """
    LLM-based fallback for entity extraction using Groq llama3-8b-8192.

    Behavior:
    - Enforces strict JSON list-of-strings output.
    - Retries once on parsing / validation failures.
    - On repeated failure or configuration issues, returns an empty list.
    """
    # First attempt
    try:
        return _call_groq_players_once(query)
    except Exception as exc:
        logger.warning("Groq entity extraction attempt 1 failed: %s", exc)

    # Retry once
    try:
        return _call_groq_players_once(query)
    except Exception as exc:
        logger.error("Groq entity extraction attempt 2 failed: %s", exc)
        return []
