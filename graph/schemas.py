"""
Pydantic schemas for Pro-Scout AI.

All agent and workflow I/O must conform to these models.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, confloat


class Trend(str, Enum):
    """Performance trend classification for a player."""

    UP = "UP"
    DOWN = "DOWN"
    STABLE = "STABLE"


class DataSource(str, Enum):
    """Source of statistical data."""

    NBA_API = "nba_api"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"


class Action(str, Enum):
    """GM decision action."""

    START = "START"
    SIT = "SIT"
    TRADE_ACCEPT = "TRADE_ACCEPT"
    TRADE_REJECT = "TRADE_REJECT"
    HOLD = "HOLD"


class NewsResult(BaseModel):
    """
    Structured news / qualitative context for a single player.

    Mirrors the PRD schema:
    {
        "player": "string",
        "news_summary": "string",
        "source_urls": ["string"],
        "retrieved_at_iso": "string"
    }
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    player: str = Field(..., min_length=1)
    news_summary: str = Field(..., min_length=1)
    source_urls: List[str] = Field(default_factory=list)
    retrieved_at_iso: str = Field(..., min_length=1)
    availability: str | None = None
    minutes_note: str | None = None
    role_note: str | None = None
    fantasy_impact: str | None = None


class StatsResult(BaseModel):
    """
    Structured statistical metrics for a single player.

    Mirrors the PRD schema:
    {
        "player": "string",
        "season_avg_points": number,
        "last5_avg_points": number,
        "trend": "UP | DOWN | STABLE",
        "data_source": "nba_api | fallback | unavailable"
    }
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    player: str = Field(..., min_length=1)
    season_avg_points: float | None = Field(default=None, ge=0)
    last5_avg_points: float | None = Field(default=None, ge=0)
    trend: Trend
    data_source: DataSource = Field(default=DataSource.NBA_API)


class DecisionResult(BaseModel):
    """
    Final GM decision for the query.

    Mirrors the PRD schema:
    {
        "action": "START | SIT | TRADE_ACCEPT | TRADE_REJECT | HOLD",
        "confidence": 0.0-1.0,
        "reasoning": "string",
        "sources": ["string"]
    }
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Action
    confidence: confloat(ge=0.0, le=1.0)  # type: ignore[type-arg]
    reasoning: str = Field(..., min_length=1)
    sources: List[str] = Field(default_factory=list)


class DecisionEnvelope(BaseModel):
    """
    Top-level structured output for the LangGraph workflow.

    This serves as the final, schema-validated response returned to the
    Streamlit frontend.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(..., min_length=1)
    players: List[str] = Field(default_factory=list)
    news: List[NewsResult] = Field(default_factory=list)
    stats: List[StatsResult] = Field(default_factory=list)
    decision: DecisionResult
    # Approximate staleness of underlying data in minutes.
    data_freshness_minutes: int = Field(default=5, ge=0)
