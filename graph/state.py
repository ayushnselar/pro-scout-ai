"""
LangGraph state definition for Pro-Scout AI.

The workflow progressively enriches this state as nodes execute.
"""

from __future__ import annotations

from typing import Any, NotRequired, Required, TypedDict

from graph.schemas import DecisionResult, NewsResult, StatsResult


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph nodes.

    Keys are populated over time:
    - entity extraction sets `players`
    - scout agent sets `news`
    - analyst agent sets `stats`
    - gm agent sets `decision`
    """

    query: Required[str]
    players: NotRequired[list[str]]
    news: NotRequired[list[NewsResult]]
    stats: NotRequired[list[StatsResult]]
    decision: NotRequired[DecisionResult]
    scout_debug: NotRequired[dict[str, Any]]
