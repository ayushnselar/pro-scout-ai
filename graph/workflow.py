"""
LangGraph workflow for Pro-Scout AI.

Step 6.1:
- Entity extraction node
- Fan-out to Scout (async) and Analyst (threaded) in parallel
- Fan-in at GM decision node
- Output returned as DecisionEnvelope via run_workflow(query).
"""

from __future__ import annotations

import logging
from time import monotonic
from typing import Awaitable, Callable

from langgraph.graph import END, StateGraph

from agents.analyst_agent import get_stats
from agents.gm_agent import make_decision
from agents.scout_agent import get_news
from graph.entity_extraction import extract_players, extract_players_llm_fallback
from graph.schemas import DecisionEnvelope, NewsResult, StatsResult
from graph.state import AgentState
from graph.validate import validate_decision_envelope

logger = logging.getLogger(__name__)


def _safe_call_async(func: Callable[..., Awaitable], *args, **kwargs):
    """
    Wrap an async callable and log any exceptions, returning a safe default.

    This helper is used only where we need extra safety; most agent-level
    fault tolerance is implemented inside each agent.
    """

    async def _runner():
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Async call %s failed: %s", func.__name__, exc)
            return None

    return _runner()


async def _entity_extraction_node(state: AgentState) -> AgentState:
    """Extract player entities from the user query."""
    query = state["query"]
    players = extract_players(query)
    source = "deterministic"
    if not players:
        # Fallback to LLM-based extraction; still safe if that returns [].
        try:
            players = extract_players_llm_fallback(query)
            source = "llm_fallback"
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("LLM entity extraction failed: %s", exc)
            players = []

    logger.info("Entity extraction: %d player(s) via %s", len(players), source)
    # Only update the players key; other keys (like query) are preserved by LangGraph.
    return {"players": players}


async def _scout_node(state: AgentState) -> AgentState:
    """Run Scout Agent to fetch news in parallel for all players."""
    players = state.get("players") or []
    try:
        news, scout_debug = await get_news(players, user_query=state.get("query"), debug=True)  # type: ignore[misc]
    except Exception as exc:  # pragma: no cover - extra guard
        logger.error("Scout agent failed: %s", exc)
        news = []
        scout_debug = {
            "per_player": {},
            "filtered_out_reasons": {"domain": 0, "keyword": 0, "empty": 0},
            "groq_succeeded": False,
        }

    # Only update the news and debug keys.
    n_news = len(news)
    groq_ok = bool((scout_debug or {}).get("groq_succeeded"))
    logger.info(
        "Scout: %d player(s) -> %d news row(s), groq_summarization_ok=%s",
        len(players),
        n_news,
        groq_ok,
    )
    return {"news": news, "scout_debug": scout_debug}


def _analyst_node(state: AgentState) -> AgentState:
    """Run Analyst Agent to fetch stats in parallel via ThreadPoolExecutor."""
    players = state.get("players") or []
    try:
        stats: list[StatsResult] = get_stats(players)
    except Exception as exc:  # pragma: no cover - extra guard
        logger.error("Analyst agent failed: %s", exc)
        stats = []

    # Only update the stats key.
    n_ok = sum(1 for s in stats if s.data_source.name == "NBA_API")
    logger.info(
        "Analyst: %d player(s) -> %d nba_api row(s), %d unavailable",
        len(players),
        n_ok,
        len(stats) - n_ok,
    )
    return {"stats": stats}


async def _gm_node(state: AgentState) -> AgentState:
    """Run GM Agent to synthesize signals into a final decision."""
    query = state["query"]
    news: list[NewsResult] = state.get("news", []) or []
    stats: list[StatsResult] = state.get("stats", []) or []

    try:
        decision = await make_decision(query, news, stats)
    except Exception as exc:  # pragma: no cover - extra guard
        logger.error("GM agent failed unexpectedly: %s", exc)
        # gm_agent already has its own fallback, so this should be rare.
        raise

    # Only update the decision key.
    logger.info("GM decision: action=%s confidence=%.2f", decision.action, decision.confidence)
    return {"decision": decision}


async def _merge_results_node(state: AgentState) -> AgentState:
    """
    Explicit fan-in join node.

    This node exists solely to ensure that both scout_agent and analyst_agent
    have completed before gm_agent runs. It does not modify state.
    """
    logger.info("Workflow fan-in: scout + analyst complete")
    return {}


def _build_graph() -> StateGraph:
    """Construct the LangGraph StateGraph for the Pro-Scout workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("entity_extraction", _entity_extraction_node)
    graph.add_node("scout_agent", _scout_node)
    graph.add_node("analyst_agent", _analyst_node)
    graph.add_node("merge_results", _merge_results_node)
    graph.add_node("gm_agent", _gm_node)

    graph.set_entry_point("entity_extraction")

    # Fan-out: entity_extraction -> scout_agent and analyst_agent in parallel.
    graph.add_edge("entity_extraction", "scout_agent")
    graph.add_edge("entity_extraction", "analyst_agent")

    # Fan-in at merge_results: requires both scout and analyst before GM.
    graph.add_edge("scout_agent", "merge_results")
    graph.add_edge("analyst_agent", "merge_results")

    # Single GM node after explicit merge.
    graph.add_edge("merge_results", "gm_agent")

    # Terminal node.
    graph.add_edge("gm_agent", END)

    return graph


_APP = _build_graph().compile()


async def _run_graph(query: str) -> AgentState:
    initial_state: AgentState = {"query": query}
    t0 = monotonic()
    logger.info("Workflow start: query_len=%d", len(query))
    try:
        return await _APP.ainvoke(initial_state)  # type: ignore[return-value]
    finally:
        logger.info("Workflow graph complete in %.2fs", monotonic() - t0)


async def run_workflow(query: str) -> DecisionEnvelope:
    """
    Execute the full multi-agent workflow for a single user query.

    This orchestrates:
    - Entity extraction
    - Parallel Scout (async) and Analyst (threaded) agents
    - GM decision synthesis
    - Final DecisionEnvelope construction and validation
    """
    final_state: AgentState = await _run_graph(query)

    envelope_dict = {
        "query": final_state["query"],
        "players": final_state.get("players") or [],
        "news": final_state.get("news") or [],
        "stats": final_state.get("stats") or [],
        "decision": final_state["decision"],
        "data_freshness_minutes": 5,
    }

    # Enforce schema-first design at the workflow boundary.
    return validate_decision_envelope(envelope_dict)


async def run_workflow_with_debug(query: str) -> tuple[DecisionEnvelope, dict]:
    """
    Execute the workflow and return (DecisionEnvelope, debug_metadata).

    Debug metadata includes Scout-agent retrieval/filtering statistics.
    """
    final_state: AgentState = await _run_graph(query)

    envelope_dict = {
        "query": final_state["query"],
        "players": final_state.get("players") or [],
        "news": final_state.get("news") or [],
        "stats": final_state.get("stats") or [],
        "decision": final_state["decision"],
        "data_freshness_minutes": 5,
    }
    envelope = validate_decision_envelope(envelope_dict)
    debug_meta = final_state.get("scout_debug") or {}
    return envelope, debug_meta
