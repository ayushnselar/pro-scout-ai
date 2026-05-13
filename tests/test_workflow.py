from __future__ import annotations

from typing import List

import pytest

from graph.schemas import Action, DataSource, DecisionEnvelope, DecisionResult, NewsResult, StatsResult, Trend


@pytest.mark.anyio
async def test_run_workflow_success_calls_all_agents_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import graph.workflow as wf

    get_news_calls = {"count": 0}
    get_stats_calls = {"count": 0}
    make_decision_calls = {"count": 0}

    async def fake_get_news(
        players: List[str],
        *,
        user_query: str | None = None,
        debug: bool = False,
    ):
        get_news_calls["count"] += 1
        assert players == ["Stephen Curry"]
        news = [
            NewsResult(
                player="Stephen Curry",
                news_summary="Healthy and starting.",
                source_urls=["https://example.com/steph"],
                retrieved_at_iso="2025-01-01T00:00:00Z",
            )
        ]
        if debug:
            return news, {"groq_succeeded": True}
        return news

    def fake_get_stats(players: List[str]) -> List[StatsResult]:
        get_stats_calls["count"] += 1
        assert players == ["Stephen Curry"]
        return [
            StatsResult(
                player="Stephen Curry",
                season_avg_points=28.0,
                last5_avg_points=30.0,
                trend=Trend.UP,
                data_source=DataSource.NBA_API,
            )
        ]

    async def fake_make_decision(
        query: str,
        news: List[NewsResult],
        stats: List[StatsResult],
    ) -> DecisionResult:
        make_decision_calls["count"] += 1
        assert query == "Start Stephen Curry?"
        assert len(news) == 1
        assert len(stats) == 1
        return DecisionResult(
            action=Action.START,
            confidence=0.9,
            reasoning="All signals favor starting Curry.",
            sources=["https://example.com/steph"],
        )

    monkeypatch.setattr(wf, "get_news", fake_get_news)
    monkeypatch.setattr(wf, "get_stats", fake_get_stats)
    monkeypatch.setattr(wf, "make_decision", fake_make_decision)

    # Use real deterministic extraction: "Start Stephen Curry?" -> ["Stephen Curry"]
    envelope = await wf.run_workflow("Start Stephen Curry?")

    assert isinstance(envelope, DecisionEnvelope)
    assert envelope.query == "Start Stephen Curry?"
    assert envelope.players == ["Stephen Curry"]
    assert envelope.decision.action == Action.START
    assert 0.0 <= envelope.decision.confidence <= 1.0
    assert envelope.data_freshness_minutes == 5

    # Ensure fan-out and join behavior.
    assert get_news_calls["count"] == 1
    assert get_stats_calls["count"] == 1
    assert make_decision_calls["count"] == 1


@pytest.mark.anyio
async def test_run_workflow_empty_extraction_holds_with_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import graph.workflow as wf

    # Force both deterministic and LLM extraction to return no players.
    def fake_extract_players(_: str) -> list[str]:
        return []

    def fake_extract_players_llm_fallback(_: str) -> list[str]:
        return []

    get_news_calls = {"count": 0}
    get_stats_calls = {"count": 0}
    make_decision_calls = {"count": 0}

    async def fake_get_news(
        players: List[str],
        *,
        user_query: str | None = None,
        debug: bool = False,
    ):
        get_news_calls["count"] += 1
        # Should be called even if players is empty.
        assert players == []
        if debug:
            return [], {"groq_succeeded": False}
        return []

    def fake_get_stats(players: List[str]) -> List[StatsResult]:
        get_stats_calls["count"] += 1
        assert players == []
        return []

    async def fake_make_decision(
        query: str,
        news: List[NewsResult],
        stats: List[StatsResult],
    ) -> DecisionResult:
        make_decision_calls["count"] += 1
        assert news == []
        assert stats == []
        return DecisionResult(
            action=Action.HOLD,
            confidence=0.2,
            reasoning="No identifiable players; defaulting to HOLD.",
            sources=[],
        )

    monkeypatch.setattr(wf, "extract_players", fake_extract_players)
    monkeypatch.setattr(wf, "extract_players_llm_fallback", fake_extract_players_llm_fallback)
    monkeypatch.setattr(wf, "get_news", fake_get_news)
    monkeypatch.setattr(wf, "get_stats", fake_get_stats)
    monkeypatch.setattr(wf, "make_decision", fake_make_decision)

    envelope = await wf.run_workflow("What should I do?")  # no names in query

    assert isinstance(envelope, DecisionEnvelope)
    assert envelope.players == []
    assert envelope.decision.action == Action.HOLD
    assert 0.0 <= envelope.decision.confidence <= 0.3
    assert envelope.data_freshness_minutes == 5

    # Ensure workflow still fans out and joins correctly.
    assert get_news_calls["count"] == 1
    assert get_stats_calls["count"] == 1
    assert make_decision_calls["count"] == 1
