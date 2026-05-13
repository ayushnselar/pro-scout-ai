import json
import sys
from types import ModuleType
from typing import Any

import pytest

from agents.gm_agent import make_decision
from graph.schemas import Action, DataSource, DecisionResult, NewsResult, StatsResult, Trend


def _install_fake_groq_for_decision(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contents_by_call: list[str],
) -> None:
    """
    Install a fake `groq` module that returns canned decision JSON content.

    gm_agent imports `Groq` inside `_call_groq_decision_once`, so injecting
    into sys.modules['groq'] is sufficient.
    """

    class _FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = _FakeMessage(content)

    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [_FakeChoice(content)]

    class _FakeCompletions:
        def create(self, *_: Any, **__: Any) -> _FakeResponse:
            if not contents_by_call:
                raise RuntimeError("No more fake Groq contents configured")
            return _FakeResponse(contents_by_call.pop(0))

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class Groq:  # noqa: N801 - external class name
        def __init__(self, *_: Any, **__: Any) -> None:
            self.chat = _FakeChat()

    fake = ModuleType("groq")
    fake.Groq = Groq  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "groq", fake)


@pytest.mark.anyio
async def test_make_decision_valid_json_returns_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    news = [
        NewsResult(
            player="Stephen Curry",
            news_summary="Healthy and starting.",
            source_urls=["https://example.com/a", "https://example.com/b"],
            retrieved_at_iso="2025-01-01T00:00:00Z",
        )
    ]
    stats = [
        StatsResult(
            player="Stephen Curry",
            season_avg_points=28.0,
            last5_avg_points=30.0,
            trend=Trend.UP,
            data_source=DataSource.NBA_API,
        )
    ]

    decision_payload = DecisionResult(
        action=Action.START,
        confidence=0.9,
        reasoning="Elite production and no injury concerns.",
        sources=["https://example.com/a", "https://other.com/ignored"],
    )

    _install_fake_groq_for_decision(
        monkeypatch,
        contents_by_call=[decision_payload.model_dump_json()],
    )

    result = await make_decision("Start Steph?", news, stats)

    assert isinstance(result, DecisionResult)
    assert result.action == Action.START
    assert 0.0 <= result.confidence <= 1.0
    assert "Elite production" in result.reasoning
    # Only URLs present in news should be kept.
    assert result.sources == ["https://example.com/a"]


@pytest.mark.anyio
async def test_make_decision_salvages_embedded_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    news = [
        NewsResult(
            player="Stephen Curry",
            news_summary="Healthy.",
            source_urls=["https://example.com/a"],
            retrieved_at_iso="2025-01-01T00:00:00Z",
        )
    ]
    stats: list[StatsResult] = []

    embedded = (
        "Sure — here's the decision:\n"
        + json.dumps(
            {
                "action": "START",
                "confidence": 0.8,
                "reasoning": "All signals favor starting.",
                "sources": ["https://example.com/a"],
            }
        )
        + "\nThanks!"
    )

    _install_fake_groq_for_decision(monkeypatch, contents_by_call=[embedded])

    result = await make_decision("Start Curry?", news, stats)
    assert result.action == Action.START
    assert result.sources == ["https://example.com/a"]


@pytest.mark.anyio
async def test_make_decision_retries_on_invalid_json_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    news = [
        NewsResult(
            player="LeBron James",
            news_summary="Questionable with ankle soreness.",
            source_urls=["https://example.com/lebron"],
            retrieved_at_iso="2025-01-01T00:00:00Z",
        )
    ]
    stats = []

    # First response: invalid JSON; second: valid DecisionResult JSON.
    valid_second = DecisionResult(
        action=Action.HOLD,
        confidence=0.6,
        reasoning="Injury risk suggests caution.",
        sources=["https://example.com/lebron"],
    )

    _install_fake_groq_for_decision(
        monkeypatch,
        contents_by_call=[
            "not-json",
            valid_second.model_dump_json(),
        ],
    )

    result = await make_decision("Trade LeBron?", news, stats)
    assert isinstance(result, DecisionResult)
    assert result.action == Action.HOLD
    assert "Injury risk" in result.reasoning


@pytest.mark.anyio
async def test_make_decision_invalid_json_twice_returns_hold_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    news: list[NewsResult] = []
    stats: list[StatsResult] = []

    # Both attempts return invalid JSON.
    _install_fake_groq_for_decision(
        monkeypatch,
        contents_by_call=[
            "not-json-1",
            "not-json-2",
        ],
    )

    result = await make_decision("Start someone?", news, stats)

    assert isinstance(result, DecisionResult)
    assert result.action == Action.HOLD
    assert 0.0 <= result.confidence <= 0.3
    assert "llm unavailable" in result.reasoning.lower()


@pytest.mark.anyio
async def test_make_decision_salvage_fails_then_retry_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    news: list[NewsResult] = []
    stats: list[StatsResult] = []

    # First content has braces but is not valid JSON; salvage will not validate.
    bad_embedded = "prefix {not valid json} suffix"
    good = DecisionResult(
        action=Action.HOLD,
        confidence=0.55,
        reasoning="Recovered on retry.",
        sources=[],
    ).model_dump_json()

    _install_fake_groq_for_decision(monkeypatch, contents_by_call=[bad_embedded, good])

    result = await make_decision("Start someone?", news, stats)
    assert result.reasoning == "Recovered on retry."


@pytest.mark.anyio
async def test_make_decision_heuristic_fallback_start_or(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    # Force both LLM attempts to fail.
    _install_fake_groq_for_decision(monkeypatch, contents_by_call=["not-json-1", "not-json-2"])

    news: list[NewsResult] = []
    stats = [
        StatsResult(
            player="Stephen Curry",
            season_avg_points=28.0,
            last5_avg_points=30.0,
            trend=Trend.UP,
            data_source=DataSource.NBA_API,
        ),
        StatsResult(
            player="LeBron James",
            season_avg_points=25.0,
            last5_avg_points=26.0,
            trend=Trend.STABLE,
            data_source=DataSource.NBA_API,
        ),
    ]

    result = await make_decision("Start Stephen Curry or LeBron James tonight?", news, stats)
    assert result.action == Action.START
    assert result.confidence >= 0.5
    assert "fallback heuristic selected" in result.reasoning.lower()
    assert "Stephen Curry" in result.reasoning
    assert result.sources == []
