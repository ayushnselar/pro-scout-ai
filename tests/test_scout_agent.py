import json
import re
import sys
import threading
from types import ModuleType
from typing import Any

import pytest

from agents.scout_agent import get_news


def _install_fake_duckduckgo_search(
    monkeypatch: pytest.MonkeyPatch,
    *,
    results_by_query_substr: dict[str, list[dict[str, Any]]],
    raise_for_queries_containing: set[str] | None = None,
    call_counter: list[int] | None = None,
) -> None:
    """
    Install a fake `ddgs` module into sys.modules.

    The real implementation imports `DDGS` inside `_ddg_text_sync`, and uses it
    as a context manager. This fake mirrors that minimal surface.
    """
    lock = threading.Lock()

    class DDGS:  # noqa: N801 - matches external library class name
        def __enter__(self) -> "DDGS":
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def text(self, query: str, *_: Any, **__: Any) -> list[dict[str, Any]]:
            if raise_for_queries_containing and any(s in query for s in raise_for_queries_containing):
                raise RuntimeError("Simulated DuckDuckGo failure")

            if call_counter is not None:
                with lock:
                    call_counter[0] += 1

            # Pick the first matching stub key; else return empty list.
            for key, results in results_by_query_substr.items():
                if key in query:
                    return results
            return []

    fake = ModuleType("ddgs")
    fake.DDGS = DDGS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ddgs", fake)


def _install_fake_groq(
    monkeypatch: pytest.MonkeyPatch,
    *,
    content_by_call: list[str],
    raise_on_call: set[int] | None = None,
    call_counter: list[int] | None = None,
) -> None:
    """
    Install a fake `groq` module into sys.modules.

    The real implementation imports `Groq` inside `_groq_summarize_batch`.
    """
    lock = threading.Lock()

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
            if call_counter is not None:
                with lock:
                    call_counter[0] += 1
                    current_call = call_counter[0]
            else:
                current_call = 1

            if raise_on_call and current_call in raise_on_call:
                raise RuntimeError("Simulated Groq failure")

            if not content_by_call:
                raise RuntimeError("No more fake Groq contents configured")
            return _FakeResponse(content_by_call.pop(0))

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class Groq:  # noqa: N801 - matches external library class name
        def __init__(self, *_: Any, **__: Any) -> None:
            self.chat = _FakeChat()

    fake = ModuleType("groq")
    fake.Groq = Groq  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "groq", fake)


@pytest.mark.anyio
async def test_get_news_success_batch_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    # Avoid real nba_api calls in query building.
    monkeypatch.setattr("agents.scout_agent._get_team", lambda p: None, raising=False)

    ddg_calls = [0]
    _install_fake_duckduckgo_search(
        monkeypatch,
        results_by_query_substr={
            "NBA injury update": [{"href": "https://a.example", "body": "injury snippet"}],
            "NBA status tonight": [{"href": "https://b.example", "body": "status snippet"}],
            "NBA minutes restriction": [{"href": "https://c.example", "body": "minutes snippet"}],
        },
        call_counter=ddg_calls,
    )

    groq_calls = [0]
    _install_fake_groq(
        monkeypatch,
        content_by_call=[
            json.dumps(
                {
                    "Stephen Curry": {
                        "availability": "AVAILABLE",
                        "minutes_note": "no restriction",
                        "role_note": "starting",
                        "news_summary": "Healthy; strong start recommendation.",
                    },
                    "LeBron James": {
                        "availability": "PROBABLE",
                        "minutes_note": "monitor minutes",
                        "role_note": "starting",
                        "news_summary": "Probable; monitor minutes.",
                    },
                }
            )
        ],
        call_counter=groq_calls,
    )

    players = ["Stephen Curry", "LeBron James"]
    results = await get_news(players)

    assert len(results) == 2
    assert [r.player for r in results] == players
    assert "Healthy; strong start recommendation." in results[0].news_summary
    assert "Probable; monitor minutes." in results[1].news_summary
    assert results[0].source_urls == ["https://a.example", "https://b.example", "https://c.example"]
    assert results[1].source_urls == ["https://a.example", "https://b.example", "https://c.example"]
    assert re.match(r".+Z$", results[0].retrieved_at_iso)
    assert groq_calls[0] == 1  # ONE batch call
    assert ddg_calls[0] == 3 * len(players)  # 3 queries per player


@pytest.mark.anyio
async def test_get_news_groq_failure_falls_back_to_snippets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    monkeypatch.setattr("agents.scout_agent._get_team", lambda p: None, raising=False)

    _install_fake_duckduckgo_search(
        monkeypatch,
        results_by_query_substr={
            "NBA injury update": [{"href": "https://a.example", "body": "injury snippet"}],
            "NBA status tonight": [{"href": "https://b.example", "body": "status snippet"}],
            "NBA minutes restriction": [{"href": "https://c.example", "body": "minutes snippet"}],
        },
    )

    groq_calls = [0]
    _install_fake_groq(
        monkeypatch,
        content_by_call=[json.dumps({"Stephen Curry": "unused"})],
        raise_on_call={1},
        call_counter=groq_calls,
    )

    results = await get_news(["Stephen Curry"])
    assert len(results) == 1
    assert results[0].player == "Stephen Curry"
    # Summary should include concatenated snippets in some order.
    assert "injury snippet" in results[0].news_summary
    assert "status snippet" in results[0].news_summary
    assert "minutes snippet" in results[0].news_summary
    assert groq_calls[0] == 1


@pytest.mark.anyio
async def test_get_news_ddg_failure_returns_valid_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    monkeypatch.setattr("agents.scout_agent._get_team", lambda p: None, raising=False)

    _install_fake_duckduckgo_search(
        monkeypatch,
        results_by_query_substr={},
        raise_for_queries_containing={"NBA injury update", "NBA status tonight", "NBA minutes restriction"},
    )

    # Even if Groq returns something, lack of snippets should still be safe.
    _install_fake_groq(
        monkeypatch,
        content_by_call=[json.dumps({"Stephen Curry": ""})],
    )

    results = await get_news(["Stephen Curry"])
    assert len(results) == 1
    assert results[0].player == "Stephen Curry"
    assert results[0].source_urls == []
    assert results[0].news_summary  # non-empty min_length=1


@pytest.mark.anyio
async def test_get_news_filters_junk_and_ensures_multi_player_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    monkeypatch.setattr("agents.scout_agent._get_team", lambda p: None, raising=False)

    ddg_calls = [0]
    _install_fake_duckduckgo_search(
        monkeypatch,
        results_by_query_substr={
            # Include junk domains/keywords/empty that should be filtered out.
            "Stephen Curry NBA injury update": [
                {"href": "https://en.wikipedia.org/wiki/Stephen", "body": "Saint Stephen was..."},
                {"href": "https://behindthename.com/name/stephen", "body": "Meaning of Stephen..."},
                {
                    "href": "https://news.example/nba",
                    "title": "Curry injury update",
                    "body": "Out with ankle soreness.",
                },
                {"href": "https://ok.example/empty", "body": "   "},
            ],
            "Stephen Curry NBA status tonight": [
                {"href": "https://nameberry.com/b/boy-baby-name-stephen", "body": "name meaning..."},
                {"href": "https://news.example/status", "body": "Status tonight: probable."},
            ],
            "Stephen Curry NBA minutes restriction": [
                {"href": "https://britannica.com/topic/Stephen", "body": "definition ..."},
            ],
        },
        call_counter=ddg_calls,
    )

    groq_calls = [0]
    # Groq returns a partial map (missing LeBron). Must still return both players.
    _install_fake_groq(
        monkeypatch,
        content_by_call=[
            json.dumps(
                {
                    "Stephen Curry": {
                        "availability": "QUESTIONABLE",
                        "minutes_note": "monitor minutes",
                        "role_note": "starting",
                        "news_summary": "Probable; monitor ankle.",
                    }
                }
            )
        ],
        call_counter=groq_calls,
    )

    players = ["Stephen Curry", "LeBron James"]
    results, meta = await get_news(players, debug=True)

    assert [r.player for r in results] == players
    assert groq_calls[0] == 1  # single batch call
    assert ddg_calls[0] >= 3  # at least 3 Curry queries; LeBron may short-circuit on errors/timeouts

    # Curry should not include junk sources; only kept URLs remain.
    curry = results[0]
    assert all("wikipedia.org" not in u for u in curry.source_urls)
    assert all("behindthename.com" not in u for u in curry.source_urls)
    assert all("nameberry.com" not in u for u in curry.source_urls)
    assert all("britannica.com" not in u for u in curry.source_urls)
    assert curry.news_summary  # from Groq

    # LeBron has no kept snippets (DDG stubs don't include LeBron-specific results here).
    # Must still return a NewsResult and use the exact fallback string.
    lebron = results[1]
    assert lebron.news_summary == "No relevant news found."

    assert meta["groq_succeeded"] is True
    assert "per_player" in meta and "filtered_out_reasons" in meta


@pytest.mark.anyio
async def test_get_news_ddg_timeouts_still_return_per_player_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    # Avoid real nba_api in query building.
    monkeypatch.setattr("agents.scout_agent._get_team", lambda p: None, raising=False)

    # Simulate very slow DDG sync calls.
    import time as _time

    def slow_ddg_sync(query: str) -> list[dict[str, Any]]:
        _time.sleep(0.2)
        return []

    # Patch timeouts to be very small so the test runs quickly.
    import agents.scout_agent as sa

    monkeypatch.setattr(sa, "_ddg_text_sync", slow_ddg_sync, raising=True)
    monkeypatch.setattr(sa, "_DDG_QUERY_TIMEOUT_S", 0.01, raising=False)
    monkeypatch.setattr(sa, "_DDG_PLAYER_TIMEOUT_S", 0.05, raising=False)
    monkeypatch.setattr(sa, "_DDG_GLOBAL_TIMEOUT_S", 0.1, raising=False)

    # Groq is not called meaningfully here; just return empty structured map.
    _install_fake_groq(
        monkeypatch,
        content_by_call=[json.dumps({})],
    )

    players = ["Stephen Curry", "LeBron James"]
    results, meta = await get_news(players, debug=True)

    # Even with DDG timeouts, we must return one NewsResult per player.
    assert [r.player for r in results] == players
    for r in results:
        assert r.news_summary  # fallback text

    assert "per_player" in meta
