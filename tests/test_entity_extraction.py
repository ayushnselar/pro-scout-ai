import json
import sys
from types import ModuleType
from typing import Any

import pytest

from graph.entity_extraction import extract_players, extract_players_llm_fallback


def test_extract_players_trade_aliases() -> None:
    players = extract_players("Trade LeBron for Steph?")
    assert players == ["LeBron James", "Stephen Curry"]


def test_extract_players_start_or_aliases() -> None:
    players = extract_players("Should I start tatum or kd tonight?")
    assert players == ["Jayson Tatum", "Kevin Durant"]


def test_extract_players_sit_and_comma_split() -> None:
    players = extract_players("sit Anthony Davis and LeBron")
    assert players == ["Anthony Davis", "LeBron James"]


def test_extract_players_trailing_filler_words() -> None:
    players = extract_players("Start curry vs lakers this week")
    assert players == ["Stephen Curry"]


def _install_fake_groq(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contents_by_call: list[str],
    call_counter: list[int],
) -> None:
    """
    Install a fake `groq` module into sys.modules.

    The real implementation imports `from groq import Groq` inside the function,
    so injecting into `sys.modules["groq"]` ensures our fake is used.
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
            call_counter[0] += 1
            if not contents_by_call:
                raise RuntimeError("No more fake Groq contents configured")
            return _FakeResponse(contents_by_call.pop(0))

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class Groq:  # noqa: N801 - matches external library class name
        def __init__(self, *_: Any, **__: Any) -> None:
            self.chat = _FakeChat()

    fake = ModuleType("groq")
    fake.Groq = Groq  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "groq", fake)


def test_extract_players_llm_fallback_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    call_counter = [0]
    contents = [json.dumps(["LeBron James", "Stephen Curry"])]
    _install_fake_groq(monkeypatch, contents_by_call=contents, call_counter=call_counter)

    players = extract_players_llm_fallback("Start LeBron or Curry?")
    assert players == ["LeBron James", "Stephen Curry"]
    assert call_counter[0] == 1


def test_extract_players_llm_fallback_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    call_counter = [0]
    contents = [
        "not-json",
        json.dumps(["Jayson Tatum"]),
    ]
    _install_fake_groq(monkeypatch, contents_by_call=contents, call_counter=call_counter)

    players = extract_players_llm_fallback("start tatum")
    assert players == ["Jayson Tatum"]
    assert call_counter[0] == 2


def test_extract_players_llm_fallback_double_failure_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    call_counter = [0]
    contents = [
        json.dumps([]),  # fails schema: must contain at least one non-empty name
        json.dumps(["   "]),  # fails schema: non-empty names
    ]
    _install_fake_groq(monkeypatch, contents_by_call=contents, call_counter=call_counter)

    players = extract_players_llm_fallback("start someone")
    assert players == []
    assert call_counter[0] == 2
