from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from agents.analyst_agent import get_stats
from graph.schemas import DataSource, Trend


def _install_fake_nba_api_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake nba_api that returns a simple game log with PTS."""

    # Fake static players
    class _Players:
        @staticmethod
        def find_players_by_full_name(name: str) -> list[dict[str, Any]]:
            return [{"id": 123, "full_name": name}]

    # Fake PlayerGameLog endpoint
    class _PlayerGameLog:
        def __init__(self, player_id: int, season: str) -> None:  # noqa: D401
            self.player_id = player_id
            self.season = season

        def get_data_frames(self) -> list[pd.DataFrame]:
            # 6 games, most recent first: 30, 28, 25, 22, 20, 18
            pts = [30, 28, 25, 22, 20, 18]
            df = pd.DataFrame({"PTS": pts})
            return [df]

    # Build fake nba_api.stats.static.players
    fake_static = ModuleType("nba_api.stats.static.players")
    fake_static.find_players_by_full_name = _Players.find_players_by_full_name  # type: ignore[attr-defined]

    fake_static_pkg = ModuleType("nba_api.stats.static")
    fake_static_pkg.players = fake_static  # type: ignore[attr-defined]

    # Build fake nba_api.stats.endpoints.playergamelog
    fake_endpoints_playergamelog = ModuleType("nba_api.stats.endpoints.playergamelog")
    fake_endpoints_playergamelog.PlayerGameLog = _PlayerGameLog  # type: ignore[attr-defined]

    fake_endpoints_pkg = ModuleType("nba_api.stats.endpoints")
    fake_endpoints_pkg.playergamelog = fake_endpoints_playergamelog  # type: ignore[attr-defined]

    # Root nba_api.stats package
    fake_stats_pkg = ModuleType("nba_api.stats")
    fake_stats_pkg.static = fake_static_pkg  # type: ignore[attr-defined]
    fake_stats_pkg.endpoints = fake_endpoints_pkg  # type: ignore[attr-defined]

    # Root nba_api package
    fake_nba_api = ModuleType("nba_api")
    fake_nba_api.stats = fake_stats_pkg  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "nba_api", fake_nba_api)
    monkeypatch.setitem(sys.modules, "nba_api.stats", fake_stats_pkg)
    monkeypatch.setitem(sys.modules, "nba_api.stats.static", fake_static_pkg)
    monkeypatch.setitem(sys.modules, "nba_api.stats.static.players", fake_static)
    monkeypatch.setitem(sys.modules, "nba_api.stats.endpoints", fake_endpoints_pkg)
    monkeypatch.setitem(
        sys.modules,
        "nba_api.stats.endpoints.playergamelog",
        fake_endpoints_playergamelog,
    )


def _install_fake_nba_api_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake nba_api that always fails to find players."""

    class _Players:
        @staticmethod
        def find_players_by_full_name(name: str) -> list[dict[str, Any]]:
            return []

    class _PlayerGameLog:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def get_data_frames(self) -> list[pd.DataFrame]:
            return []

    fake_static = ModuleType("nba_api.stats.static.players")
    fake_static.find_players_by_full_name = _Players.find_players_by_full_name  # type: ignore[attr-defined]

    fake_static_pkg = ModuleType("nba_api.stats.static")
    fake_static_pkg.players = fake_static  # type: ignore[attr-defined]

    fake_endpoints_playergamelog = ModuleType("nba_api.stats.endpoints.playergamelog")
    fake_endpoints_playergamelog.PlayerGameLog = _PlayerGameLog  # type: ignore[attr-defined]

    fake_endpoints_pkg = ModuleType("nba_api.stats.endpoints")
    fake_endpoints_pkg.playergamelog = fake_endpoints_playergamelog  # type: ignore[attr-defined]

    fake_stats_pkg = ModuleType("nba_api.stats")
    fake_stats_pkg.static = fake_static_pkg  # type: ignore[attr-defined]
    fake_stats_pkg.endpoints = fake_endpoints_pkg  # type: ignore[attr-defined]

    fake_nba_api = ModuleType("nba_api")
    fake_nba_api.stats = fake_stats_pkg  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "nba_api", fake_nba_api)
    monkeypatch.setitem(sys.modules, "nba_api.stats", fake_stats_pkg)
    monkeypatch.setitem(sys.modules, "nba_api.stats.static", fake_static_pkg)
    monkeypatch.setitem(sys.modules, "nba_api.stats.static.players", fake_static)
    monkeypatch.setitem(sys.modules, "nba_api.stats.endpoints", fake_endpoints_pkg)
    monkeypatch.setitem(
        sys.modules,
        "nba_api.stats.endpoints.playergamelog",
        fake_endpoints_playergamelog,
    )


def test_get_stats_success_trend_and_averages(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_nba_api_success(monkeypatch)

    players = ["Stephen Curry"]
    results = get_stats(players)

    assert len(results) == 1
    stats = results[0]
    assert stats.player == "Stephen Curry"
    assert stats.data_source == DataSource.NBA_API
    # Season average: mean of [30, 28, 25, 22, 20, 18] ~= 23.83
    assert stats.season_avg_points is not None
    assert stats.last5_avg_points is not None
    # With last5 being [30, 28, 25, 22, 20] (avg 25), delta < 2 => STABLE.
    assert stats.trend == Trend.STABLE


def test_get_stats_fallback_on_player_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_nba_api_failure(monkeypatch)

    players = ["Some Unknown Player"]
    results = get_stats(players)

    assert len(results) == 1
    stats = results[0]
    assert stats.player == "Some Unknown Player"
    assert stats.data_source == DataSource.UNAVAILABLE
    assert stats.season_avg_points is None
    assert stats.last5_avg_points is None
    assert stats.trend == Trend.STABLE
