"""
Analyst Agent — statistical performance retrieval (nba_api).

Step 4.1: Parallelize per-player stats retrieval with ThreadPoolExecutor.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import List

from graph.schemas import DataSource, StatsResult, Trend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PointsSummary:
    season_avg: float
    last5_avg: float
    trend: Trend


def _current_nba_season_string(today: date | None = None) -> str:
    """
    Return NBA season string in nba_api format, e.g. \"2025-26\".

    NBA season typically starts in Oct.
    """
    d = today or date.today()
    if d.month >= 10:
        start_year = d.year
    else:
        start_year = d.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _classify_trend(season_avg: float, last5_avg: float) -> Trend:
    """
    Classify points trend as UP/DOWN/STABLE.

    Simple heuristic: +/- 2 points vs season average.
    """
    delta = last5_avg - season_avg
    if delta > 2.0:
        return Trend.UP
    if delta < -2.0:
        return Trend.DOWN
    return Trend.STABLE


def _summarize_points(points: list[float]) -> _PointsSummary:
    if not points:
        # With no games available, treat as stable at 0.0.
        return _PointsSummary(season_avg=0.0, last5_avg=0.0, trend=Trend.STABLE)

    season_avg = float(sum(points) / len(points))
    last5_slice = points[:5] if len(points) >= 5 else points
    last5_avg = float(sum(last5_slice) / len(last5_slice))
    return _PointsSummary(
        season_avg=season_avg,
        last5_avg=last5_avg,
        trend=_classify_trend(season_avg, last5_avg),
    )


def fetch_single_player_stats(player: str) -> StatsResult:
    """
    Fetch stats for a single player using nba_api.

    - Resolve player ID by full name
    - Retrieve player game log for current season
    - Compute season average points, last-5 average points, and trend
    """
    try:
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.static import players as nba_players

        matches = nba_players.find_players_by_full_name(player)
        if not matches:
            raise ValueError(f"Player not found: {player}")

        # Prefer the closest match: first result is generally best for full name.
        player_id = int(matches[0]["id"])

        season = _current_nba_season_string()
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        frames = gamelog.get_data_frames()
        if not frames:
            raise RuntimeError("nba_api returned no data frames")

        df = frames[0]
        if "PTS" not in df.columns:
            raise RuntimeError("nba_api game log missing PTS column")

        if df.empty:
            raise RuntimeError("nba_api game log is empty for season")

        # nba_api returns most recent games first for PlayerGameLog.
        pts_series = df["PTS"].tolist()
        points = [float(x) for x in pts_series if x is not None]
        if not points:
            raise RuntimeError("nba_api game log contains no valid points")

        summary = _summarize_points(points)

        return StatsResult(
            player=player,
            season_avg_points=summary.season_avg,
            last5_avg_points=summary.last5_avg,
            trend=summary.trend,
            data_source=DataSource.NBA_API,
        )
    except Exception as exc:
        # Step 4.2 fallback: never raise; return schema-valid unavailable result.
        logger.warning("nba_api stats unavailable for %r: %s", player, exc)
        return StatsResult(
            player=player,
            season_avg_points=None,
            last5_avg_points=None,
            trend=Trend.STABLE,
            data_source=DataSource.UNAVAILABLE,
        )


def get_stats(players: List[str]) -> List[StatsResult]:
    """
    Retrieve stats for players in parallel using ThreadPoolExecutor.

    Step 4.1 requirement:
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_single_player_stats, players)
    """
    if not players:
        return []

    logger.info("Analyst get_stats: fetching %d player(s)", len(players))
    with ThreadPoolExecutor() as executor:
        results_iter = executor.map(fetch_single_player_stats, players)
        return list(results_iter)
