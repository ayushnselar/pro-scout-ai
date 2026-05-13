"""
Pro-Scout AI — Streamlit frontend.

Step 7.1:
- Text input box
- Submit button
- Loading indicator
- Display action, confidence, reasoning, sources
- Expanders for news and stats
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from time import monotonic
from typing import List

# Ensure project root is on sys.path so `graph` and `agents` can be imported
# when running `streamlit run frontend/app.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from graph.logging_config import configure_pro_scout_logging
from graph.schemas import DecisionEnvelope, NewsResult, StatsResult
from graph.workflow import run_workflow_with_debug

configure_pro_scout_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Pro-Scout AI", page_icon="🏀", layout="centered")

st.title("Pro-Scout AI")
st.caption("Multi-agent fantasy decision assistant (MVP)")


def _format_confidence(conf: float) -> str:
    pct = round(conf * 100)
    return f"{pct}%"


def _render_sources(sources: List[str]) -> None:
    if not sources:
        st.write("No sources available.")
        return
    for url in sources:
        st.markdown(f"- [{url}]({url})")


def _render_news(news: List[NewsResult]) -> None:
    if not news:
        st.write("No news available.")
        return
    for item in news:
        with st.container(border=True):
            st.markdown(f"**{item.player}**")
            if item.availability:
                st.markdown(f"**Availability:** {item.availability}")
            if item.minutes_note:
                st.markdown(f"**Minutes:** {item.minutes_note}")
            if item.role_note:
                st.markdown(f"**Role:** {item.role_note}")
            if item.fantasy_impact:
                st.markdown("**Fantasy impact:**")
                st.write(item.fantasy_impact)
            else:
                st.write(item.news_summary)
            if item.source_urls:
                st.markdown("**Sources:**")
                _render_sources(item.source_urls)


def _render_stats(stats: List[StatsResult]) -> None:
    if not stats:
        st.write("No stats available.")
        return
    for s in stats:
        with st.container(border=True):
            st.markdown(f"**{s.player}**")
            st.write(f"Season avg points: {s.season_avg_points if s.season_avg_points is not None else 'N/A'}")
            st.write(f"Last 5 avg points: {s.last5_avg_points if s.last5_avg_points is not None else 'N/A'}")
            st.write(f"Trend: {s.trend}")
            st.write(f"Data source: {s.data_source}")


def _run_workflow_blocking(query: str) -> tuple[DecisionEnvelope, dict]:
    # Streamlit does not natively support async; run the coroutine to completion,
    # but enforce a hard upper bound so the UI never hangs indefinitely.
    async def _runner() -> tuple[DecisionEnvelope, dict]:
        return await asyncio.wait_for(run_workflow_with_debug(query), timeout=15.0)

    t0 = monotonic()
    try:
        return asyncio.run(_runner())
    finally:
        logger.info("Workflow wall time %.2fs", monotonic() - t0)


query = st.text_input(
    "Ask a fantasy decision question",
    placeholder="Examples: Start Jayson Tatum?  Trade LeBron James for Stephen Curry?",
)

col_submit, col_spacer = st.columns([1, 3])
with col_submit:
    submitted = st.button("Run Pro-Scout AI", type="primary", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        q = query.strip()
        logger.info("Submit: running workflow (query_len=%d)", len(q))
        with st.spinner("Scouting, analyzing, and deciding..."):
            try:
                envelope, scout_debug = _run_workflow_blocking(q)
            except asyncio.TimeoutError:
                logger.warning("Workflow timed out after UI wait (15s cap)")
                st.error(
                    "The analysis took too long and timed out. Please try again in a moment, or narrow your question."
                )
                st.stop()
            except Exception:
                logger.exception("Workflow failed with an unexpected error")
                raise

        decision = envelope.decision
        logger.info(
            "Workflow result: action=%s confidence=%.2f players=%d",
            decision.action,
            decision.confidence,
            len(envelope.players),
        )

        st.subheader("Decision")
        st.markdown(f"**Action:** {decision.action}")
        st.markdown(f"**Confidence:** {_format_confidence(decision.confidence)}")
        st.markdown("**Reasoning:**")
        st.write(decision.reasoning)

        st.markdown("**Sources:**")
        _render_sources(decision.sources)

        st.caption(f"Data freshness: ~{envelope.data_freshness_minutes} minutes")

        with st.expander("News signals"):
            _render_news(envelope.news)

        with st.expander("Statistical signals"):
            _render_stats(envelope.stats)

        with st.expander("Debug (raw agent outputs)"):
            gm_fallback_triggered = any(
                s in decision.reasoning.lower() for s in ("system error", "invalid model output", "llm unavailable")
            )
            st.markdown(f"**Extracted players:** `{envelope.players}`")
            st.markdown(f"**GM fallback triggered:** `{gm_fallback_triggered}`")
            st.markdown("**News (per player):**")
            for n in envelope.news:
                st.write(
                    {
                        "player": n.player,
                        "summary": n.news_summary,
                        "url_count": len(n.source_urls),
                    }
                )
            st.markdown("**Stats (per player):**")
            for s in envelope.stats:
                st.write(
                    {
                        "player": s.player,
                        "season_avg_points": s.season_avg_points,
                        "last5_avg_points": s.last5_avg_points,
                        "trend": str(s.trend),
                        "data_source": str(s.data_source),
                    }
                )

        with st.expander("Debug (Scout retrieval stats)"):
            st.json(scout_debug)
