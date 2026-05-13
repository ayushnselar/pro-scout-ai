"""
Central logging configuration for Pro-Scout AI (STEP 7.2).

Streamlit configures the root logger; this module aligns our namespaces
(`graph`, `agents`, `frontend`) with a configurable level without
replacing Streamlit's handlers.

Set ``PRO_SCOUT_LOG_LEVEL`` to ``DEBUG``, ``INFO``, ``WARNING``, or ``ERROR``
(default: ``INFO``).
"""

from __future__ import annotations

import logging
import os

_NOISY_THIRD_PARTY = ("urllib3", "httpx", "httpcore", "hpack", "h2")


def configure_pro_scout_logging() -> None:
    """Apply log levels for app-owned packages and quiet common HTTP noise."""
    raw = os.getenv("PRO_SCOUT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, raw, logging.INFO)
    for name in ("graph", "agents", "frontend"):
        logging.getLogger(name).setLevel(level)
    for name in _NOISY_THIRD_PARTY:
        logging.getLogger(name).setLevel(logging.WARNING)
