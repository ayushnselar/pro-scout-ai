"""
Validation helpers for Pro-Scout AI schemas.

All LLM and external API outputs must be validated against these
helpers before being used by the system.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, List, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from graph.schemas import DecisionEnvelope, DecisionResult, NewsResult, StatsResult

logger = logging.getLogger(__name__)


TModel = TypeVar("TModel", bound=BaseModel)


def _coerce_model(model: Type[TModel], data: Any) -> TModel:
    """
    Coerce arbitrary data into a concrete Pydantic model.

    - If `data` is already an instance of `model`, it is returned as-is.
    - Otherwise, `model.model_validate` is used for strict validation.
    """
    if isinstance(data, model):
        return data
    return model.model_validate(data)


def _coerce_model_list(model: Type[TModel], items: Iterable[Any]) -> List[TModel]:
    """Validate a sequence of items into a list of models."""
    return [_coerce_model(model, item) for item in items]


def validate_news_results(items: Sequence[Any]) -> List[NewsResult]:
    """Validate a collection of raw news payloads into `NewsResult` objects."""
    return _coerce_model_list(NewsResult, items)


def validate_stats_results(items: Sequence[Any]) -> List[StatsResult]:
    """Validate a collection of raw stats payloads into `StatsResult` objects."""
    return _coerce_model_list(StatsResult, items)


def validate_decision_result(data: Any) -> DecisionResult:
    """Validate raw decision payload into a `DecisionResult`."""
    return _coerce_model(DecisionResult, data)


def validate_decision_envelope(data: Any) -> DecisionEnvelope:
    """Validate raw workflow output into a `DecisionEnvelope`."""
    try:
        return _coerce_model(DecisionEnvelope, data)
    except ValidationError as exc:
        logger.warning("DecisionEnvelope validation failed: %s", exc)
        raise


def validate_json_model(model: Type[TModel], json_str: str) -> TModel:
    """
    Parse a JSON string and validate it against the given model.

    Raises `ValueError` on JSON parsing or schema validation failure.
    """
    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed for model %s: %s", model.__name__, exc)
        raise ValueError("Invalid JSON payload") from exc

    try:
        return _coerce_model(model, payload)
    except ValidationError as exc:
        logger.warning("Schema validation failed for model %s: %s", model.__name__, exc)
        raise ValueError("Payload failed schema validation") from exc


def validate_json_model_list(model: Type[TModel], json_str: str) -> List[TModel]:
    """
    Parse a JSON array string and validate each element against the given model.

    Raises `ValueError` on JSON parsing or schema validation failure.
    """
    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON payload") from exc

    if not isinstance(payload, list):
        raise ValueError("Expected JSON array for list validation")

    try:
        return _coerce_model_list(model, payload)
    except ValidationError as exc:
        raise ValueError("One or more items failed schema validation") from exc
