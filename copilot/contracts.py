from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

ALLOWED_OPERATIONS = {"count", "sum", "mean", "median", "min", "max", "nunique", "ratio", "pct"}
ALLOWED_BASE_OPERATIONS = {"count", "sum", "mean", "median", "min", "max", "nunique"}
ALLOWED_FILTER_OPS = {"eq", "neq", "gt", "gte", "lt", "lte", "in", "contains"}
ALLOWED_INTENTS = {"summary", "comparison", "trend", "anomaly", "distribution", "breakdown"}
ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "table", "none"}
_COLUMN_RE = re.compile(r"^[\w\s\-\(\)%./:]+$")


class ValidationError(ValueError):
    """Raised when AI output fails strict schema checks."""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    operation: str
    column: str | None = None
    numerator: dict[str, Any] | None = None
    denominator: dict[str, Any] | None = None
    where: list[dict[str, Any]] = field(default_factory=list)
    format: str = "auto"
    help_text: str = ""


@dataclass(frozen=True)
class CopilotQueryPlan:
    question: str
    intent: str
    metrics: list[MetricSpec]
    dimensions: list[str] = field(default_factory=list)
    filters: list[dict[str, Any]] = field(default_factory=list)
    time_grain: str | None = None
    chart_type: str | None = None
    explanation_focus: str | None = None
    assumptions: list[str] = field(default_factory=list)
    raw_plan: dict[str, Any] = field(default_factory=dict)


@dataclass
class CopilotAnswer:
    headline: str
    bullets: list[str]
    confidence: str
    evidence_df: Any
    chart_spec: dict[str, Any]
    assumptions: list[str] = field(default_factory=list)


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValidationError("No JSON object found in model response.")

    for idx in range(end, start, -1):
        candidate = cleaned[start : idx + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValidationError("Could not decode model JSON response.")


def _validate_column(column: Any, columns: list[str] | None) -> str:
    if not isinstance(column, str) or not column.strip():
        raise ValidationError("Column must be a non-empty string.")
    col = column.strip()
    if not _COLUMN_RE.match(col):
        raise ValidationError(f"Unsafe column name: {col}")
    if columns is not None and col not in columns:
        raise ValidationError(f"Unknown column: {col}")
    return col


def _normalize_where(raw_where: Any, columns: list[str] | None) -> list[dict[str, Any]]:
    if raw_where is None:
        return []
    if not isinstance(raw_where, list):
        raise ValidationError("where must be a list.")

    normalized: list[dict[str, Any]] = []
    for item in raw_where:
        if not isinstance(item, dict):
            raise ValidationError("where items must be objects.")
        col = _validate_column(item.get("column"), columns)
        op = str(item.get("op", "eq")).strip().lower()
        if op not in ALLOWED_FILTER_OPS:
            raise ValidationError(f"Unsupported filter op: {op}")
        normalized.append({"column": col, "op": op, "value": item.get("value")})
    return normalized


def _parse_operand(raw: Any, columns: list[str] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValidationError("numerator/denominator must be objects.")

    op = str(raw.get("operation", "count")).strip().lower()
    if op not in ALLOWED_BASE_OPERATIONS:
        raise ValidationError("numerator/denominator only support base operations.")

    column = raw.get("column")
    parsed_col = _validate_column(column, columns) if column is not None else None
    where = _normalize_where(raw.get("where", []), columns)

    return {"operation": op, "column": parsed_col, "where": where}


def parse_metric_spec(raw: dict[str, Any], columns: list[str] | None = None) -> MetricSpec:
    if not isinstance(raw, dict):
        raise ValidationError("Metric spec must be an object.")

    name = str(raw.get("name", "Metric")).strip()[:64]
    if not name:
        raise ValidationError("Metric name is required.")

    operation = str(raw.get("operation", "count")).strip().lower()
    if operation not in ALLOWED_OPERATIONS:
        raise ValidationError(f"Unsupported metric operation: {operation}")

    column = raw.get("column")
    parsed_column = _validate_column(column, columns) if column is not None else None
    where = _normalize_where(raw.get("where", []), columns)

    numerator = None
    denominator = None
    if operation in {"ratio", "pct"}:
        numerator = _parse_operand(raw.get("numerator"), columns)
        denominator = _parse_operand(raw.get("denominator"), columns)

    format_hint = str(raw.get("format", "auto")).strip().lower()
    if format_hint not in {"auto", "number", "currency", "percent", "days", "integer"}:
        format_hint = "auto"

    help_text = str(raw.get("help_text", "")).strip()[:160]

    return MetricSpec(
        name=name,
        operation=operation,
        column=parsed_column,
        numerator=numerator,
        denominator=denominator,
        where=where,
        format=format_hint,
        help_text=help_text,
    )


def parse_query_plan(payload: dict[str, Any], columns: list[str] | None = None) -> CopilotQueryPlan:
    if not isinstance(payload, dict):
        raise ValidationError("Plan payload must be an object.")

    question = str(payload.get("question", "")).strip()[:500]
    intent = str(payload.get("intent", "summary")).strip().lower()
    if intent not in ALLOWED_INTENTS:
        intent = "summary"

    raw_metrics = payload.get("metrics", [])
    if not isinstance(raw_metrics, list):
        raise ValidationError("metrics must be a list.")

    metrics: list[MetricSpec] = []
    for raw in raw_metrics[:6]:
        try:
            metrics.append(parse_metric_spec(raw, columns))
        except ValidationError:
            continue

    if not metrics:
        metrics.append(MetricSpec(name="Record Count", operation="count", column=None))

    dimensions: list[str] = []
    raw_dimensions = payload.get("dimensions", [])
    if isinstance(raw_dimensions, list):
        for dim in raw_dimensions[:2]:
            try:
                dimensions.append(_validate_column(dim, columns))
            except ValidationError:
                continue

    filters = _normalize_where(payload.get("filters", []), columns)

    time_grain = payload.get("time_grain")
    if isinstance(time_grain, str):
        time_grain = time_grain.strip().upper()[:4]
    else:
        time_grain = None

    chart_type = str(payload.get("chart_type", "table")).strip().lower()
    if chart_type not in ALLOWED_CHART_TYPES:
        chart_type = "table"

    explanation_focus = payload.get("explanation_focus")
    if isinstance(explanation_focus, str):
        explanation_focus = explanation_focus.strip()[:120]
    else:
        explanation_focus = None

    assumptions: list[str] = []
    raw_assumptions = payload.get("assumptions", [])
    if isinstance(raw_assumptions, list):
        assumptions = [str(item).strip()[:120] for item in raw_assumptions[:3] if str(item).strip()]

    return CopilotQueryPlan(
        question=question,
        intent=intent,
        metrics=metrics,
        dimensions=dimensions,
        filters=filters,
        time_grain=time_grain,
        chart_type=chart_type,
        explanation_focus=explanation_focus,
        assumptions=assumptions,
        raw_plan=payload,
    )


def parse_query_plan_text(text: str, columns: list[str] | None = None) -> CopilotQueryPlan:
    payload = _extract_json_object(text)
    return parse_query_plan(payload, columns)


def parse_kpi_specs_text(text: str, columns: list[str], max_items: int = 5) -> list[MetricSpec]:
    payload = _extract_json_object(text)
    raw_items: Any

    if isinstance(payload.get("metrics"), list):
        raw_items = payload.get("metrics")
    elif isinstance(payload.get("kpis"), list):
        raw_items = payload.get("kpis")
    else:
        raw_items = []

    specs: list[MetricSpec] = []
    for raw in raw_items[:max_items]:
        try:
            specs.append(parse_metric_spec(raw, columns))
        except ValidationError:
            continue
    return specs
