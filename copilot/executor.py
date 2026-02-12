from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from copilot.confidence import compute_confidence
from copilot.contracts import CopilotAnswer, CopilotQueryPlan, MetricSpec


def apply_where_filters(df: pd.DataFrame, where: list[dict[str, Any]]) -> pd.DataFrame:
    if not where:
        return df

    out = df
    for flt in where:
        col = flt.get("column")
        if col not in out.columns:
            continue
        op = flt.get("op", "eq")
        val = flt.get("value")
        series = out[col]

        if op == "eq":
            mask = series == val
        elif op == "neq":
            mask = series != val
        elif op == "gt":
            mask = pd.to_numeric(series, errors="coerce") > pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        elif op == "gte":
            mask = pd.to_numeric(series, errors="coerce") >= pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        elif op == "lt":
            mask = pd.to_numeric(series, errors="coerce") < pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        elif op == "lte":
            mask = pd.to_numeric(series, errors="coerce") <= pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        elif op == "in":
            options = val if isinstance(val, list) else [val]
            mask = series.isin(options)
        elif op == "contains":
            needle = "" if val is None else str(val)
            mask = series.fillna("").astype(str).str.contains(needle, case=False, na=False)
        else:
            mask = pd.Series(True, index=out.index)

        out = out[mask.fillna(False)]

    return out


def _run_base_operation(df: pd.DataFrame, operation: str, column: str | None) -> float:
    if operation == "count":
        if column and column in df.columns:
            return float(df[column].notna().sum())
        return float(len(df))

    if not column or column not in df.columns:
        return float("nan")

    series = df[column].dropna()
    if operation == "sum":
        return float(pd.to_numeric(series, errors="coerce").dropna().sum())
    if operation == "mean":
        return float(pd.to_numeric(series, errors="coerce").dropna().mean())
    if operation == "median":
        return float(pd.to_numeric(series, errors="coerce").dropna().median())
    if operation == "min":
        return float(pd.to_numeric(series, errors="coerce").dropna().min())
    if operation == "max":
        return float(pd.to_numeric(series, errors="coerce").dropna().max())
    if operation == "nunique":
        return float(series.nunique(dropna=True))
    return float("nan")


def _evaluate_operand(df: pd.DataFrame, operand: dict[str, Any] | None) -> float:
    if not operand:
        return float("nan")
    scoped = apply_where_filters(df, operand.get("where", []))
    return _run_base_operation(scoped, str(operand.get("operation", "count")), operand.get("column"))


def evaluate_metric(df: pd.DataFrame, spec: MetricSpec) -> float:
    scoped = apply_where_filters(df, spec.where)

    if spec.operation in {"count", "sum", "mean", "median", "min", "max", "nunique"}:
        return _run_base_operation(scoped, spec.operation, spec.column)

    if spec.operation in {"ratio", "pct"}:
        num = _evaluate_operand(scoped, spec.numerator)
        den = _evaluate_operand(scoped, spec.denominator)
        if den in {0, 0.0} or np.isnan(den):
            return float("nan")
        ratio = num / den
        return float(ratio * 100.0 if spec.operation == "pct" else ratio)

    return float("nan")


def format_metric_value(value: float, fmt: str = "auto") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"

    if fmt == "integer":
        return f"{int(round(value)):,}"
    if fmt == "currency":
        return f"${value:,.2f}"
    if fmt == "percent":
        return f"{value:,.2f}%"
    if fmt == "days":
        return f"{value:,.1f} days"

    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def evaluate_kpi_specs(df: pd.DataFrame, specs: list[MetricSpec], max_items: int = 5) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for spec in specs[:max_items]:
        raw_value = evaluate_metric(df, spec)
        fmt = spec.format
        if spec.operation == "pct" and fmt == "auto":
            fmt = "percent"
        rows.append((spec.name[:24], format_metric_value(raw_value, fmt), spec.help_text))
    return rows


def build_metric_evidence(df: pd.DataFrame, specs: list[MetricSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        raw_value = evaluate_metric(df, spec)
        fmt = spec.format
        if spec.operation == "pct" and fmt == "auto":
            fmt = "percent"
        rows.append(
            {
                "metric": spec.name,
                "operation": spec.operation,
                "value": format_metric_value(raw_value, fmt),
                "raw_value": raw_value,
                "column": spec.column or "(row-level)",
            }
        )
    return pd.DataFrame(rows)


def build_dimension_evidence(df: pd.DataFrame, dimension: str, spec: MetricSpec, limit: int = 12) -> pd.DataFrame:
    if dimension not in df.columns or df.empty:
        return pd.DataFrame()

    base = df.copy()
    labels = base[dimension].fillna("(Missing)").astype(str)
    top_labels = labels.value_counts().head(limit).index.tolist()

    rows: list[dict[str, Any]] = []
    for label in top_labels:
        subset = base[labels == label]
        value = evaluate_metric(subset, spec)
        rows.append({dimension: label, "metric": spec.name, "value": value})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["value_display"] = out["value"].apply(lambda v: format_metric_value(v, "auto"))
    return out.sort_values("value", ascending=False)


def execute_query_plan(
    full_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    plan: CopilotQueryPlan,
    evidence_limit: int = 200,
) -> CopilotAnswer:
    scoped = apply_where_filters(filtered_df, plan.filters)
    metrics_table = build_metric_evidence(scoped, plan.metrics)

    headline = "AI Decision Copilot Summary"
    bullets: list[str] = []

    if not metrics_table.empty:
        lead = metrics_table.iloc[0]
        headline = f"{lead['metric']}: {lead['value']}"
        for _, row in metrics_table.head(4).iterrows():
            bullets.append(f"{row['metric']} = {row['value']} (from {row['column']}).")

    evidence_df = metrics_table.copy()
    chart_spec: dict[str, Any] = {"type": "table", "x": None, "y": None}

    if plan.dimensions and plan.metrics:
        dim = plan.dimensions[0]
        dim_evidence = build_dimension_evidence(scoped, dim, plan.metrics[0], limit=15)
        if not dim_evidence.empty:
            evidence_df = dim_evidence.head(evidence_limit)
            top_row = evidence_df.iloc[0]
            bullets.append(
                f"Top {dim}: {top_row[dim]} at {format_metric_value(float(top_row['value']), 'auto')}."
            )
            chart_spec = {"type": "bar", "x": dim, "y": "value"}

    if len(evidence_df) > evidence_limit:
        evidence_df = evidence_df.head(evidence_limit)

    missingness = float(scoped.isna().mean().mean() * 100) if not scoped.empty else 100.0
    confidence_label, _, conf_reason = compute_confidence(
        rows_used=len(scoped),
        total_rows=len(filtered_df),
        metrics_computed=len(metrics_table),
        missingness_pct=missingness,
        plan_valid=True,
    )

    bullets.append(conf_reason)
    if plan.explanation_focus:
        bullets.append(f"Focus: {plan.explanation_focus}.")

    if not bullets:
        bullets.append("Insufficient signal to compute robust metrics for this question.")

    return CopilotAnswer(
        headline=headline,
        bullets=bullets[:4],
        confidence=confidence_label,
        evidence_df=evidence_df,
        chart_spec=chart_spec,
        assumptions=plan.assumptions,
    )


def metric_spec_to_dict(spec: MetricSpec) -> dict[str, Any]:
    # Keep serializer small and explicit for auditability.
    return {
        "name": spec.name,
        "operation": spec.operation,
        "column": spec.column,
        "numerator": spec.numerator,
        "denominator": spec.denominator,
        "where": spec.where,
        "format": spec.format,
        "help_text": spec.help_text,
    }
