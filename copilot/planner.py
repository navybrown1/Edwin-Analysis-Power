from __future__ import annotations

from typing import Any

import pandas as pd

from copilot.contracts import MetricSpec, ValidationError, parse_kpi_specs_text, parse_query_plan_text


def _schema_preview(df: pd.DataFrame, max_columns: int = 40) -> str:
    lines: list[str] = []
    for col in df.columns[:max_columns]:
        dtype = str(df[col].dtype)
        nunique = int(df[col].nunique(dropna=True))
        lines.append(f"- {col} ({dtype}, {nunique} unique)")
    return "\n".join(lines)


def build_query_plan_prompt(
    question: str,
    context: str,
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
) -> str:
    return (
        "You are an analytics planner. Return ONLY valid JSON.\n"
        "Create a deterministic analysis plan for the user question using this exact schema:\n"
        "{\n"
        '  "question": "string",\n'
        '  "intent": "summary|comparison|trend|anomaly|distribution|breakdown",\n'
        '  "metrics": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "operation": "count|sum|mean|median|min|max|nunique|ratio|pct",\n'
        '      "column": "optional column name",\n'
        '      "numerator": {"operation": "count|sum|mean|median|min|max|nunique", "column": "optional", "where": []},\n'
        '      "denominator": {"operation": "count|sum|mean|median|min|max|nunique", "column": "optional", "where": []},\n'
        '      "where": [{"column": "name", "op": "eq|neq|gt|gte|lt|lte|in|contains", "value": "any"}],\n'
        '      "format": "auto|number|currency|percent|days|integer",\n'
        '      "help_text": "short explanation"\n'
        "    }\n"
        "  ],\n"
        '  "dimensions": ["column names"],\n'
        '  "filters": [{"column": "name", "op": "eq|neq|gt|gte|lt|lte|in|contains", "value": "any"}],\n'
        '  "time_grain": "D|W|M|Q or null",\n'
        '  "chart_type": "bar|line|scatter|table|none",\n'
        '  "explanation_focus": "string",\n'
        '  "assumptions": ["string"]\n'
        "}\n\n"
        "Rules:\n"
        "- Use only columns that exist in the provided schema.\n"
        "- Favor robust business metrics (median/ratios) over fragile ones.\n"
        "- Keep metrics <= 4 and dimensions <= 2.\n"
        "- No markdown, no prose outside JSON.\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"DATA CONTEXT:\n{context}\n\n"
        "SCHEMA PREVIEW:\n"
        f"{_schema_preview(df)}\n\n"
        f"NUMERIC COLUMNS: {', '.join(numeric_cols[:20]) if numeric_cols else 'none'}\n"
        f"CATEGORICAL COLUMNS: {', '.join(categorical_cols[:20]) if categorical_cols else 'none'}\n"
        f"DATETIME COLUMNS: {', '.join(datetime_cols[:20]) if datetime_cols else 'none'}\n"
    )


def plan_query_with_gemini(
    model: Any,
    *,
    question: str,
    context: str,
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
):
    prompt = build_query_plan_prompt(
        question=question,
        context=context,
        df=df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
    )

    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 900, "temperature": 0.1},
    )
    raw_text = response.text if hasattr(response, "text") else str(response)
    plan = parse_query_plan_text(raw_text, columns=df.columns.tolist())
    return plan, raw_text


def build_kpi_prompt(df: pd.DataFrame, max_cols: int = 30, max_items: int = 5) -> str:
    column_lines: list[str] = []
    for col in df.columns[:max_cols]:
        dtype = str(df[col].dtype)
        nunique = int(df[col].nunique(dropna=True))
        sample = df[col].dropna().head(3).tolist()
        column_lines.append(f"- {col} ({dtype}, {nunique} unique): {sample}")

    return (
        "Return ONLY valid JSON for executive KPI cards.\n"
        "Schema:\n"
        "{\n"
        '  "kpis": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "operation": "count|sum|mean|median|min|max|nunique|ratio|pct",\n'
        '      "column": "optional",\n'
        '      "numerator": {"operation": "count|sum|mean|median|min|max|nunique", "column": "optional", "where": []},\n'
        '      "denominator": {"operation": "count|sum|mean|median|min|max|nunique", "column": "optional", "where": []},\n'
        '      "where": [{"column": "name", "op": "eq|neq|gt|gte|lt|lte|in|contains", "value": "any"}],\n'
        '      "format": "auto|number|currency|percent|days|integer",\n'
        '      "help_text": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Generate exactly {max_items} KPIs suitable for executives.\n"
        "No Python code. No eval expressions. JSON only.\n"
        "Prefer robust KPIs and avoid impossible values.\n\n"
        f"Dataset size: {len(df):,} rows x {df.shape[1]} columns\n"
        "Columns:\n"
        + "\n".join(column_lines)
    )


def generate_kpi_specs_with_gemini(model: Any, df: pd.DataFrame, max_items: int = 5) -> tuple[list[MetricSpec], str]:
    prompt = build_kpi_prompt(df=df, max_items=max_items)
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 900, "temperature": 0.15},
    )
    raw_text = response.text if hasattr(response, "text") else str(response)
    specs = parse_kpi_specs_text(raw_text, columns=df.columns.tolist(), max_items=max_items)
    return specs, raw_text
