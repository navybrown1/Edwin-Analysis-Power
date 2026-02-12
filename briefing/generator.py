from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from insights.decision_cards import DecisionCard


@dataclass(frozen=True)
class BoardBrief:
    generated_at: str
    summary: str
    kpis: list[tuple[str, str, str]]
    decisions: list[DecisionCard]
    risks: list[str]
    assumptions: list[str]
    actions: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)


def build_board_brief(
    *,
    total_rows: int,
    filtered_rows: int,
    columns: int,
    kpis: list[tuple[str, str, str]],
    decision_cards: list[DecisionCard],
    insights: list[str],
    filters: list[str],
    source_info: dict[str, Any],
) -> BoardBrief:
    summary = (
        f"Filtered scope includes {filtered_rows:,} of {total_rows:,} rows across {columns} columns. "
        "Primary opportunities and risks are evidence-backed and action-oriented for leadership review."
    )

    risks = [ins for ins in insights if any(tag in ins.lower() for tag in ["missing", "outlier", "risk", "warning"])][:3]
    if not risks:
        risks = insights[:2] if insights else ["No explicit risk signal detected in current filters."]

    assumptions = [
        "All conclusions are based on the currently filtered dataset snapshot.",
        "AI recommendations assume metric definitions are semantically correct for uploaded columns.",
    ]
    limitation = str(source_info.get("limitations", "")).strip()
    if limitation:
        assumptions.append(f"Source limitation noted: {limitation}")

    actions = [
        "Day 1-2: Validate top opportunity segment with domain owner and lock target metric.",
        "Day 3-4: Launch focused intervention on top risk segment and monitor KPI drift daily.",
        "Day 5-7: Review outcomes, document wins, and adjust filters/thresholds for next cycle.",
    ]

    provenance = {
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
        "columns": columns,
        "filters": filters,
        "source": source_info,
    }

    return BoardBrief(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        summary=summary,
        kpis=kpis,
        decisions=decision_cards[:3],
        risks=risks,
        assumptions=assumptions[:3],
        actions=actions,
        provenance=provenance,
    )


def board_brief_to_markdown(brief: BoardBrief) -> str:
    kpi_lines = "\n".join(f"- **{label}**: {value} — {help_text}" for label, value, help_text in brief.kpis)
    decision_lines = "\n".join(
        (
            f"### {card.title}\n"
            f"- Metric delta: {card.metric_delta}\n"
            f"- Expected impact: {card.impact_estimate}\n"
            f"- Rationale: {card.rationale}\n"
            f"- Evidence: {card.evidence}"
        )
        for card in brief.decisions
    )
    risk_lines = "\n".join(f"- {item}" for item in brief.risks)
    assumption_lines = "\n".join(f"- {item}" for item in brief.assumptions)
    action_lines = "\n".join(f"- {item}" for item in brief.actions)

    filters = brief.provenance.get("filters", [])
    filters_text = ", ".join(filters) if filters else "No active filters"

    return (
        "# Board Brief\n\n"
        f"Generated at: {brief.generated_at}\n\n"
        "## Executive Summary\n"
        f"{brief.summary}\n\n"
        "## KPI Snapshot\n"
        f"{kpi_lines or '- No KPI data available'}\n\n"
        "## Decisions With Evidence\n"
        f"{decision_lines or 'No decision cards available.'}\n\n"
        "## Risks and Assumptions\n"
        "### Risks\n"
        f"{risk_lines}\n\n"
        "### Assumptions\n"
        f"{assumption_lines}\n\n"
        "## 7-Day Action Plan\n"
        f"{action_lines}\n\n"
        "## Provenance\n"
        f"- Total rows: {brief.provenance.get('total_rows', 0):,}\n"
        f"- Filtered rows: {brief.provenance.get('filtered_rows', 0):,}\n"
        f"- Columns: {brief.provenance.get('columns', 0)}\n"
        f"- Active filters: {filters_text}\n"
        f"- Source system: {brief.provenance.get('source', {}).get('source_system', '')}\n"
    )


def board_brief_to_html(brief: BoardBrief) -> str:
    md = board_brief_to_markdown(brief)
    body = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'>"
        "<title>Board Brief</title>"
        "<style>body{font-family:Inter,Arial,sans-serif;margin:28px;line-height:1.5;color:#0f172a;}"
        "h1,h2,h3{color:#1d4ed8;}"
        "</style></head><body>"
        f"{body}"
        "</body></html>"
    )
