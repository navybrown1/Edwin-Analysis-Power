from briefing.generator import BoardBrief, board_brief_to_html, board_brief_to_markdown, build_board_brief
from insights.decision_cards import DecisionCard


def test_board_brief_output_contains_required_sections():
    brief = build_board_brief(
        total_rows=1000,
        filtered_rows=250,
        columns=12,
        kpis=[("Revenue", "$1,200", "Monthly revenue")],
        decision_cards=[
            DecisionCard(
                title="Biggest Opportunity",
                metric_delta="Revenue +8%",
                impact_estimate="Lift expected",
                rationale="Strong segment",
                evidence="Segment A leads",
            ),
            DecisionCard(
                title="Biggest Risk",
                metric_delta="Churn +4%",
                impact_estimate="Downside risk",
                rationale="Weak retention",
                evidence="Cohort C declines",
            ),
            DecisionCard(
                title="Recommended Next Action",
                metric_delta="Prioritize Segment A",
                impact_estimate="Faster gains",
                rationale="Largest share",
                evidence="42% volume",
            ),
        ],
        insights=["Most missing field is region."],
        filters=["Region=East"],
        source_info={"source_system": "CSV"},
    )

    md = board_brief_to_markdown(brief)
    html = board_brief_to_html(brief)

    assert "## Executive Summary" in md
    assert "## KPI Snapshot" in md
    assert "## Decisions With Evidence" in md
    assert "## Risks and Assumptions" in md
    assert "## 7-Day Action Plan" in md
    assert "Board Brief" in html
