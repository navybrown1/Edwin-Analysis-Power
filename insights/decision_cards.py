from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DecisionCard:
    title: str
    metric_delta: str
    impact_estimate: str
    rationale: str
    evidence: str


def _fmt_number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def _numeric_shift_table(df_full: pd.DataFrame, df_filtered: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in numeric_cols:
        full = pd.to_numeric(df_full[col], errors="coerce").dropna()
        filt = pd.to_numeric(df_filtered[col], errors="coerce").dropna()
        if full.empty or filt.empty:
            continue
        full_mean = float(full.mean())
        filt_mean = float(filt.mean())
        delta = filt_mean - full_mean
        delta_pct = (delta / full_mean * 100.0) if full_mean != 0 else 0.0
        rows.append(
            {
                "metric": col,
                "full_mean": full_mean,
                "filtered_mean": filt_mean,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )
    return pd.DataFrame(rows)


def generate_decision_cards(
    df_full: pd.DataFrame,
    df_filtered: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[DecisionCard]:
    cards: list[DecisionCard] = []

    compare = _numeric_shift_table(df_full, df_filtered, numeric_cols)

    if not compare.empty:
        opp = compare.sort_values("delta_pct", ascending=False).iloc[0]
        cards.append(
            DecisionCard(
                title="Biggest Opportunity",
                metric_delta=f"{opp['metric']}: {_fmt_number(float(opp['delta']))} ({float(opp['delta_pct']):.1f}%)",
                impact_estimate=(
                    f"If sustained, expected directional lift is about {abs(float(opp['delta_pct'])):.1f}% "
                    f"versus full-dataset baseline."
                ),
                rationale=(
                    f"Filtered segment outperforms baseline for {opp['metric']} and is the strongest positive shift."
                ),
                evidence=(
                    f"Baseline mean {_fmt_number(float(opp['full_mean']))}, filtered mean {_fmt_number(float(opp['filtered_mean']))}."
                ),
            )
        )

        risk = compare.sort_values("delta_pct", ascending=True).iloc[0]
        cards.append(
            DecisionCard(
                title="Biggest Risk",
                metric_delta=f"{risk['metric']}: {_fmt_number(float(risk['delta']))} ({float(risk['delta_pct']):.1f}%)",
                impact_estimate=(
                    f"Current downside exposure is approximately {abs(float(risk['delta_pct'])):.1f}% "
                    f"from baseline behavior."
                ),
                rationale=(
                    f"This is the largest negative movement against the full-dataset benchmark."
                ),
                evidence=(
                    f"Baseline mean {_fmt_number(float(risk['full_mean']))}, filtered mean {_fmt_number(float(risk['filtered_mean']))}."
                ),
            )
        )

    if categorical_cols and not df_filtered.empty:
        cat = categorical_cols[0]
        counts = df_filtered[cat].fillna("(Missing)").astype(str).value_counts()
        if not counts.empty:
            top_label = str(counts.index[0])
            share = counts.iloc[0] / max(len(df_filtered), 1) * 100.0
            cards.append(
                DecisionCard(
                    title="Recommended Next Action",
                    metric_delta=f"Focus on {cat}: {top_label} ({share:.1f}% share)",
                    impact_estimate="Prioritizing the dominant segment should accelerate measurable gains in the next 7 days.",
                    rationale="Concentrating intervention on the most frequent segment maximizes near-term operational impact.",
                    evidence=f"{top_label} appears {int(counts.iloc[0]):,} times in the current filtered scope.",
                )
            )

    while len(cards) < 3:
        cards.append(
            DecisionCard(
                title="Recommended Next Action" if len(cards) == 2 else "Biggest Opportunity",
                metric_delta="Insufficient stable signal",
                impact_estimate="Gather more rows or broaden filters for stronger confidence.",
                rationale="Current sample does not provide enough contrast across segments.",
                evidence=f"Filtered rows: {len(df_filtered):,}.",
            )
        )

    return cards[:3]
