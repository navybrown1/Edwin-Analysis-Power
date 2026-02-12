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
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    if abs(value) >= 0.01:
        return f"{value:,.3f}"
    return "<0.01" if value != 0 else "0"


def _find_finance_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = df.columns.tolist()
    lowered = {c.lower(): c for c in cols}

    def pick(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in lowered:
                return lowered[c]
        for c in candidates:
            for col in cols:
                if c in col.lower():
                    return col
        return None

    return {
        "debit": pick(["debit", "withdrawal", "outflow", "expense"]),
        "credit": pick(["credit", "deposit", "inflow", "income"]),
        "amount": pick(["amount", "transaction amount", "value"]),
        "date": pick(["transaction_date", "date", "posted", "timestamp"]),
        "segment": pick(["description", "category", "merchant", "payee", "transaction_type", "memo"]),
    }


def _pick_segment_column(df: pd.DataFrame, categorical_cols: list[str], preferred: str | None) -> str | None:
    if preferred and preferred in df.columns:
        return preferred

    ranked: list[tuple[str, int]] = []
    for col in categorical_cols:
        nunique = int(df[col].nunique(dropna=True))
        if 2 <= nunique <= max(50, int(len(df) * 0.35)):
            ranked.append((col, nunique))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[0][0] if ranked else (categorical_cols[0] if categorical_cols else None)


def _build_finance_cards(
    df: pd.DataFrame,
    categorical_cols: list[str],
    *,
    filter_active: bool,
) -> list[DecisionCard]:
    cols = _find_finance_columns(df)

    debit_col = cols["debit"]
    credit_col = cols["credit"]
    amount_col = cols["amount"]
    seg_col = _pick_segment_column(df, categorical_cols, cols["segment"])

    work = df.copy()
    debit = pd.to_numeric(work[debit_col], errors="coerce") if debit_col else pd.Series(0, index=work.index, dtype=float)
    credit = pd.to_numeric(work[credit_col], errors="coerce") if credit_col else pd.Series(0, index=work.index, dtype=float)

    if amount_col and amount_col in work.columns:
        amount = pd.to_numeric(work[amount_col], errors="coerce")
        if amount.notna().sum() > 0:
            net = amount
        else:
            net = credit.fillna(0) - debit.fillna(0)
    else:
        net = credit.fillna(0) - debit.fillna(0)

    work["__net__"] = net
    work["__debit__"] = debit.fillna(0)

    cards: list[DecisionCard] = []

    if seg_col and seg_col in work.columns:
        seg = work[seg_col].fillna("(Missing)").astype(str)
        grouped = (
            work.assign(__segment=seg)
            .groupby("__segment", as_index=False)
            .agg(
                txn_count=("__net__", "count"),
                total_net=("__net__", "sum"),
                total_debit=("__debit__", "sum"),
            )
        )
        grouped = grouped[grouped["txn_count"] >= 2]

        if not grouped.empty:
            opp = grouped.sort_values("total_net", ascending=False).iloc[0]
            risk = grouped.sort_values("total_net", ascending=True).iloc[0]

            cards.append(
                DecisionCard(
                    title="Biggest Opportunity",
                    metric_delta=f"Segment '{opp['__segment']}': net {_fmt_number(float(opp['total_net']))}",
                    impact_estimate=(
                        "Scale behavior in this segment first; it is currently the strongest positive net contributor."
                    ),
                    rationale=(
                        "This segment has the highest net contribution in the observed period."
                    ),
                    evidence=(
                        f"{int(opp['txn_count']):,} transactions with net {_fmt_number(float(opp['total_net']))}."
                    ),
                )
            )

            cards.append(
                DecisionCard(
                    title="Biggest Risk",
                    metric_delta=f"Segment '{risk['__segment']}': net {_fmt_number(float(risk['total_net']))}",
                    impact_estimate=(
                        "Contain losses in this segment to reduce downside quickly in the next 7 days."
                    ),
                    rationale=(
                        "This segment has the weakest net performance and is the largest drag on cash flow."
                    ),
                    evidence=(
                        f"{int(risk['txn_count']):,} transactions with net {_fmt_number(float(risk['total_net']))}."
                    ),
                )
            )

            cards.append(
                DecisionCard(
                    title="Recommended Next Action",
                    metric_delta=(
                        f"Audit top 10 transactions in '{risk['__segment']}' above "
                        f"{_fmt_number(float(max(work['__debit__'].quantile(0.9), 1)))} debit"
                    ),
                    impact_estimate=(
                        "Expected outcome: reduced avoidable outflow and clearer transaction categorization."
                    ),
                    rationale=(
                        "Targeting the worst segment and highest-value debits yields the highest short-term control leverage."
                    ),
                    evidence=(
                        f"Current median debit is {_fmt_number(float(work['__debit__'][work['__debit__'] > 0].median() if (work['__debit__'] > 0).any() else 0.0))}."
                    ),
                )
            )

    if cards:
        return cards

    total_net = float(work["__net__"].sum())
    total_debit = float(work["__debit__"].sum())
    typical_debit = float(work.loc[work["__debit__"] > 0, "__debit__"].median()) if (work["__debit__"] > 0).any() else 0.0

    return [
        DecisionCard(
            title="Biggest Opportunity",
            metric_delta=f"Net flow: {_fmt_number(total_net)}",
            impact_estimate="Increase positive inflow channels and protect recurring positive balances.",
            rationale="Net flow is the clearest high-level indicator for this banking dataset.",
            evidence=f"Rows analyzed: {len(work):,}.",
        ),
        DecisionCard(
            title="Biggest Risk",
            metric_delta=f"Total debit: {_fmt_number(total_debit)}",
            impact_estimate="Track high-value debits daily to avoid avoidable leakage.",
            rationale="Debit outflow concentration usually drives risk in transaction datasets.",
            evidence=f"Typical non-zero debit: {_fmt_number(typical_debit)}.",
        ),
        DecisionCard(
            title="Recommended Next Action",
            metric_delta="Create a weekly debit watchlist",
            impact_estimate="Expected outcome: faster exception handling and fewer unexplained high-cost transactions.",
            rationale="Actionability improves when monitoring is tied to concrete thresholds and weekly review cadence.",
            evidence=f"Filter mode: {'active segment' if filter_active else 'full dataset'}.",
        ),
    ]


def _build_generic_cards(
    df_full: pd.DataFrame,
    df_filtered: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    filter_active: bool,
) -> list[DecisionCard]:
    compare_rows: list[dict[str, float | str]] = []
    for col in numeric_cols:
        full = pd.to_numeric(df_full[col], errors="coerce").dropna()
        filt = pd.to_numeric(df_filtered[col], errors="coerce").dropna()
        if full.empty or filt.empty:
            continue
        full_mean = float(full.mean())
        filt_mean = float(filt.mean())
        delta = filt_mean - full_mean
        delta_pct = (delta / full_mean * 100.0) if abs(full_mean) > 1e-9 else 0.0
        if abs(delta_pct) < 0.1 and not filter_active:
            continue
        compare_rows.append(
            {
                "metric": col,
                "full_mean": full_mean,
                "filtered_mean": filt_mean,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )

    cards: list[DecisionCard] = []

    if compare_rows:
        comp = pd.DataFrame(compare_rows)
        opp = comp.sort_values("delta_pct", ascending=False).iloc[0]
        risk = comp.sort_values("delta_pct", ascending=True).iloc[0]

        cards.append(
            DecisionCard(
                title="Biggest Opportunity",
                metric_delta=f"{opp['metric']}: {_fmt_number(float(opp['delta']))} ({float(opp['delta_pct']):.1f}%)",
                impact_estimate="Scale the strongest-performing segment or condition first.",
                rationale="This metric shows the highest positive shift versus baseline.",
                evidence=f"Baseline {_fmt_number(float(opp['full_mean']))}, current {_fmt_number(float(opp['filtered_mean']))}.",
            )
        )
        cards.append(
            DecisionCard(
                title="Biggest Risk",
                metric_delta=f"{risk['metric']}: {_fmt_number(float(risk['delta']))} ({float(risk['delta_pct']):.1f}%)",
                impact_estimate="Mitigate this weak signal first to avoid downstream performance drag.",
                rationale="This is the largest negative movement against baseline behavior.",
                evidence=f"Baseline {_fmt_number(float(risk['full_mean']))}, current {_fmt_number(float(risk['filtered_mean']))}.",
            )
        )

    if categorical_cols and not df_filtered.empty:
        cat = _pick_segment_column(df_filtered, categorical_cols, None)
        if cat:
            counts = df_filtered[cat].fillna("(Missing)").astype(str).value_counts()
            if not counts.empty:
                top_label = str(counts.index[0])
                share = counts.iloc[0] / max(len(df_filtered), 1) * 100.0
                cards.append(
                    DecisionCard(
                        title="Recommended Next Action",
                        metric_delta=f"Focus on {cat}: {top_label} ({share:.1f}% share)",
                        impact_estimate="Prioritize this dominant segment for fastest measurable movement.",
                        rationale="Action concentration on the highest-volume segment improves execution speed.",
                        evidence=f"{top_label} appears {int(counts.iloc[0]):,} times.",
                    )
                )

    while len(cards) < 3:
        cards.append(
            DecisionCard(
                title="Recommended Next Action" if len(cards) == 2 else "Biggest Opportunity",
                metric_delta="Need segmentation",
                impact_estimate="Apply one categorical or date filter to expose real contrasts.",
                rationale="Without segmentation, differences collapse and decisions become non-actionable.",
                evidence=f"Rows in current scope: {len(df_filtered):,}.",
            )
        )

    return cards[:3]


def generate_decision_cards(
    df_full: pd.DataFrame,
    df_filtered: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    dataset_context: dict | None = None,
    filter_active: bool = False,
) -> list[DecisionCard]:
    domain = (dataset_context or {}).get("domain", "generic")

    if domain == "banking":
        return _build_finance_cards(df_filtered, categorical_cols, filter_active=filter_active)

    return _build_generic_cards(
        df_full,
        df_filtered,
        numeric_cols,
        categorical_cols,
        filter_active=filter_active,
    )
