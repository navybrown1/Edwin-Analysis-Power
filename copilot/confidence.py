from __future__ import annotations


def compute_confidence(
    *,
    rows_used: int,
    total_rows: int,
    metrics_computed: int,
    missingness_pct: float,
    plan_valid: bool,
    r_squared: float | None = None,
) -> tuple[str, float, str]:
    """Return confidence label, numeric score, and short rationale."""
    if total_rows <= 0:
        return "Low", 0.0, "No rows available after filtering."

    coverage = min(1.0, rows_used / max(total_rows, 1))
    metric_factor = min(1.0, metrics_computed / 3.0)
    missing_penalty = min(0.4, max(0.0, missingness_pct / 100.0) * 0.4)

    score = 0.0
    score += 0.5 * coverage
    score += 0.25 * metric_factor
    score += 0.15 * (1.0 if plan_valid else 0.4)

    if r_squared is not None:
        score += 0.1 * max(0.0, min(1.0, r_squared))
    else:
        score += 0.05

    score -= missing_penalty
    score = max(0.0, min(1.0, score))

    if score >= 0.75:
        label = "High"
    elif score >= 0.5:
        label = "Medium"
    else:
        label = "Low"

    rationale = (
        f"Coverage {coverage * 100:.1f}% of filtered rows, "
        f"{metrics_computed} validated metric(s), "
        f"missingness impact {missingness_pct:.1f}%."
    )
    return label, score, rationale
