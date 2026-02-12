import math

import pandas as pd
import pytest

from copilot.contracts import MetricSpec, ValidationError, parse_metric_spec
from copilot.executor import evaluate_metric


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
            "grp": ["a", "a", "b", "b"],
        }
    )


def test_base_operations():
    df = _df()
    assert evaluate_metric(df, MetricSpec(name="c", operation="count")) == 4
    assert evaluate_metric(df, MetricSpec(name="s", operation="sum", column="x")) == 10
    assert evaluate_metric(df, MetricSpec(name="m", operation="mean", column="x")) == 2.5
    assert evaluate_metric(df, MetricSpec(name="med", operation="median", column="x")) == 2.5
    assert evaluate_metric(df, MetricSpec(name="mn", operation="min", column="x")) == 1
    assert evaluate_metric(df, MetricSpec(name="mx", operation="max", column="x")) == 4
    assert evaluate_metric(df, MetricSpec(name="nu", operation="nunique", column="grp")) == 2


def test_ratio_and_pct_operations():
    df = _df()
    ratio_spec = MetricSpec(
        name="ratio",
        operation="ratio",
        numerator={"operation": "sum", "column": "y", "where": []},
        denominator={"operation": "sum", "column": "x", "where": []},
    )
    pct_spec = MetricSpec(
        name="pct",
        operation="pct",
        numerator={"operation": "count", "column": None, "where": [{"column": "grp", "op": "eq", "value": "a"}]},
        denominator={"operation": "count", "column": None, "where": []},
    )

    assert evaluate_metric(df, ratio_spec) == 10
    assert evaluate_metric(df, pct_spec) == 50


def test_rejects_malicious_spec():
    with pytest.raises(ValidationError):
        parse_metric_spec({"name": "x", "operation": "eval", "column": "x"}, columns=["x"])
