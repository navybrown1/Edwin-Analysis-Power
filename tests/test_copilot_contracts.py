import pytest

from copilot.contracts import ValidationError, parse_metric_spec, parse_query_plan_text


def test_parse_query_plan_text_valid():
    text = '''
    {
      "question": "What changed?",
      "intent": "comparison",
      "metrics": [{"name": "Total", "operation": "count"}],
      "dimensions": ["segment"],
      "filters": [{"column": "segment", "op": "eq", "value": "A"}],
      "time_grain": "M",
      "chart_type": "bar",
      "explanation_focus": "executive",
      "assumptions": ["sample"]
    }
    '''
    plan = parse_query_plan_text(text, columns=["segment", "value"])
    assert plan.intent == "comparison"
    assert len(plan.metrics) == 1
    assert plan.dimensions == ["segment"]


def test_parse_query_plan_text_invalid_json():
    with pytest.raises(ValidationError):
        parse_query_plan_text("not-json", columns=["a"])


def test_metric_spec_rejects_unsafe_operation():
    with pytest.raises(ValidationError):
        parse_metric_spec({"name": "Bad", "operation": "__import__('os').system('id')"}, columns=["a"])
