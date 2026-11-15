import json

import httpx2 as httpx
import numpy as np
import pandas as pd
import pytest

from tradingagents import (Backtester, ClaudeLLM, CSVProvider, LLMRefusal, MockLLM, ReflectionMemory,
                           SyntheticProvider, TradingAgentsGraph, TradingConfig, metrics)
from tradingagents.backtest import action_to_target, simulate
from tradingagents.indicators import rsi, technical_snapshot
from tradingagents.memory import Lesson
from tradingagents.prompts import FINAL_SCHEMA
from tradingagents.report import render_markdown


@pytest.fixture
def data():
    return SyntheticProvider(start="2023-01-02", end="2024-12-31")


def test_point_in_time_prices_and_news(data):
    as_of = pd.Timestamp("2024-03-15")
    assert data.get_prices("AAA", as_of, 400).index.max() <= as_of
    news = data.get_news("AAA", as_of, 30)
    assert news and all(n.date <= as_of for n in news)


def test_indicators(data):
    df = data.get_prices("AAA", "2024-06-28", 365)
    r = rsi(df["Close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    snap = technical_snapshot(df)
    assert -1 <= snap["composite_score"] <= 1 and snap["date"] == "2024-06-28"


def test_metrics_known_values():
    m = metrics(pd.Series([0.1, -0.5, 0.2]))
    assert m["CR"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert m["MDD"] == pytest.approx(-0.5)


def test_simulate_has_no_lookahead():
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series([100, 110, 99, 99], index=idx, dtype=float)
    # Go long at close of day 2: earns only day2->day3 (-10%), not day1->day2 (+10%).
    r = simulate(close, pd.Series({idx[1]: 1.0}), cost_bps=0)
    assert r.iloc[0] == 0 and r.iloc[1] == pytest.approx(-0.1)


def test_action_mapping():
    assert action_to_target("BUY", 0.5, 0, False) == 0.5
    assert action_to_target("SELL", 0.5, 1, False) == 0.0
    assert action_to_target("SELL", 0.5, 1, True) == -0.5
    assert action_to_target("HOLD", 0.5, 0.3, False) == 0.3


def test_full_pipeline_with_mock(data):
    cfg = TradingConfig(max_debate_rounds=2, max_risk_rounds=2)
    g = TradingAgentsGraph(MockLLM(), data, cfg)
    s = g.propagate("AAA", "2024-05-01")
    assert set(s.analyst_reports) == {"fundamentals", "sentiment", "news", "technical"}
    assert len(s.investment_debate) == 4 and len(s.risk_debate) == 6
    assert s.action in {"BUY", "SELL", "HOLD"}
    assert set(FINAL_SCHEMA["required"]) <= set(s.final_decision)
    assert "Final action" in render_markdown(s)


def test_memory_recall_excludes_future():
    m = ReflectionMemory()
    m.add(Lesson("AAA", "2024-01-05", "BUY", 0.02, "past"))
    m.add(Lesson("AAA", "2024-02-05", "BUY", 0.02, "future"))
    got = m.recall("AAA", "2024-02-01", 3)
    assert len(got) == 1 and "past" in got[0]


def test_backtest_with_mock(data):
    g = TradingAgentsGraph(MockLLM(), data, TradingConfig(analysts=("technical",)))
    res = Backtester(g, data, decision_interval=10).run("AAA", "2024-01-02", "2024-06-28")
    assert len(res.decisions) > 5 and set(res.baselines) == {"buy_and_hold", "sma_cross", "macd"}
    assert len(g.memory.items) == len(res.decisions) - 1  # reflection after each realised period
    assert list(res.summary().columns) == ["CR", "ARR", "SR", "MDD"]


def test_csv_provider(tmp_path, data):
    df = data.history("AAA").reset_index(names="Date")
    df.to_csv(tmp_path / "AAA.csv", index=False)
    pd.DataFrame({"date": ["2024-01-03", "2099-01-01"], "title": ["old", "future"]}).to_csv(tmp_path / "AAA_news.csv", index=False)
    p = CSVProvider(tmp_path)
    assert len(p.get_prices("AAA", "2024-01-31", 30)) > 10
    assert [n.title for n in p.get_news("AAA", "2024-01-31", 60)] == ["old"]


def _fake_claude(responses):
    import anthropic

    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    client = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return client, seen


def _msg(text, stop="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_claude_request_shape_and_structured_output():
    decision = {"action": "BUY", "position_size": 0.5, "confidence": 0.7, "rationale": "r", "risk_controls": "c"}
    client, seen = _fake_claude([_msg(json.dumps(decision))])
    llm = ClaudeLLM(client=client)
    assert llm.structured("sys", "prompt", FINAL_SCHEMA) == decision
    body = seen[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default"
    assert llm.usage.calls == 1 and llm.usage.input_tokens == 10


def test_claude_refusal_raises():
    client, _ = _fake_claude([_msg("", stop="refusal")])
    with pytest.raises(LLMRefusal):
        ClaudeLLM(client=client).text("sys", "prompt")
