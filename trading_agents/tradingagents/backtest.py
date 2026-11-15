"""Backtester and evaluation metrics from the paper (CR, ARR, SR, MDD) plus baselines."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import DataProvider
from .indicators import macd, sma
from .state import AgentState

log = logging.getLogger(__name__)
TRADING_DAYS = 252


def metrics(returns: pd.Series) -> dict:
    """Cumulative return, annualised return, Sharpe ratio (rf=0), max drawdown."""
    r = returns.fillna(0.0)
    if r.empty:
        return {"CR": 0.0, "ARR": 0.0, "SR": 0.0, "MDD": 0.0}
    equity = (1 + r).cumprod()
    cr = float(equity.iloc[-1] - 1)
    arr = float(equity.iloc[-1] ** (TRADING_DAYS / len(r)) - 1)
    sd = r.std(ddof=1)
    sr = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd and sd > 0 else 0.0
    mdd = float((equity / equity.cummax() - 1).min())
    return {"CR": cr, "ARR": arr, "SR": sr, "MDD": mdd}


def action_to_target(action: str, size: float, current: float, allow_short: bool) -> float:
    if action == "BUY":
        return size if size > 0 else 1.0
    if action == "SELL":
        return -(size if size > 0 else 1.0) if allow_short else 0.0
    return current


def simulate(close: pd.Series, targets: pd.Series, cost_bps: float = 5.0) -> pd.Series:
    """Position set at close t earns the return from t to t+1; trading cost on turnover."""
    pos = targets.reindex(close.index).ffill().fillna(0.0)
    asset_ret = close.pct_change().shift(-1).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    return pos * asset_ret - turnover * cost_bps / 1e4


# --- rule-based baselines (as compared in the paper) ---------------------------
def baseline_targets(close: pd.Series, name: str) -> pd.Series:
    if name == "buy_and_hold":
        return pd.Series(1.0, index=close.index)
    if name == "sma_cross":
        return (sma(close, 10) > sma(close, 50)).astype(float)
    if name == "macd":
        m = macd(close)
        return (m["macd"] > m["macd_signal"]).astype(float)
    raise ValueError(name)


@dataclass
class BacktestResult:
    ticker: str
    returns: pd.Series
    positions: pd.Series
    decisions: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = {"TradingAgents": self.metrics, **self.baselines}
        return pd.DataFrame(rows).T[["CR", "ARR", "SR", "MDD"]]


class Backtester:
    def __init__(self, graph, data: DataProvider, decision_interval: int = 1, allow_short: bool = False,
                 cost_bps: float = 5.0, reflect: bool = True, on_decision=None):
        self.graph, self.data = graph, data
        self.interval, self.allow_short, self.cost_bps = max(1, decision_interval), allow_short, cost_bps
        self.reflect, self.on_decision = reflect, on_decision

    def run(self, ticker: str, start, end) -> BacktestResult:
        days = self.data.trading_days(ticker, start, end)
        if len(days) < 2:
            raise ValueError("need at least two trading days in the backtest window")
        close = self.data.history(ticker)["Close"].loc[days[0]:days[-1]]
        decision_days = days[:-1:self.interval]

        targets: dict[pd.Timestamp, float] = {}
        decisions: list[dict] = []
        position = 0.0
        prev: tuple[AgentState, pd.Timestamp] | None = None
        for i, day in enumerate(decision_days):
            if self.reflect and prev is not None:
                s, d0 = prev
                self.graph.reflect(s, float(close.loc[day] / close.loc[d0] - 1))
            state = self.graph.propagate(ticker, day, current_position=position)
            fd = state.final_decision
            position = action_to_target(state.action, float(fd.get("position_size", 1.0)), position, self.allow_short)
            targets[day] = position
            rec = {"date": str(day.date()), "action": state.action, "position": position,
                   "confidence": fd.get("confidence"), "rationale": fd.get("rationale", ""), "errors": state.errors}
            decisions.append(rec)
            if self.on_decision:
                self.on_decision(i + 1, len(decision_days), rec, state)
            prev = (state, day)

        tgt = pd.Series(targets, dtype=float)
        rets = simulate(close, tgt, self.cost_bps).iloc[:-1]
        positions = tgt.reindex(close.index).ffill().fillna(0.0)
        base = {}
        for name in ("buy_and_hold", "sma_cross", "macd"):
            # Baselines see full history up to each day (indicators warm up before start).
            full = self.data.history(ticker)["Close"].loc[:days[-1]]
            t = baseline_targets(full, name).loc[days[0]:days[-1]]
            base[name] = metrics(simulate(close, t, self.cost_bps).iloc[:-1])
        return BacktestResult(ticker, rets, positions, decisions, metrics(rets), base)
