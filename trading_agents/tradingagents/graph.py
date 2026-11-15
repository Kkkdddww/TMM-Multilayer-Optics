"""Orchestrates one trading decision through the whole organisation."""
from __future__ import annotations

import logging

import pandas as pd

from .agents import Organisation
from .config import TradingConfig
from .data import DataProvider
from .indicators import technical_snapshot
from .llm import LLM, LLMRefusal
from .memory import Lesson, ReflectionMemory
from .state import AgentState

log = logging.getLogger(__name__)


class TradingAgentsGraph:
    def __init__(self, llm: LLM, data: DataProvider, config: TradingConfig | None = None,
                 memory: ReflectionMemory | None = None):
        self.cfg = config or TradingConfig()
        self.llm, self.data = llm, data
        self.memory = memory or ReflectionMemory()
        self.org = Organisation(llm, self.cfg)

    def gather(self, ticker: str, date) -> AgentState:
        date = pd.Timestamp(date)
        prices = self.data.get_prices(ticker, date, self.cfg.price_lookback_days)
        if prices.empty:
            raise ValueError(f"no price data for {ticker} on or before {date.date()}")
        bars = prices.tail(15).round(2)
        bars.index = bars.index.date
        return AgentState(
            ticker=ticker,
            date=str(prices.index[-1].date()),
            market=technical_snapshot(prices),
            recent_bars=bars.to_string(),
            fundamentals=self.data.get_fundamentals(ticker, date),
            news=[n.render() for n in self.data.get_news(ticker, date, self.cfg.news_lookback_days)][-30:],
            lessons=self.memory.recall(ticker, str(date.date()), self.cfg.memory_k),
        )

    def propagate(self, ticker: str, date, current_position: float = 0.0) -> AgentState:
        state = self.gather(ticker, date)
        try:
            self.org.analyst_team(state)
            self.org.research_debate(state)
            self.org.trader(state, current_position)
            self.org.risk_debate(state)
            self.org.fund_manager(state, current_position)
        except LLMRefusal as e:
            log.warning("decision pipeline refused for %s %s: %s", ticker, state.date, e)
            state.errors.append(str(e))
            state.final_decision = {"action": "HOLD", "position_size": abs(current_position),
                                    "confidence": 0.0, "rationale": f"Pipeline refused: {e}", "risk_controls": ""}
        return state

    def reflect(self, state: AgentState, realized_return: float) -> Lesson:
        try:
            text = self.org.reflect(state, realized_return)
        except LLMRefusal as e:
            text = f"(no reflection: {e})"
        lesson = Lesson(state.ticker, state.date, state.action, float(realized_return), text)
        self.memory.add(lesson)
        return lesson
