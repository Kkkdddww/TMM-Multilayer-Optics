"""Shared state passed between agents (the paper's structured communication protocol)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DebateTurn:
    speaker: str
    content: str


@dataclass
class AgentState:
    ticker: str
    date: str
    # Raw inputs gathered by the framework.
    market: dict = field(default_factory=dict)          # technical snapshot
    recent_bars: str = ""                               # recent OHLCV table (text)
    fundamentals: dict = field(default_factory=dict)
    news: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)    # retrieved reflection memory
    # Structured documents produced by agents.
    analyst_reports: dict[str, str] = field(default_factory=dict)
    investment_debate: list[DebateTurn] = field(default_factory=list)
    research_decision: dict = field(default_factory=dict)
    trader_proposal: dict = field(default_factory=dict)
    risk_debate: list[DebateTurn] = field(default_factory=list)
    final_decision: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def action(self) -> str:
        return self.final_decision.get("action", "HOLD")

    def to_dict(self) -> dict:
        return asdict(self)
