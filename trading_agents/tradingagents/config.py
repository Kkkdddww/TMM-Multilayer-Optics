"""Run-time configuration for the TradingAgents framework."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TradingConfig:
    # LLM settings. The paper splits work between "quick-thinking" models
    # (data summarisation) and "deep-thinking" models (debate, decisions).
    # Here both tiers use the same model and are separated by effort level.
    model: str = "claude-opus-5-5"
    quick_effort: str = "low"     # analysts, debate turns
    deep_effort: str = "high"     # research facilitator, trader, fund manager
    max_tokens: int = 16000
    use_server_fallbacks: bool = True

    # Debate settings (paper: n rounds of bull/bear and risk debates).
    max_debate_rounds: int = 1
    max_risk_rounds: int = 1

    # Analysts to run. Any subset of: fundamentals, sentiment, news, technical.
    analysts: tuple[str, ...] = ("fundamentals", "sentiment", "news", "technical")

    # Data windows (calendar days before the decision date).
    price_lookback_days: int = 180
    news_lookback_days: int = 7

    # Reflection memory: how many past lessons to show the decision makers.
    memory_k: int = 3

    language: str = "en"  # "en" or "ko" for report language
    extra: dict = field(default_factory=dict)
