"""The agent roles of the TradingAgents organisation.

Analyst team -> Researcher team (bull vs bear debate + facilitator) -> Trader
-> Risk management team (aggressive / neutral / conservative debate)
-> Fund manager (final approval).
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from . import prompts as P
from .config import TradingConfig
from .llm import LLM, LLMRefusal
from .state import AgentState, DebateTurn

log = logging.getLogger(__name__)


def _clip01(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _json(d: dict) -> str:
    return json.dumps(d, indent=2, ensure_ascii=False, default=str)


def _transcript(turns: list[DebateTurn]) -> str:
    return "\n\n".join(f"### {t.speaker}\n{t.content}" for t in turns) or "(no arguments yet)"


def _reports(state: AgentState) -> str:
    return "\n\n".join(f"## {k.title()} Analyst Report\n{v}" for k, v in state.analyst_reports.items())


def _lessons(state: AgentState) -> str:
    return "\n".join(f"- {x}" for x in state.lessons) or "(none)"


class Organisation:
    def __init__(self, llm: LLM, config: TradingConfig):
        self.llm, self.cfg = llm, config
        self.suffix = P.lang_suffix(config.language)

    def _sys(self, role_prompt: str) -> str:
        return f"{P.FIRM}\n\n{role_prompt}{self.suffix}"

    def _hints(self, state: AgentState, role: str, stance: str | None = None) -> dict:
        return {"role": role, "stance": stance, "score": state.market.get("composite_score", 0.0)}

    # --- I. Analyst team ----------------------------------------------------
    def _analyst_input(self, name: str, s: AgentState) -> str:
        head = f"Ticker: {s.ticker}\nDecision date: {s.date}\n\n"
        if name == "technical":
            return head + f"Technical indicators (as of close):\n{_json(s.market)}\n\nRecent daily bars:\n{s.recent_bars}"
        if name == "fundamentals":
            return head + f"Fundamental data:\n{_json(s.fundamentals) if s.fundamentals else '(no fundamental data available)'}"
        news = "\n".join(s.news) or "(no news in the lookback window)"
        if name == "news":
            return head + f"Recent headlines:\n{news}"
        # sentiment: headline tone plus price/volume behaviour
        keys = ("close", "return_5d", "return_20d", "rsi_14", "volume_vs_20d_avg", "annualized_vol_20d")
        pv = {k: s.market.get(k) for k in keys}
        return head + f"Recent headlines:\n{news}\n\nPrice/volume context:\n{_json(pv)}"

    def run_analyst(self, name: str, state: AgentState) -> tuple[str, str]:
        try:
            text = self.llm.text(
                self._sys(P.ANALYSTS[name] + "\n\n" + P.ANALYST_FORMAT),
                self._analyst_input(name, state),
                deep=False,
                hints=self._hints(state, f"{name}_analyst"),
            )
        except LLMRefusal as e:
            state.errors.append(f"{name} analyst: {e}")
            text = f"(report unavailable: {e})"
        return name, text

    def analyst_team(self, state: AgentState) -> None:
        names = [a for a in self.cfg.analysts if a in P.ANALYSTS]
        with ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
            for name, text in pool.map(lambda n: self.run_analyst(n, state), names):
                state.analyst_reports[name] = text

    # --- II. Researcher team ------------------------------------------------
    def research_debate(self, state: AgentState) -> None:
        sides = [("Bull Researcher", P.BULL, "bull"), ("Bear Researcher", P.BEAR, "bear")]
        for rnd in range(self.cfg.max_debate_rounds):
            for speaker, role, stance in sides:
                prompt = (
                    f"Ticker: {state.ticker} | Date: {state.date} | Round {rnd + 1}/{self.cfg.max_debate_rounds}\n\n"
                    f"{_reports(state)}\n\n# Debate so far\n{_transcript(state.investment_debate)}\n\n"
                    f"# Lessons from past decisions\n{_lessons(state)}\n\n"
                    "Give your argument (under 250 words)."
                )
                text = self.llm.text(self._sys(role), prompt, deep=False, hints=self._hints(state, speaker, stance))
                state.investment_debate.append(DebateTurn(speaker, text))

        prompt = (
            f"Ticker: {state.ticker} | Date: {state.date}\n\n{_reports(state)}\n\n"
            f"# Bull vs Bear debate\n{_transcript(state.investment_debate)}\n\n"
            f"# Lessons from past decisions\n{_lessons(state)}\n\nDecide which side prevails and write the plan."
        )
        state.research_decision = self.llm.structured(
            self._sys(P.RESEARCH_MANAGER), prompt, P.RESEARCH_SCHEMA, deep=True,
            hints=self._hints(state, "Research Manager"),
        )

    # --- III. Trader --------------------------------------------------------
    def trader(self, state: AgentState, current_position: float) -> None:
        prompt = (
            f"Ticker: {state.ticker} | Date: {state.date} | Current position: {current_position:+.2f} "
            "(fraction of capital; + long, - short, 0 flat)\n\n"
            f"{_reports(state)}\n\n# Research Manager decision\n{_json(state.research_decision)}\n\n"
            f"# Lessons from past decisions\n{_lessons(state)}\n\nWrite your trading proposal."
        )
        out = self.llm.structured(self._sys(P.TRADER), prompt, P.TRADER_SCHEMA, deep=True, hints=self._hints(state, "Trader"))
        out["position_size"], out["confidence"] = _clip01(out.get("position_size")), _clip01(out.get("confidence"))
        state.trader_proposal = out

    # --- IV. Risk management team -------------------------------------------
    def risk_debate(self, state: AgentState) -> None:
        for rnd in range(self.cfg.max_risk_rounds):
            for stance in ("aggressive", "neutral", "conservative"):
                speaker = f"{stance.title()} Risk Analyst"
                prompt = (
                    f"Ticker: {state.ticker} | Date: {state.date} | Round {rnd + 1}/{self.cfg.max_risk_rounds}\n\n"
                    f"# Trader proposal\n{_json(state.trader_proposal)}\n\n"
                    f"# Market snapshot\n{_json(state.market)}\n\n{_reports(state)}\n\n"
                    f"# Risk debate so far\n{_transcript(state.risk_debate)}\n\n{P.RISK_FORMAT}"
                )
                text = self.llm.text(self._sys(P.RISK[stance]), prompt, deep=False, hints=self._hints(state, speaker, stance))
                state.risk_debate.append(DebateTurn(speaker, text))

    # --- V. Fund manager ----------------------------------------------------
    def fund_manager(self, state: AgentState, current_position: float) -> None:
        prompt = (
            f"Ticker: {state.ticker} | Date: {state.date} | Current position: {current_position:+.2f}\n\n"
            f"# Research Manager decision\n{_json(state.research_decision)}\n\n"
            f"# Trader proposal\n{_json(state.trader_proposal)}\n\n"
            f"# Risk team debate\n{_transcript(state.risk_debate)}\n\n"
            f"# Lessons from past decisions\n{_lessons(state)}\n\nApprove or adjust and give the final decision."
        )
        out = self.llm.structured(self._sys(P.FUND_MANAGER), prompt, P.FINAL_SCHEMA, deep=True, hints=self._hints(state, "Fund Manager"))
        out["position_size"], out["confidence"] = _clip01(out.get("position_size")), _clip01(out.get("confidence"))
        state.final_decision = out

    # --- Reflection ---------------------------------------------------------
    def reflect(self, state: AgentState, realized_return: float) -> str:
        prompt = (
            f"Ticker: {state.ticker} | Decision date: {state.date}\n"
            f"Final decision: {_json(state.final_decision)}\n"
            f"Technical snapshot: {_json(state.market)}\n"
            f"Research decision: {_json(state.research_decision)}\n"
            f"Realised return of the underlying until the next decision: {realized_return:+.2%}"
        )
        return self.llm.text(self._sys(P.REFLECTION), prompt, deep=False, hints=self._hints(state, "Reflection"))
