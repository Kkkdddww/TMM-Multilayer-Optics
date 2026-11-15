"""Render an AgentState as a readable Markdown report."""
from __future__ import annotations

import json

from .state import AgentState


def render_markdown(s: AgentState) -> str:
    fd = s.final_decision
    lines = [
        f"# TradingAgents decision: {s.ticker} on {s.date}",
        "",
        f"**Final action: {fd.get('action', 'HOLD')}** | position size {fd.get('position_size', 0):.2f} "
        f"| confidence {fd.get('confidence', 0):.2f}",
        "",
        f"> {fd.get('rationale', '')}",
        "",
        f"**Risk controls:** {fd.get('risk_controls', '')}",
        "",
        "## Market snapshot",
        "```json",
        json.dumps(s.market, indent=2),
        "```",
        "## I. Analyst team",
    ]
    for k, v in s.analyst_reports.items():
        lines += [f"### {k.title()} analyst", v, ""]
    lines.append("## II. Researcher debate")
    for t in s.investment_debate:
        lines += [f"### {t.speaker}", t.content, ""]
    rd = s.research_decision
    lines += [
        f"**Research Manager: {rd.get('winner', '?')} prevails.**", "",
        *[f"- {a}" for a in rd.get("key_arguments", [])], "",
        rd.get("investment_plan", ""), "",
        "## III. Trader proposal",
        "```json", json.dumps(s.trader_proposal, indent=2, ensure_ascii=False), "```",
        "## IV. Risk management debate",
    ]
    for t in s.risk_debate:
        lines += [f"### {t.speaker}", t.content, ""]
    if s.lessons:
        lines += ["## Lessons recalled", *[f"- {x}" for x in s.lessons], ""]
    if s.errors:
        lines += ["## Errors", *[f"- {e}" for e in s.errors], ""]
    return "\n".join(lines)
