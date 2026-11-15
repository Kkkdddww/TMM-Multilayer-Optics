"""Command-line interface.

    python -m tradingagents decide   --ticker NVDA --date 2024-06-03
    python -m tradingagents backtest --ticker NVDA --start 2024-01-02 --end 2024-03-29 --interval 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .backtest import Backtester
from .config import TradingConfig
from .data import CSVProvider, SyntheticProvider, YFinanceProvider
from .graph import TradingAgentsGraph
from .llm import ClaudeLLM, MockLLM
from .memory import ReflectionMemory
from .report import render_markdown


def build(args):
    cfg = TradingConfig(
        model=args.model, quick_effort=args.quick_effort, deep_effort=args.deep_effort,
        max_debate_rounds=args.rounds, max_risk_rounds=args.risk_rounds, language=args.language,
        analysts=tuple(args.analysts.split(",")),
    )
    if args.provider == "yfinance":
        data = YFinanceProvider(live_extras=args.live_extras)
    elif args.provider == "csv":
        data = CSVProvider(args.data_dir)
    else:
        data = SyntheticProvider()
    llm = MockLLM() if args.llm == "mock" else ClaudeLLM(
        cfg.model, cfg.quick_effort, cfg.deep_effort, cfg.max_tokens, not args.no_fallbacks)
    memory = ReflectionMemory(Path(args.out) / "memory.json" if args.persist_memory else None)
    return TradingAgentsGraph(llm, data, cfg, memory), data, llm


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tradingagents", description="Multi-agent LLM trading (TradingAgents).")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ticker", required=True)
    common.add_argument("--provider", choices=["synthetic", "yfinance", "csv"], default="synthetic")
    common.add_argument("--data-dir", default="data")
    common.add_argument("--live-extras", action="store_true", help="yfinance: include current news/fundamentals (live use only)")
    common.add_argument("--llm", choices=["claude", "mock"], default="claude")
    common.add_argument("--model", default="claude-opus-5-5")
    common.add_argument("--quick-effort", default="low")
    common.add_argument("--deep-effort", default="high")
    common.add_argument("--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    common.add_argument("--rounds", type=int, default=1, help="bull/bear debate rounds")
    common.add_argument("--risk-rounds", type=int, default=1)
    common.add_argument("--analysts", default="fundamentals,sentiment,news,technical")
    common.add_argument("--language", choices=["en", "ko"], default="en")
    common.add_argument("--out", default="results")
    common.add_argument("--persist-memory", action="store_true")
    common.add_argument("-v", "--verbose", action="store_true")

    d = sub.add_parser("decide", parents=[common], help="one decision for a ticker/date")
    d.add_argument("--date", default=str(pd.Timestamp.today().date()))
    d.add_argument("--position", type=float, default=0.0)

    b = sub.add_parser("backtest", parents=[common], help="walk-forward backtest")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--interval", type=int, default=1, help="trading days between decisions")
    b.add_argument("--allow-short", action="store_true")
    b.add_argument("--cost-bps", type=float, default=5.0)
    b.add_argument("--no-reflect", action="store_true")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    graph, data, llm = build(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "decide":
        state = graph.propagate(args.ticker, args.date, args.position)
        md = render_markdown(state)
        stem = out / f"{args.ticker}_{state.date}"
        stem.with_suffix(".md").write_text(md)
        stem.with_suffix(".json").write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False, default=str))
        fd = state.final_decision
        print(f"{args.ticker} {state.date}: {fd.get('action')} size={fd.get('position_size', 0):.2f} "
              f"conf={fd.get('confidence', 0):.2f}\n{fd.get('rationale', '')}\nReport: {stem.with_suffix('.md')}")
    else:
        def progress(i, n, rec, _state):
            print(f"[{i}/{n}] {rec['date']} {rec['action']:<4} pos={rec['position']:+.2f}", file=sys.stderr)

        bt = Backtester(graph, data, args.interval, args.allow_short, args.cost_bps, not args.no_reflect, progress)
        res = bt.run(args.ticker, args.start, args.end)
        summary = res.summary()
        print(summary.to_string(float_format=lambda x: f"{x:+.4f}"))
        stem = out / f"backtest_{args.ticker}_{args.start}_{args.end}"
        summary.to_csv(stem.with_suffix(".csv"))
        stem.with_suffix(".json").write_text(json.dumps(
            {"metrics": res.metrics, "baselines": res.baselines, "decisions": res.decisions}, indent=2, ensure_ascii=False))
        print(f"Saved: {stem}.csv / .json")
    print(f"LLM calls: {llm.usage.calls}, input tokens: {llm.usage.input_tokens}, output tokens: {llm.usage.output_tokens}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
