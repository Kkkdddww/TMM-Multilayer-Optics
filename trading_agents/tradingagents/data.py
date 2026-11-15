"""Market data providers.

Every provider enforces point-in-time access: a request "as of" date D only
returns information dated on or before D, so backtests cannot peek ahead.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True)
class NewsItem:
    date: pd.Timestamp
    title: str
    summary: str = ""
    source: str = ""

    def render(self) -> str:
        s = f"[{self.date.date()}] {self.title}"
        if self.source:
            s += f" ({self.source})"
        if self.summary:
            s += f" - {self.summary}"
        return s


class DataProvider:
    """Base class. Subclasses implement ``_load_prices`` and optionally news/fundamentals."""

    def __init__(self) -> None:
        self._cache: dict[str, pd.DataFrame] = {}

    # --- to implement -------------------------------------------------------
    def _load_prices(self, ticker: str) -> pd.DataFrame:
        raise NotImplementedError

    def _load_news(self, ticker: str) -> list[NewsItem]:
        return []

    def get_fundamentals(self, ticker: str, as_of: pd.Timestamp) -> dict:
        return {}

    # --- public, point-in-time API -----------------------------------------
    def history(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._cache:
            df = self._load_prices(ticker)
            missing = [c for c in PRICE_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"{ticker}: price data missing columns {missing}")
            df = df[PRICE_COLUMNS].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            self._cache[ticker] = df.sort_index().dropna(subset=["Close"])
        return self._cache[ticker]

    def get_prices(self, ticker: str, as_of, lookback_days: int = 365) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        df = self.history(ticker)
        return df.loc[(df.index <= as_of) & (df.index > as_of - pd.Timedelta(days=lookback_days))]

    def get_news(self, ticker: str, as_of, lookback_days: int = 7) -> list[NewsItem]:
        as_of = pd.Timestamp(as_of)
        start = as_of - pd.Timedelta(days=lookback_days)
        return [n for n in self._load_news(ticker) if start < n.date <= as_of]

    def trading_days(self, ticker: str, start, end) -> list[pd.Timestamp]:
        idx = self.history(ticker).index
        return list(idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))])


class YFinanceProvider(DataProvider):
    """Yahoo Finance via ``yfinance``.

    Prices are point-in-time. News and fundamentals from Yahoo are *current*
    snapshots, so for historical backtests they leak future information; set
    ``live_extras=False`` (default) to disable them outside live use.
    """

    def __init__(self, start: str = "2015-01-01", end: str | None = None, live_extras: bool = False):
        super().__init__()
        self.start, self.end, self.live_extras = start, end, live_extras

    def _load_prices(self, ticker: str) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(ticker, start=self.start, end=self.end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker} (network blocked or bad ticker?)")
        return df

    def _load_news(self, ticker: str) -> list[NewsItem]:
        if not self.live_extras:
            return []
        import yfinance as yf

        out = []
        for raw in yf.Ticker(ticker).news or []:
            c = raw.get("content", raw)
            ts = c.get("pubDate") or c.get("providerPublishTime")
            if ts is None:
                continue
            date = pd.to_datetime(ts, unit="s" if isinstance(ts, (int, float)) else None)
            date = date.tz_localize(None) if date.tzinfo else date
            provider = c.get("provider")
            out.append(
                NewsItem(
                    date=date,
                    title=c.get("title", ""),
                    summary=c.get("summary", "") or "",
                    source=provider.get("displayName", "") if isinstance(provider, dict) else str(c.get("publisher", "")),
                )
            )
        return out

    def get_fundamentals(self, ticker: str, as_of) -> dict:
        if not self.live_extras:
            return {}
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        keys = [
            "longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE",
            "priceToBook", "profitMargins", "operatingMargins", "returnOnEquity",
            "revenueGrowth", "earningsGrowth", "debtToEquity", "currentRatio",
            "freeCashflow", "dividendYield", "beta",
        ]
        return {k: info[k] for k in keys if k in info}


class CSVProvider(DataProvider):
    """Local CSV files: ``{dir}/{TICKER}.csv`` with Date,Open,High,Low,Close,Volume.

    Optional ``{dir}/{TICKER}_news.csv`` with date,title[,summary,source] and
    ``{dir}/{TICKER}_fundamentals.csv`` with date plus any metric columns (the
    latest row dated on or before the decision date is used).
    """

    def __init__(self, directory: str | Path):
        super().__init__()
        self.dir = Path(directory)

    def _load_prices(self, ticker: str) -> pd.DataFrame:
        df = pd.read_csv(self.dir / f"{ticker}.csv")
        date_col = next(c for c in df.columns if c.lower() in ("date", "datetime", "timestamp"))
        df = df.rename(columns={c: c.capitalize() for c in df.columns if c.lower() in [p.lower() for p in PRICE_COLUMNS]})
        return df.set_index(pd.to_datetime(df[date_col]))

    def _load_news(self, ticker: str) -> list[NewsItem]:
        p = self.dir / f"{ticker}_news.csv"
        if not p.exists():
            return []
        df = pd.read_csv(p).fillna("")
        return [
            NewsItem(pd.Timestamp(r["date"]), str(r["title"]), str(r.get("summary", "")), str(r.get("source", "")))
            for _, r in df.iterrows()
        ]

    def get_fundamentals(self, ticker: str, as_of) -> dict:
        p = self.dir / f"{ticker}_fundamentals.csv"
        if not p.exists():
            return {}
        df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return {}
        row = df.iloc[-1].to_dict()
        row["date"] = str(pd.Timestamp(row["date"]).date())
        return row


class SyntheticProvider(DataProvider):
    """Deterministic regime-switching random walk with synthetic news.

    Useful for offline development and tests. News headlines are generated
    from *past* returns only, so they carry no look-ahead information.
    """

    BULL = ["{t} beats earnings estimates", "Analysts upgrade {t} on strong demand",
            "{t} announces share buyback", "{t} expands into new markets"]
    BEAR = ["{t} misses revenue guidance", "Regulators open probe into {t}",
            "{t} faces supply chain disruptions", "Analyst downgrades {t} citing slowing growth"]
    NEUTRAL = ["{t} to present at industry conference", "{t} names new board member",
               "{t} reaffirms full-year outlook"]

    def __init__(self, start: str = "2023-01-01", end: str = "2025-12-31", seed: int = 7):
        super().__init__()
        self.start, self.end, self.seed = start, end, seed

    def _rng(self, ticker: str) -> np.random.Generator:
        h = int(hashlib.sha256(f"{ticker}:{self.seed}".encode()).hexdigest()[:8], 16)
        return np.random.default_rng(h)

    def _load_prices(self, ticker: str) -> pd.DataFrame:
        rng = self._rng(ticker)
        idx = pd.bdate_range(self.start, self.end)
        n = len(idx)
        regime = np.zeros(n)
        state, mu = 0, [0.0008, -0.0006, 0.0]
        for i in range(n):
            if rng.random() < 0.02:
                state = int(rng.integers(0, 3))
            regime[i] = mu[state]
        rets = regime + rng.normal(0, 0.015, n)
        close = 100 * np.exp(np.cumsum(rets))
        open_ = close * np.exp(rng.normal(0, 0.004, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
        vol = rng.lognormal(15, 0.3, n).astype(int)
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)

    def _load_news(self, ticker: str) -> list[NewsItem]:
        rng = self._rng(ticker + ":news")
        df = self.history(ticker)
        past5 = df["Close"].pct_change(5)
        items = []
        for date, r in past5.items():
            if pd.isna(r) or rng.random() > 0.35:
                continue
            p_bull = 1 / (1 + np.exp(-r * 40))
            u = rng.random()
            pool = self.BULL if u < p_bull * 0.8 else self.BEAR if u < 0.8 else self.NEUTRAL
            items.append(NewsItem(date, pool[int(rng.integers(len(pool)))].format(t=ticker), source="SyntheticWire"))
        return items

    def get_fundamentals(self, ticker: str, as_of) -> dict:
        rng = self._rng(ticker + f":fund:{pd.Timestamp(as_of).quarter}:{pd.Timestamp(as_of).year}")
        return {
            "note": "synthetic fundamentals for offline testing",
            "trailingPE": round(float(rng.uniform(10, 40)), 2),
            "revenueGrowth": round(float(rng.normal(0.08, 0.1)), 3),
            "profitMargins": round(float(rng.uniform(0.05, 0.3)), 3),
            "debtToEquity": round(float(rng.uniform(10, 150)), 1),
            "returnOnEquity": round(float(rng.uniform(0.02, 0.35)), 3),
        }
