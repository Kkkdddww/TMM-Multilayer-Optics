"""Technical indicators computed with pandas (no TA-Lib dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(s, n)
    sd = s.rolling(n, min_periods=n).std()
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + k * sd, "bb_lower": mid - k * sd})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["sma_10"], out["sma_50"], out["sma_200"] = sma(c, 10), sma(c, 50), sma(c, 200)
    out["ema_10"] = ema(c, 10)
    out["rsi_14"] = rsi(c, 14)
    out = out.join(macd(c)).join(bollinger(c))
    out["atr_14"] = atr(out)
    out["ret_1d"] = c.pct_change()
    out["vol_20d"] = out["ret_1d"].rolling(20).std() * np.sqrt(252)
    return out


def _f(x) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 4)


def technical_snapshot(df: pd.DataFrame) -> dict:
    """Latest indicator values plus a bounded composite score in [-1, 1].

    The score is a transparent quantitative prior handed to the agents; it is
    not the decision itself.
    """
    if len(df) < 2:
        return {"composite_score": 0.0, "bars": len(df)}
    ind = compute_all(df)
    last = ind.iloc[-1]
    c = float(last["Close"])

    def ret(n: int) -> float | None:
        return _f(c / float(ind["Close"].iloc[-n - 1]) - 1) if len(ind) > n else None

    votes: list[float] = []
    if not pd.isna(last["sma_50"]):
        votes.append(1.0 if c > last["sma_50"] else -1.0)
    if not pd.isna(last["sma_10"]) and not pd.isna(last["sma_50"]):
        votes.append(1.0 if last["sma_10"] > last["sma_50"] else -1.0)
    if not pd.isna(last["macd_hist"]):
        votes.append(float(np.tanh(last["macd_hist"] / (0.01 * c + 1e-9))))
    if not pd.isna(last["rsi_14"]):
        r = float(last["rsi_14"])
        votes.append(-1.0 if r > 75 else 1.0 if r < 25 else (r - 50) / 50)
    m20 = ret(20)
    if m20 is not None:
        votes.append(float(np.tanh(m20 * 10)))
    score = float(np.clip(np.mean(votes), -1, 1)) if votes else 0.0

    return {
        "date": str(ind.index[-1].date()),
        "close": _f(c),
        "return_5d": ret(5),
        "return_20d": m20,
        "return_60d": ret(60),
        "sma_10": _f(last["sma_10"]),
        "sma_50": _f(last["sma_50"]),
        "sma_200": _f(last["sma_200"]),
        "rsi_14": _f(last["rsi_14"]),
        "macd": _f(last["macd"]),
        "macd_signal": _f(last["macd_signal"]),
        "macd_hist": _f(last["macd_hist"]),
        "bb_upper": _f(last["bb_upper"]),
        "bb_lower": _f(last["bb_lower"]),
        "atr_14": _f(last["atr_14"]),
        "annualized_vol_20d": _f(last["vol_20d"]),
        "volume_vs_20d_avg": _f(last["Volume"] / ind["Volume"].tail(20).mean())
        if "Volume" in ind and ind["Volume"].tail(20).mean() > 0
        else None,
        "composite_score": round(score, 4),
        "bars": len(ind),
    }
