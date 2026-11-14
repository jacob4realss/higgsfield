"""Market data (Yahoo Finance via yfinance) and technical indicators.

`MarketData` is the only class that touches the network, so tests can swap in
a fake with the same methods.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ---------------------------------------------------------------- indicators

def sma(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    return float(close.iloc[-window:].mean())


def rsi(close: pd.Series, window: int = 14) -> float | None:
    """Wilder's RSI."""
    if len(close) <= window:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    last_gain, last_loss = float(gain.iloc[-1]), float(loss.iloc[-1])
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, float] | None:
    if len(close) < slow + signal:
        return None
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return {"macd": float(line.iloc[-1]), "signal": float(sig.iloc[-1]), "histogram": float(line.iloc[-1] - sig.iloc[-1])}


def annualized_volatility(close: pd.Series, periods_per_year: int) -> float | None:
    rets = close.pct_change().dropna()
    if len(rets) < 2:
        return None
    return float(rets.std() * math.sqrt(periods_per_year))


def max_drawdown(close: pd.Series) -> float | None:
    if close.empty:
        return None
    running_max = close.cummax()
    return float((close / running_max - 1).min())


def atr(df: pd.DataFrame, window: int = 14) -> float | None:
    if len(df) <= window:
        return None
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / window, adjust=False).mean().iloc[-1])


def technical_summary(df: pd.DataFrame, periods_per_year: int) -> dict[str, Any]:
    close = df["Close"].dropna()
    last = float(close.iloc[-1])

    def pct_vs(level: float | None) -> float | None:
        return None if level is None else round((last / level - 1) * 100, 2)

    sma20, sma50, sma200 = sma(close, 20), sma(close, 50), sma(close, 200)
    window = close.iloc[-min(len(close), periods_per_year):]
    return {
        "last_close": round(last, 6),
        "bars": len(close),
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "pct_vs_sma50": pct_vs(sma50), "pct_vs_sma200": pct_vs(sma200),
        "rsi14": rsi(close),
        "macd": macd(close),
        "atr14": atr(df) if {"High", "Low"} <= set(df.columns) else None,
        "annualized_volatility": annualized_volatility(close, periods_per_year),
        "max_drawdown": max_drawdown(close),
        "trailing_high": float(window.max()), "trailing_low": float(window.min()),
        "return_1m_pct": _period_return(close, 21 if periods_per_year == 252 else 30),
        "return_3m_pct": _period_return(close, 63 if periods_per_year == 252 else 90),
    }


def _period_return(close: pd.Series, bars: int) -> float | None:
    if len(close) <= bars:
        return None
    return round((float(close.iloc[-1]) / float(close.iloc[-bars - 1]) - 1) * 100, 2)


# ---------------------------------------------------------------- data access

class MarketData:
    """Thin wrapper around yfinance."""

    def __init__(self) -> None:
        import yfinance as yf  # imported lazily so tests don't need network setup

        self._yf = yf

    def history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        df = self._yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            raise LookupError(f"No price history returned for {symbol!r}")
        return df

    def last_price(self, symbol: str) -> float:
        fi = self._yf.Ticker(symbol).fast_info
        try:
            price = fi.last_price
        except Exception:  # fast_info can raise on odd symbols
            price = None
        if price is None or (isinstance(price, float) and math.isnan(price)):
            price = float(self.history(symbol, period="5d")["Close"].dropna().iloc[-1])
        return float(price)

    def quote(self, symbol: str) -> dict[str, Any]:
        fi = self._yf.Ticker(symbol).fast_info
        out: dict[str, Any] = {"symbol": symbol}
        for key in ("last_price", "previous_close", "open", "day_high", "day_low",
                    "year_high", "year_low", "last_volume", "market_cap", "currency"):
            try:
                out[key] = getattr(fi, key)
            except Exception:
                out[key] = None
        if out.get("last_price") is None:
            out["last_price"] = self.last_price(symbol)
        lp, pc = out.get("last_price"), out.get("previous_close")
        if lp and pc:
            out["change_pct"] = round((lp / pc - 1) * 100, 3)
        return out

    def info(self, symbol: str) -> dict[str, Any]:
        return dict(self._yf.Ticker(symbol).info or {})

    def news(self, symbol: str, limit: int = 8) -> list[dict[str, Any]]:
        items = self._yf.Ticker(symbol).news or []
        out = []
        for item in items[:limit]:
            c = item.get("content", item)  # yfinance >= 0.2.50 nests under "content"
            out.append({
                "title": c.get("title"),
                "summary": c.get("summary") or c.get("description"),
                "published": c.get("pubDate") or c.get("providerPublishTime"),
                "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher"),
                "url": (c.get("canonicalUrl") or {}).get("url") or c.get("link"),
            })
        return out
