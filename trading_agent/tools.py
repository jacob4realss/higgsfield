"""Tools the agent can call. `build_tools` binds them to one market's profile,
data source and paper account, so each market agent only sees its own world."""

from __future__ import annotations

import json
import math
from typing import Any, Callable

from anthropic import beta_tool

from .markets import MarketProfile
from .portfolio import OrderRejected, PaperBroker

Approver = Callable[[dict, str], bool]

FUNDAMENTAL_FIELDS = (
    "longName", "sector", "industry", "country", "marketCap", "enterpriseValue",
    "trailingPE", "forwardPE", "pegRatio", "priceToBook", "priceToSalesTrailing12Months",
    "enterpriseToEbitda", "profitMargins", "operatingMargins", "grossMargins",
    "revenueGrowth", "earningsGrowth", "returnOnEquity", "returnOnAssets",
    "totalRevenue", "freeCashflow", "totalCash", "totalDebt", "debtToEquity", "currentRatio",
    "dividendYield", "payoutRatio", "beta", "sharesOutstanding", "shortPercentOfFloat",
    "recommendationKey", "targetMeanPrice", "targetLowPrice", "targetHighPrice",
    "numberOfAnalystOpinions", "earningsTimestamp", "exDividendDate", "quoteType",
)

VALID_PERIODS = ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
VALID_INTERVALS = ("1d", "1wk", "1mo")


def _clean(obj: Any) -> Any:
    """Make values JSON-safe (NaN -> None, numpy -> float, round floats)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except Exception:
            pass
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else round(obj, 6)
    return obj


def _ok(payload: Any) -> str:
    return json.dumps(_clean(payload), default=str)


def _err(message: str) -> str:
    return json.dumps({"error": message})


def build_tools(profile: MarketProfile, data, broker: PaperBroker, approve: Approver) -> list:
    norm = profile.normalize_symbol

    def mark_prices() -> dict[str, float]:
        return {sym: data.last_price(sym) for sym in broker.state.positions}

    # ------------------------------------------------------------ research
    @beta_tool
    def get_quote(symbols: list[str]) -> str:
        """Get the latest quote for up to 10 symbols: last price, previous close, % change, day and
        52-week range, volume, market cap. Prices may be delayed ~15 minutes.

        Args:
            symbols: Ticker symbols, e.g. ["AAPL", "MSFT"].
        """
        out = []
        for raw in symbols[:10]:
            try:
                out.append(data.quote(norm(raw)))
            except Exception as e:
                out.append({"symbol": raw, "error": str(e)})
        return _ok(out)

    @beta_tool
    def get_technicals(symbol: str) -> str:
        """Compute technical indicators from one year of daily bars: SMA 20/50/200 and distance from them,
        RSI(14), MACD(12,26,9), ATR(14), annualized volatility, max drawdown, 52-week high/low, and
        1- and 3-month returns.

        Args:
            symbol: Ticker symbol, e.g. "NVDA".
        """
        from .market_data import technical_summary
        try:
            sym = norm(symbol)
            df = data.history(sym, period="1y", interval="1d")
            return _ok({"symbol": sym, **technical_summary(df, profile.periods_per_year)})
        except Exception as e:
            return _err(str(e))

    @beta_tool
    def get_price_history(symbol: str, period: str = "6mo", interval: str = "1wk") -> str:
        """Get OHLCV price bars (max 120 most recent) to inspect trend and price action.

        Args:
            symbol: Ticker symbol.
            period: Lookback: one of 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
            interval: Bar size: one of 1d, 1wk, 1mo.
        """
        if period not in VALID_PERIODS or interval not in VALID_INTERVALS:
            return _err(f"period must be one of {VALID_PERIODS}; interval one of {VALID_INTERVALS}")
        try:
            sym = norm(symbol)
            df = data.history(sym, period=period, interval=interval).tail(120)
            bars = [
                {"date": idx.strftime("%Y-%m-%d"), "open": r["Open"], "high": r["High"],
                 "low": r["Low"], "close": r["Close"], "volume": r.get("Volume")}
                for idx, r in df.iterrows()
            ]
            return _ok({"symbol": sym, "period": period, "interval": interval, "bars": bars})
        except Exception as e:
            return _err(str(e))

    @beta_tool
    def get_news(symbol: str) -> str:
        """Get recent news headlines and summaries for a symbol from Yahoo Finance.

        Args:
            symbol: Ticker symbol.
        """
        try:
            sym = norm(symbol)
            return _ok({"symbol": sym, "news": data.news(sym)})
        except Exception as e:
            return _err(str(e))

    @beta_tool
    def get_fundamentals(symbol: str) -> str:
        """Get company fundamentals: sector, valuation multiples (P/E, PEG, P/B, EV/EBITDA), margins,
        growth, returns, balance sheet, dividend, beta, short interest, analyst consensus and price
        targets, and the next earnings date (earningsTimestamp, unix seconds).

        Args:
            symbol: Ticker symbol.
        """
        try:
            sym = norm(symbol)
            info = data.info(sym)
            if not info:
                return _err(f"no fundamentals available for {sym}")
            fields = {k: info.get(k) for k in FUNDAMENTAL_FIELDS}
            fields = {k: v for k, v in _clean(fields).items() if v is not None}
            return _ok({"symbol": sym, **fields})
        except Exception as e:
            return _err(str(e))

    # ------------------------------------------------------------ account
    @beta_tool
    def get_portfolio() -> str:
        """Get the paper account marked to market: cash, equity, gross exposure, realized P&L, total
        return, each position's quantity, cost, value, unrealized P&L and weight, and the risk limits."""
        try:
            return _ok(broker.snapshot(mark_prices()))
        except Exception as e:
            return _err(f"could not price the portfolio: {e}")

    @beta_tool
    def get_trade_history(limit: int = 20) -> str:
        """Get the most recent paper trades with their recorded rationale.

        Args:
            limit: Number of most recent trades to return (max 100).
        """
        return _ok(broker.state.trades[-max(1, min(limit, 100)):])

    @beta_tool
    def preview_order(symbol: str, side: str, quantity: float) -> str:
        """Check a market order against the account's risk limits WITHOUT executing it. Returns the
        estimated fill (including slippage and fees), resulting position size and exposure, or the
        reason it would be rejected. Use this before place_order to size trades.

        Args:
            symbol: Ticker symbol.
            side: "buy" or "sell".
            quantity: Number of shares (whole numbers for stocks).
        """
        try:
            sym = norm(symbol)
            return _ok(broker.check_order(sym, side, quantity, data.last_price(sym), mark_prices()))
        except (OrderRejected, ValueError) as e:
            return _err(f"would be rejected: {e}")
        except Exception as e:
            return _err(str(e))

    @beta_tool
    def place_order(symbol: str, side: str, quantity: float, rationale: str) -> str:
        """Place a PAPER market order (simulated; no real money). It fills at the latest price plus
        slippage, subject to risk limits, and only after the user approves it. Only call this when the
        user has asked you to trade or has explicitly agreed to a specific trade you proposed.

        Args:
            symbol: Ticker symbol.
            side: "buy" or "sell".
            quantity: Number of shares (whole numbers for stocks).
            rationale: One to three sentences: thesis, key risk, and exit plan (stop / target).
        """
        try:
            sym = norm(symbol)
            plan = broker.check_order(sym, side, quantity, data.last_price(sym), mark_prices())
        except (OrderRejected, ValueError) as e:
            return _err(f"order rejected: {e}")
        except Exception as e:
            return _err(str(e))
        if not approve(plan, rationale):
            return _err("the user declined this order; it was not placed. Ask what they want to change.")
        trade = broker.execute(plan, rationale)
        return _ok({"status": "filled (paper)", "trade": trade.__dict__})

    available = {
        "get_fundamentals": get_fundamentals,
    }
    tools = [get_quote, get_technicals, get_price_history, get_news]
    for name in profile.extra_tools:
        if name not in available:
            raise NotImplementedError(f"tool {name!r} for market {profile.key!r} is not implemented yet")
        tools.append(available[name])
    tools += [get_portfolio, get_trade_history, preview_order, place_order]
    return tools
