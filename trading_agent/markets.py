"""One profile per market. Each profile defines an independent agent:
its own instructions, symbol rules, risk limits, accounting and extra tools.

To add a market, create another `MarketProfile` and register it in `MARKETS`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .portfolio import RiskLimits, Valuer

if False:  # typing only
    from .market_data import MarketData


@dataclass(frozen=True)
class MarketProfile:
    key: str
    title: str
    instructions: str                      # market-specific part of the system prompt
    limits: RiskLimits
    settlement: str                        # "cash" or "margin"
    periods_per_year: int                  # bars per year for daily data (volatility scaling)
    normalize_symbol: Callable[[str], str]  # raises ValueError for symbols outside this market
    make_valuer: Callable[["MarketData"], Valuer]
    extra_tools: tuple[str, ...] = ()      # names of market-specific tools (see tools.py)
    example_symbols: tuple[str, ...] = field(default_factory=tuple)


def _linear(_data: "MarketData") -> Valuer:
    return lambda symbol, qty, price: qty * price


# ------------------------------------------------------------------ stocks

_STOCK_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")


def normalize_stock(symbol: str) -> str:
    s = symbol.strip().upper().replace(".", "-").replace("/", "-")
    if s.endswith("-USD") or "=" in s or s.startswith("^"):
        raise ValueError(f"{symbol!r} is not a US stock/ETF ticker (crypto, forex, futures and indices belong to other agents)")
    if not _STOCK_RE.match(s):
        raise ValueError(f"{symbol!r} is not a valid US ticker (expected e.g. AAPL, SPY, BRK-B)")
    return s


STOCKS = MarketProfile(
    key="stocks",
    title="US Stocks & ETFs",
    instructions="""\
You trade US-listed stocks and ETFs (NYSE, Nasdaq) only. Tickers look like AAPL, MSFT, SPY, BRK-B.

Market specifics:
- Regular session is 9:30-16:00 US/Eastern, Mon-Fri. Outside it, the latest price is stale; say so and
  note that a real order would fill at the next open, possibly with a gap.
- Combine fundamentals (get_fundamentals: valuation, growth, margins, balance sheet, analyst targets,
  next earnings date) with technicals. Flag upcoming earnings: holding through them is a binary risk.
- Prefer diversified exposure; consider whether a broad ETF achieves the user's goal more simply.
- Long-only cash account, whole shares, no margin, no shorting.""",
    limits=RiskLimits(starting_cash=100_000, max_position_pct=0.20, max_order_usd=25_000,
                      max_gross_exposure=1.0, allow_short=False, quantity_step=1,
                      slippage_bps=5, commission_bps=0),
    settlement="cash",
    periods_per_year=252,
    normalize_symbol=normalize_stock,
    make_valuer=_linear,
    extra_tools=("get_fundamentals",),
    example_symbols=("AAPL", "NVDA", "MSFT", "SPY", "QQQ"),
)


# ------------------------------------------------------------------ crypto

def normalize_crypto(symbol: str) -> str:
    s = symbol.strip().upper().replace("/", "-").replace("_", "-")
    s = re.sub(r"-?USDT?$", "", s) if not s.endswith("-USD") else s[:-4]
    if not re.match(r"^[A-Z0-9]{2,10}$", s):
        raise ValueError(f"{symbol!r} is not a crypto symbol (expected e.g. BTC, ETH-USD, SOL/USD)")
    return f"{s}-USD"


CRYPTO = MarketProfile(
    key="crypto",
    title="Crypto (spot, USD pairs)",
    instructions="""\
You trade spot cryptocurrencies against USD only. Symbols are Yahoo-style: BTC-USD, ETH-USD, SOL-USD
(you can pass BTC, ETH/USD, etc.; they are normalized).

Market specifics:
- Trades 24/7; there is no close. Use 365 periods/year when reasoning about volatility.
- Volatility is several times that of equities: 60-100%+ annualized is normal, and 20-30% daily moves
  happen in small caps. Size positions so a 50% drawdown in one coin is survivable.
- Fundamentals are thin. Use get_asset_profile (market cap, supply, volume) plus web search for
  catalysts: ETF flows, regulation, exchange events, token unlocks, protocol upgrades, hacks.
- Treat anything outside the top ~20 by market cap as speculative and keep it small. Check volume:
  illiquid coins have wide real-world spreads.
- Long-only spot account, fractional quantities allowed, no leverage, no shorting. Fees are 10 bps.""",
    limits=RiskLimits(starting_cash=50_000, max_position_pct=0.30, max_order_usd=10_000,
                      max_gross_exposure=1.0, allow_short=False, quantity_step=0.00001,
                      slippage_bps=10, commission_bps=10),
    settlement="cash",
    periods_per_year=365,
    normalize_symbol=normalize_crypto,
    make_valuer=_linear,
    extra_tools=("get_asset_profile",),
    example_symbols=("BTC-USD", "ETH-USD", "SOL-USD"),
)


# ------------------------------------------------------------------ forex

MAJOR_CURRENCIES = ("USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")


def normalize_forex(symbol: str) -> str:
    s = re.sub(r"[\s/\-_]", "", symbol.strip().upper()).removesuffix("=X")
    if not re.match(r"^[A-Z]{6}$", s) or s[:3] == s[3:]:
        raise ValueError(f"{symbol!r} is not a currency pair (expected e.g. EURUSD, EUR/USD, USDJPY)")
    return f"{s}=X"


def make_forex_valuer(data: "MarketData", ttl: float = 60.0) -> Valuer:
    """Pair BASEQUOTE: quantity is in BASE units, price is QUOTE per BASE.
    Notional (USD) = quantity * price * (QUOTE -> USD rate, live)."""
    cache: dict[str, tuple[float, float]] = {}

    def quote_to_usd(ccy: str) -> float:
        if ccy == "USD":
            return 1.0
        hit = cache.get(ccy)
        if hit and time.time() - hit[1] < ttl:
            return hit[0]
        try:
            rate = data.last_price(f"{ccy}USD=X")
        except Exception:
            rate = 1.0 / data.last_price(f"USD{ccy}=X")
        cache[ccy] = (rate, time.time())
        return rate

    def valuer(symbol: str, qty: float, price: float) -> float:
        pair = symbol.removesuffix("=X")
        return qty * price * quote_to_usd(pair[3:6])

    return valuer


FOREX = MarketProfile(
    key="forex",
    title="Forex (spot FX, margin)",
    instructions="""\
You trade spot currency pairs on margin. Symbols: EURUSD=X, USDJPY=X, GBPUSD=X, etc. (EUR/USD and
EURUSD are normalized). Quantity is in units of the BASE currency (the first one); it must be a
multiple of 1,000 (a micro lot). 100,000 units = 1 standard lot.

Market specifics:
- Trades 24h Sun 17:00 - Fri 17:00 US/Eastern. Weekend gaps are a risk for open positions.
- This is a margin account: opening a position moves no cash; P&L is marked in USD. Leverage is
  capped at 5x gross. Shorting is normal (selling EURUSD = long USD vs EUR).
- No company fundamentals. Drivers are central-bank policy and rate differentials, inflation, growth
  and risk sentiment. Use web search for the economic calendar (CPI, NFP, central-bank meetings) and
  avoid opening new risk right before a major release.
- Use get_currency_strength to see which currencies are leading or lagging against USD.
- Size with ATR: state a stop distance (e.g. 1.5-2x ATR) and keep the loss at the stop to <= 1% of
  equity. 1 pip = 0.0001 for most pairs, 0.01 for JPY pairs.""",
    limits=RiskLimits(starting_cash=25_000, max_position_pct=3.0, max_order_usd=100_000,
                      max_gross_exposure=5.0, allow_short=True, quantity_step=1_000,
                      slippage_bps=1, commission_bps=0),
    settlement="margin",
    periods_per_year=260,
    normalize_symbol=normalize_forex,
    make_valuer=make_forex_valuer,
    extra_tools=("get_currency_strength",),
    example_symbols=("EURUSD=X", "USDJPY=X", "GBPUSD=X"),
)


# Enabled agents. CRYPTO and FOREX above are drafts: their market-specific tools
# (get_asset_profile, get_currency_strength) are not implemented yet, so they are
# not registered. Add them here once they are finished and tested.
MARKETS: dict[str, MarketProfile] = {m.key: m for m in (STOCKS,)}
PLANNED_MARKETS: dict[str, MarketProfile] = {m.key: m for m in (CRYPTO, FOREX)}


def get_market(key: str) -> MarketProfile:
    try:
        return MARKETS[key.lower()]
    except KeyError:
        if key.lower() in PLANNED_MARKETS:
            raise ValueError(f"the {key} agent is planned but not enabled yet; available: {', '.join(MARKETS)}") from None
        raise ValueError(f"unknown market {key!r}; choose from {', '.join(MARKETS)}") from None
