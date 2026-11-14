"""Offline tests for the stocks agent: symbol rules, indicators, paper broker, tools.
Run: python -m pytest trading_agent/tests -q"""

import json

import numpy as np
import pandas as pd
import pytest

from trading_agent.market_data import rsi, sma, technical_summary
from trading_agent.markets import STOCKS, get_market, normalize_stock
from trading_agent.portfolio import OrderRejected, PaperBroker, RiskLimits
from trading_agent.tools import build_tools


class FakeData:
    def __init__(self, prices):
        self.prices = dict(prices)

    def last_price(self, s):
        return self.prices[s]

    def quote(self, s):
        return {"symbol": s, "last_price": self.prices[s]}

    def history(self, s, period="1y", interval="1d"):
        idx = pd.date_range("2025-01-01", periods=260, freq="B")
        close = pd.Series(np.linspace(80, self.prices[s], 260), index=idx)
        return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6})

    def info(self, s):
        return {"longName": "Test Co", "trailingPE": float("nan"), "forwardPE": 20.0}

    def news(self, s, limit=8):
        return []


@pytest.fixture
def broker(tmp_path):
    return PaperBroker(tmp_path / "stocks.json", STOCKS.limits)


def tools_by_name(data, broker, approve=lambda p, r: True):
    return {t.name: t for t in build_tools(STOCKS, data, broker, approve)}


# ---------------------------------------------------------------- symbols

@pytest.mark.parametrize("raw,expected", [("aapl", "AAPL"), ("BRK.B", "BRK-B"), (" spy ", "SPY")])
def test_normalize_stock_ok(raw, expected):
    assert normalize_stock(raw) == expected


@pytest.mark.parametrize("raw", ["BTC-USD", "EURUSD=X", "^GSPC", "ES=F", "VOD.L.X", "TOOLONGX", ""])
def test_normalize_stock_rejects_other_markets(raw):
    with pytest.raises(ValueError):
        normalize_stock(raw)


def test_only_stocks_enabled():
    assert get_market("stocks") is STOCKS
    with pytest.raises(ValueError, match="planned"):
        get_market("crypto")


# ---------------------------------------------------------------- indicators

def test_indicators():
    up = pd.Series(np.arange(1, 101, dtype=float))
    assert sma(up, 10) == pytest.approx(95.5)
    assert rsi(up) == 100.0
    s = technical_summary(FakeData({"X": 100}).history("X"), 252)
    assert s["last_close"] == pytest.approx(100)
    assert s["max_drawdown"] == pytest.approx(0)


# ---------------------------------------------------------------- broker

def test_buy_sell_roundtrip_and_persistence(broker, tmp_path):
    lim = broker.limits
    plan = broker.check_order("AAPL", "buy", 100, 100.0, {})
    assert plan["fill_price"] == pytest.approx(100 * (1 + lim.slippage_bps / 1e4))
    broker.execute(plan, "test")
    snap = broker.snapshot({"AAPL": 110.0})
    assert snap["positions"][0]["quantity"] == 100
    assert snap["equity_usd"] == pytest.approx(lim.starting_cash - plan["fill_price"] * 100 + 11_000, abs=0.01)

    reloaded = PaperBroker(tmp_path / "stocks.json", STOCKS.limits)
    assert reloaded.state.positions["AAPL"]["quantity"] == 100

    sell = reloaded.check_order("AAPL", "sell", 100, 110.0, {"AAPL": 110.0})
    reloaded.execute(sell, "take profit")
    assert "AAPL" not in reloaded.state.positions
    assert reloaded.state.realized_pnl == pytest.approx(100 * (sell["fill_price"] - plan["fill_price"]))
    assert reloaded.state.cash == pytest.approx(lim.starting_cash + reloaded.state.realized_pnl)


@pytest.mark.parametrize("side,qty,price,msg", [
    ("buy", 300, 100.0, "max_order_usd"),        # $30k > $25k cap
    ("buy", 10.5, 100.0, "multiple of 1"),       # whole shares only
    ("sell", 1, 100.0, "short selling"),         # no shorts
    ("hold", 1, 100.0, "side"),
    ("buy", 0, 100.0, "positive"),
])
def test_risk_rejections(broker, side, qty, price, msg):
    with pytest.raises(OrderRejected, match=msg):
        broker.check_order("AAPL", side, qty, price, {})


def test_position_concentration_limit(broker):
    # 20% of $100k equity = $20k; two $12k buys -> second breaches the limit.
    broker.execute(broker.check_order("MSFT", "buy", 120, 100.0, {}), "first")
    with pytest.raises(OrderRejected, match="limit is 20%"):
        broker.check_order("MSFT", "buy", 120, 100.0, {"MSFT": 100.0})


def test_insufficient_cash(tmp_path):
    b = PaperBroker(tmp_path / "a.json", RiskLimits(starting_cash=1_000, max_position_pct=1.0))
    with pytest.raises(OrderRejected, match="insufficient cash"):
        b.check_order("AAPL", "buy", 20, 100.0, {})


def test_reducing_order_allowed_even_if_over_limit(broker):
    broker.execute(broker.check_order("NVDA", "buy", 190, 100.0, {}), "x")
    # price doubles -> position is now far above 20%, but selling must still be allowed
    plan = broker.check_order("NVDA", "sell", 50, 200.0, {"NVDA": 200.0})
    assert plan["position_after"] == 140


# ---------------------------------------------------------------- tools

def test_tools_end_to_end(broker):
    data = FakeData({"AAPL": 100.0})
    t = tools_by_name(data, broker)
    assert json.loads(t["get_quote"].call({"symbols": ["aapl"]}))[0]["last_price"] == 100.0
    assert json.loads(t["get_technicals"].call({"symbol": "AAPL"}))["rsi14"] is not None
    fundamentals = json.loads(t["get_fundamentals"].call({"symbol": "AAPL"}))
    assert "trailingPE" not in fundamentals and fundamentals["forwardPE"] == 20.0  # NaN dropped
    assert "error" in json.loads(t["get_technicals"].call({"symbol": "BTC-USD"}))

    preview = json.loads(t["preview_order"].call({"symbol": "AAPL", "side": "buy", "quantity": 50}))
    assert preview["position_after"] == 50 and not broker.state.positions  # preview doesn't trade

    filled = json.loads(t["place_order"].call({"symbol": "AAPL", "side": "buy", "quantity": 50, "rationale": "test"}))
    assert filled["status"].startswith("filled")
    snap = json.loads(t["get_portfolio"].call({}))
    assert snap["positions"][0]["symbol"] == "AAPL"
    assert json.loads(t["get_trade_history"].call({}))[0]["rationale"] == "test"


def test_declined_order_not_executed(broker):
    t = tools_by_name(FakeData({"AAPL": 100.0}), broker, approve=lambda p, r: False)
    out = json.loads(t["place_order"].call({"symbol": "AAPL", "side": "buy", "quantity": 5, "rationale": "x"}))
    assert "declined" in out["error"] and not broker.state.positions
