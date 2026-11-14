"""Paper-trading broker: cash + positions + trade log, persisted as JSON.

All account values are in USD. A `valuer(symbol, quantity, price) -> usd notional`
callable converts a position to USD, so markets quoted in other currencies
(e.g. USD/JPY) still roll up into one USD account.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

Valuer = Callable[[str, float, float], float]


def _linear_valuer(symbol: str, quantity: float, price: float) -> float:
    return quantity * price


@dataclass(frozen=True)
class RiskLimits:
    starting_cash: float = 100_000.0
    max_position_pct: float = 0.20      # |position value| / equity after the trade
    max_order_usd: float = 25_000.0     # notional of a single order
    max_gross_exposure: float = 1.0     # sum |position values| / equity (1.0 = no leverage)
    allow_short: bool = False
    quantity_step: float = 1.0          # smallest tradable increment (shares, coins, currency units)
    slippage_bps: float = 5.0           # applied against you on every fill
    commission_bps: float = 0.0         # fee as bps of notional


@dataclass
class Trade:
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    notional_usd: float
    fee_usd: float
    rationale: str


@dataclass
class AccountState:
    cash: float
    positions: dict[str, dict] = field(default_factory=dict)  # symbol -> {"quantity", "avg_price"}
    trades: list[dict] = field(default_factory=list)
    realized_pnl: float = 0.0
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OrderRejected(Exception):
    pass


class PaperBroker:
    """Paper account for one market.

    settlement="cash":   buying debits cash by the notional (stocks, spot crypto).
    settlement="margin": opening a position moves no cash; only realized P&L and fees
                         hit cash, and equity = cash + unrealized P&L (forex/CFD style).
    """

    def __init__(self, path: str | Path, limits: RiskLimits, valuer: Valuer = _linear_valuer,
                 settlement: str = "cash") -> None:
        if settlement not in ("cash", "margin"):
            raise ValueError("settlement must be 'cash' or 'margin'")
        self.path = Path(path)
        self.limits = limits
        self.valuer = valuer
        self.settlement = settlement
        self.state = self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> AccountState:
        if self.path.exists():
            return AccountState(**json.loads(self.path.read_text()))
        return AccountState(cash=self.limits.starting_cash)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(self.state), f, indent=2)
        os.replace(tmp, self.path)

    def reset(self) -> None:
        self.state = AccountState(cash=self.limits.starting_cash)
        self.save()

    # ------------------------------------------------------------ accounting core
    def _metrics(self, state: AccountState, prices: dict[str, float]) -> tuple[float, float, dict[str, dict]]:
        """(equity, gross exposure, per-position values) for a state at `prices`."""
        equity, gross, per = state.cash, 0.0, {}
        for sym, pos in state.positions.items():
            qty, avg, px = pos["quantity"], pos["avg_price"], prices[sym]
            notional = self.valuer(sym, qty, px)
            unrealized = notional - self.valuer(sym, qty, avg)
            equity += notional if self.settlement == "cash" else unrealized
            gross += abs(notional)
            per[sym] = {"notional": notional, "unrealized": unrealized}
        return equity, gross, per

    def _apply(self, state: AccountState, sym: str, signed: float, fill: float, fee: float) -> None:
        pos = state.positions.get(sym, {"quantity": 0.0, "avg_price": 0.0})
        cur, avg = pos["quantity"], pos["avg_price"]
        realized = 0.0
        if cur != 0 and (cur > 0) != (signed > 0):  # closes some existing exposure
            closed = math.copysign(min(abs(cur), abs(signed)), cur)
            realized = self.valuer(sym, closed, fill) - self.valuer(sym, closed, avg)
        new_qty = cur + signed
        if abs(new_qty) < 1e-12:
            state.positions.pop(sym, None)
        else:
            if cur == 0 or (cur > 0) != (new_qty > 0):     # opened or flipped
                new_avg = fill
            elif (cur > 0) == (signed > 0):                # added
                new_avg = (abs(cur) * avg + abs(signed) * fill) / abs(new_qty)
            else:                                          # partially reduced
                new_avg = avg
            state.positions[sym] = {"quantity": new_qty, "avg_price": new_avg}
        if self.settlement == "cash":
            state.cash -= self.valuer(sym, signed, fill) + fee
        else:
            state.cash += realized - fee
        state.realized_pnl += realized - fee

    # ------------------------------------------------------------ public API
    def snapshot(self, prices: dict[str, float]) -> dict:
        """Mark the account to market. `prices` must cover every open position."""
        equity, gross, per = self._metrics(self.state, prices)
        rows = []
        for sym, pos in sorted(self.state.positions.items()):
            qty, avg, px = pos["quantity"], pos["avg_price"], prices[sym]
            rows.append({
                "symbol": sym, "quantity": qty, "side": "long" if qty > 0 else "short",
                "avg_price": avg, "last_price": px,
                "notional_usd": round(per[sym]["notional"], 2),
                "unrealized_pnl_usd": round(per[sym]["unrealized"], 2),
                "unrealized_pnl_pct": round((px / avg - 1) * 100 * (1 if qty > 0 else -1), 2) if avg else None,
                "weight_pct_of_equity": round(abs(per[sym]["notional"]) / equity * 100, 2) if equity > 0 else None,
            })
        return {
            "settlement": self.settlement,
            "cash_usd": round(self.state.cash, 2),
            "equity_usd": round(equity, 2),
            "gross_exposure_x": round(gross / equity, 3) if equity > 0 else None,
            "realized_pnl_usd": round(self.state.realized_pnl, 2),
            "total_return_pct": round((equity / self.limits.starting_cash - 1) * 100, 2),
            "positions": rows,
            "n_trades": len(self.state.trades),
            "risk_limits": asdict(self.limits),
        }

    def check_order(self, symbol: str, side: str, quantity: float, price: float, prices: dict[str, float]) -> dict:
        """Validate an order against the risk limits without executing it. Returns the planned fill."""
        lim = self.limits
        side = side.lower()
        if side not in ("buy", "sell"):
            raise OrderRejected("side must be 'buy' or 'sell'")
        if not (isinstance(quantity, (int, float)) and math.isfinite(quantity) and quantity > 0):
            raise OrderRejected("quantity must be a positive number")
        steps = quantity / lim.quantity_step
        if abs(steps - round(steps)) > 1e-6:
            raise OrderRejected(f"quantity must be a multiple of {lim.quantity_step}")
        if not (math.isfinite(price) and price > 0):
            raise OrderRejected("no valid price available")

        slip = lim.slippage_bps / 10_000
        fill = price * (1 + slip) if side == "buy" else price * (1 - slip)
        signed = quantity if side == "buy" else -quantity
        notional = abs(self.valuer(symbol, quantity, fill))
        fee = notional * lim.commission_bps / 10_000
        if notional > lim.max_order_usd:
            raise OrderRejected(f"order notional ${notional:,.0f} exceeds max_order_usd ${lim.max_order_usd:,.0f}")

        current = self.state.positions.get(symbol, {"quantity": 0.0})["quantity"]
        new_qty = current + signed
        if new_qty < -1e-12 and not lim.allow_short:
            raise OrderRejected(f"short selling is disabled for this market (holding {current}, selling {quantity})")

        after = AccountState(**json.loads(json.dumps(asdict(self.state))))
        self._apply(after, symbol, signed, fill, fee)
        marks = {**prices, symbol: price}
        equity_after, gross_after, per_after = self._metrics(after, marks)
        if self.settlement == "cash" and after.cash < -1e-6:
            raise OrderRejected(f"insufficient cash: order needs ${notional + fee:,.2f}, have ${self.state.cash:,.2f}")
        if equity_after <= 0:
            raise OrderRejected("order would leave the account with non-positive equity")
        pos_pct = abs(per_after.get(symbol, {"notional": 0.0})["notional"]) / equity_after
        if abs(new_qty) > abs(current) + 1e-12:  # only exposure-increasing orders are size-limited
            if pos_pct > lim.max_position_pct + 1e-9:
                raise OrderRejected(f"position would be {pos_pct:.0%} of equity; limit is {lim.max_position_pct:.0%}")
            if gross_after / equity_after > lim.max_gross_exposure + 1e-9:
                raise OrderRejected(f"gross exposure would be {gross_after / equity_after:.2f}x; limit is {lim.max_gross_exposure:.2f}x")
        return {
            "symbol": symbol, "side": side, "quantity": quantity, "reference_price": price,
            "fill_price": fill, "notional_usd": round(notional, 2), "fee_usd": round(fee, 2),
            "position_after": new_qty, "position_pct_of_equity_after": round(pos_pct * 100, 2),
            "gross_exposure_after_x": round(gross_after / equity_after, 3),
            "equity_after_usd": round(equity_after, 2),
        }

    def execute(self, plan: dict, rationale: str) -> Trade:
        signed = plan["quantity"] if plan["side"] == "buy" else -plan["quantity"]
        self._apply(self.state, plan["symbol"], signed, plan["fill_price"], plan["fee_usd"])
        trade = Trade(
            timestamp=datetime.now(timezone.utc).isoformat(), symbol=plan["symbol"], side=plan["side"],
            quantity=plan["quantity"], price=plan["fill_price"], notional_usd=plan["notional_usd"],
            fee_usd=plan["fee_usd"], rationale=rationale,
        )
        self.state.trades.append(asdict(trade))
        self.save()
        return trade
