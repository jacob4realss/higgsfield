"""Command line: one agent per market.

    python -m trading_agent stocks                    # interactive chat
    python -m trading_agent stocks "Analyze NVDA"     # one-shot
    python -m trading_agent stocks --portfolio        # print the paper account
    python -m trading_agent stocks --reset            # reset the paper account
"""

from __future__ import annotations

import argparse
import json
import sys

from .markets import MARKETS, PLANNED_MARKETS, get_market

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _tty(code: str) -> str:
    return code if sys.stdout.isatty() else ""


def make_approver(auto: bool):
    def approve(plan: dict, rationale: str) -> bool:
        print(f"\n{_tty(BOLD)}PAPER ORDER{_tty(RESET)}  {plan['side'].upper()} {plan['quantity']:g} {plan['symbol']} "
              f"@ ~{plan['fill_price']:.4f}  (notional ${plan['notional_usd']:,.2f}, fee ${plan['fee_usd']:,.2f})")
        print(f"  position after: {plan['position_after']:g}  ({plan['position_pct_of_equity_after']}% of equity)")
        print(f"  rationale: {rationale}")
        if auto:
            print("  auto-approved (--auto-approve)")
            return True
        if not sys.stdin.isatty():
            print("  declined: no terminal to confirm (use --auto-approve for unattended runs)")
            return False
        return input("  Approve? [y/N] ").strip().lower() in ("y", "yes")
    return approve


def on_tool(name: str, args: dict) -> None:
    shown = json.dumps(args)
    print(f"{_tty(DIM)}  → {name}({shown[:160]}{'…' if len(shown) > 160 else ''}){_tty(RESET)}")


def main(argv: list[str] | None = None) -> None:
    choices = sorted(MARKETS) + sorted(PLANNED_MARKETS)
    ap = argparse.ArgumentParser(prog="trading_agent", description="Claude-powered paper-trading agents, one per market.")
    ap.add_argument("market", choices=choices, help="which market agent to run")
    ap.add_argument("prompt", nargs="*", help="one-shot request; omit for interactive chat")
    ap.add_argument("--auto-approve", action="store_true", help="execute paper orders without asking")
    ap.add_argument("--no-web", action="store_true", help="disable the web search tool")
    ap.add_argument("--portfolio", action="store_true", help="print the paper account and exit")
    ap.add_argument("--reset", action="store_true", help="reset the paper account to starting cash and exit")
    args = ap.parse_args(argv)

    try:
        profile = get_market(args.market)
    except ValueError as e:
        ap.error(str(e))

    from .agent import MarketAgent, make_broker
    from .market_data import MarketData

    if args.reset or args.portfolio:
        data = MarketData()
        broker = make_broker(profile, data)
        if args.reset:
            broker.reset()
            print(f"{profile.title} paper account reset to ${profile.limits.starting_cash:,.0f}.")
        else:
            prices = {s: data.last_price(s) for s in broker.state.positions}
            print(json.dumps(broker.snapshot(prices), indent=2))
        return

    agent = MarketAgent(profile, approve=make_approver(args.auto_approve), on_tool=on_tool,
                        web_search=not args.no_web)

    if args.prompt:
        agent.ask(" ".join(args.prompt))
        return

    print(f"{_tty(BOLD)}{profile.title} agent{_tty(RESET)} (paper trading; type 'quit' to exit)")
    print(f"Account file: {agent.broker.path}")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("quit", "exit", "q"):
            break
        if line:
            try:
                agent.ask(line)
            except KeyboardInterrupt:
                print("\n[interrupted]")
