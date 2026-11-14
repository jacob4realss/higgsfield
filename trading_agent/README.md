# Trading agents (one per market)

Claude-powered research + **paper-trading** agents. Each market gets its own independent agent with
its own instructions, symbol rules, risk limits and paper account. Shared plumbing: market data
(Yahoo Finance via `yfinance`), indicators, the paper broker and the agent loop.

| Agent    | Status  | Account            |
|----------|---------|--------------------|
| `stocks` | ✅ ready | $100k cash, long-only |
| `crypto` | planned (profile drafted in `markets.py`) | |
| `forex`  | planned (profile + margin accounting drafted) | |

> Paper trading only: orders are simulated against live (≈15-min delayed) prices. Nothing here
> places real trades, and it is not financial advice.

## Setup

```bash
pip install -r trading_agent/requirements.txt
export ANTHROPIC_API_KEY=...        # or `ant auth login`
```

## Use the stocks agent

```bash
python -m trading_agent stocks                                   # interactive chat
python -m trading_agent stocks "Compare NVDA and AMD for a 1-year hold"
python -m trading_agent stocks "Build me a 5-stock dividend portfolio with $50k"
python -m trading_agent stocks --portfolio                       # show paper account
python -m trading_agent stocks --reset                           # back to starting cash
```

Every order is shown for approval (`y/N`) before it fills; `--auto-approve` skips that for unattended
runs. `--no-web` disables web search.

## What the stocks agent can do

| Tool | Purpose |
|------|---------|
| `get_quote` | price, change, day/52-week range, volume, market cap |
| `get_technicals` | SMA 20/50/200, RSI, MACD, ATR, volatility, drawdown, 1m/3m return |
| `get_price_history` | OHLCV bars |
| `get_fundamentals` | valuation, margins, growth, balance sheet, analyst targets, next earnings |
| `get_news` | recent headlines |
| `web_search` | anything newer / broader (Anthropic server tool) |
| `get_portfolio`, `get_trade_history` | the paper account |
| `preview_order`, `place_order` | risk-checked paper market orders |

**Risk limits (stocks):** max 20% of equity per position, max $25k per order, no margin, no
shorting, whole shares, 5 bps slippage. Edit `STOCKS.limits` in `markets.py` to change them.

## Configuration

| Env var | Default |
|---------|---------|
| `TRADING_AGENT_MODEL` | `claude-opus-5-5` |
| `TRADING_AGENT_EFFORT` | `high` |
| `TRADING_AGENT_HOME` | `./paper_accounts` (one JSON file per market) |

Requests opt into server-side refusal fallbacks (`fallbacks: "default"`).

## Layout

```
markets.py      MarketProfile per market (prompt, symbol rules, limits, accounting)
agent.py        system prompt + Claude tool-runner loop (MarketAgent)
tools.py        tools bound to one market's data + account
portfolio.py    PaperBroker (cash or margin settlement, risk checks, JSON persistence)
market_data.py  yfinance wrapper + indicator math
cli.py          `python -m trading_agent <market>`
tests/          offline tests (python -m pytest trading_agent/tests)
```

Adding a market = new `MarketProfile` + any market-specific tools in `tools.py`, then register it
in `MARKETS`.
