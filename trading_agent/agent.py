"""The agent loop: one Claude conversation bound to one market's tools and paper account."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import anthropic

from .market_data import MarketData
from .markets import MarketProfile
from .portfolio import PaperBroker
from .tools import Approver, build_tools

MODEL = os.environ.get("TRADING_AGENT_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("TRADING_AGENT_EFFORT", "high")
ACCOUNTS_DIR = Path(os.environ.get("TRADING_AGENT_HOME", Path.cwd() / "paper_accounts"))

BASE_PROMPT = """\
You are the {title} agent: a disciplined research analyst and portfolio manager running a PAPER
(simulated) trading account. No real money is involved, but behave as if it were real.

How you work:
- Ground every claim in data from your tools or web search. Never invent prices, figures or news.
  If a tool fails or data is missing, say so instead of guessing.
- Before recommending a trade, check the current portfolio, the quote, technicals and the relevant
  fundamentals/news. For news or events after your training data, use web search.
- Present both the bull and the bear case, the main risks, and what would make you wrong.
- Every trade idea needs: size (and why), entry, a stop or invalidation level, a target or review
  condition, and the effect on portfolio concentration. Use preview_order to size against limits.
- Only call place_order when the user has asked you to trade or has agreed to a specific proposal.
  The user approves each order; a declined order is final for that request.
- Respect the risk limits; if the user asks for something they forbid, explain the limit instead of
  working around it (e.g. by splitting orders).
- Be concise: lead with the conclusion, then the supporting numbers. Use short tables for comparisons.
- You only handle this market. If asked about another asset class, say that a separate agent covers it.
- You are not a licensed financial adviser; mention this once when giving recommendations, not on
  every message.

{instructions}

Current date: {date} (UTC).
"""


def system_prompt(profile: MarketProfile) -> str:
    return BASE_PROMPT.format(
        title=profile.title,
        instructions=profile.instructions,
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )


def make_broker(profile: MarketProfile, data: MarketData, accounts_dir: Path = ACCOUNTS_DIR) -> PaperBroker:
    return PaperBroker(
        accounts_dir / f"{profile.key}.json",
        profile.limits,
        valuer=profile.make_valuer(data),
        settlement=profile.settlement,
    )


class MarketAgent:
    """A multi-turn conversation with one market agent."""

    def __init__(
        self,
        profile: MarketProfile,
        approve: Approver,
        on_text: Callable[[str], None] = print,
        on_tool: Callable[[str, dict], None] | None = None,
        client: anthropic.Anthropic | None = None,
        data: MarketData | None = None,
        broker: PaperBroker | None = None,
        web_search: bool = True,
    ) -> None:
        self.profile = profile
        self.client = client or anthropic.Anthropic()
        self.data = data or MarketData()
        self.broker = broker or make_broker(profile, self.data)
        self.on_text, self.on_tool = on_text, on_tool
        self.tools: list = build_tools(profile, self.data, self.broker, approve)
        if web_search:
            self.tools.append({"type": "web_search_20260209", "name": "web_search", "max_uses": 5})
        self.system = system_prompt(profile)
        self.messages: list = []

    def ask(self, user_message: str, max_iterations: int = 30) -> str:
        """Send one user message; run tools until Claude finishes. Returns the final text."""
        self.messages.append({"role": "user", "content": user_message})
        runner = self.client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=16000,
            system=self.system,
            tools=self.tools,
            messages=list(self.messages),
            output_config={"effort": EFFORT},
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            max_iterations=max_iterations,
        )
        final_text = ""
        last = None
        for message in runner:
            last = message
            # Keep our own copy of the conversation for the next turn (full content blocks,
            # including thinking/fallback/server-tool blocks, appended unchanged).
            self.messages.append({"role": "assistant", "content": message.content})
            texts = []
            for block in message.content:
                if block.type == "text" and block.text.strip():
                    texts.append(block.text)
                elif block.type in ("tool_use", "server_tool_use") and self.on_tool:
                    self.on_tool(block.name, dict(block.input or {}))
            if texts:
                final_text = "\n".join(texts)
                self.on_text(final_text)
            tool_results = runner.generate_tool_call_response()
            if tool_results is not None:
                self.messages.append(tool_results)

        if last is not None and last.stop_reason == "refusal":
            self.on_text("[The model declined this request.]")
        elif last is not None and last.stop_reason == "max_tokens":
            self.on_text("[Response was cut off at the token limit.]")
        elif last is not None and last.stop_reason == "tool_use":
            self.on_text(f"[Stopped after {max_iterations} steps; ask me to continue.]")
        return final_text
