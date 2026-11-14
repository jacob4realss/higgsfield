"""Claude-powered trading / investing agents - one independent agent per market.

Each market (stocks, crypto, forex) has its own system prompt, symbol rules,
risk limits and paper-trading account. They share the market-data layer,
indicator math, paper broker and agent loop.
"""

from .markets import MARKETS, MarketProfile, get_market
from .portfolio import PaperBroker, RiskLimits

__all__ = ["MARKETS", "MarketProfile", "get_market", "PaperBroker", "RiskLimits"]
