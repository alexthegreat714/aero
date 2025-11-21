"""
Web module for Aero Agent.

Provides web search, scraping, and content parsing capabilities.
"""

from aero.web.search import WebSearcher, SearchResult
from aero.web.scraper import WebScraper, ScrapedPage
from aero.web.parser import ContentParser

__all__ = [
    "WebSearcher",
    "SearchResult",
    "WebScraper",
    "ScrapedPage",
    "ContentParser",
]
