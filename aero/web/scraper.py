"""
Web scraping utilities for Aero Agent.

Provides tools for fetching and extracting web content.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    """Container for scraped web page content."""

    url: str
    status_code: int
    content: str
    html: str
    title: Optional[str]
    links: list[str]
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "status_code": self.status_code,
            "title": self.title,
            "content_length": len(self.content),
            "links_count": len(self.links),
            "timestamp": self.timestamp.isoformat(),
        }


class WebScraper:
    """
    Web scraper for Aero Agent.

    Provides:
    - Page fetching
    - Content extraction
    - Link extraction
    - Rate limiting

    Example:
        scraper = WebScraper()
        page = scraper.fetch("https://example.com")
        print(page.title)
        print(page.content)
    """

    def __init__(
        self,
        user_agent: str = "Aero-Agent/0.1.0",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize the scraper.

        Args:
            user_agent: User agent string
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
        })

        logger.info("WebScraper initialized")

    def fetch(self, url: str) -> Optional[ScrapedPage]:
        """
        Fetch a web page.

        Args:
            url: URL to fetch

        Returns:
            ScrapedPage or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                response = self._session.get(url, timeout=self.timeout)

                if response.status_code == 200:
                    # Parse content
                    html = response.text
                    content = self._extract_text(html)
                    title = self._extract_title(html)
                    links = self._extract_links(html, url)

                    return ScrapedPage(
                        url=url,
                        status_code=response.status_code,
                        content=content,
                        html=html,
                        title=title,
                        links=links,
                    )
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    return ScrapedPage(
                        url=url,
                        status_code=response.status_code,
                        content="",
                        html="",
                        title=None,
                        links=[],
                    )

            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")

        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None

    def _extract_text(self, html: str) -> str:
        """
        Extract text content from HTML.

        Simple implementation - consider using BeautifulSoup for production.
        """
        import re

        # Remove script and style elements
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def _extract_title(self, html: str) -> Optional[str]:
        """Extract page title from HTML."""
        import re

        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract links from HTML."""
        import re
        from urllib.parse import urljoin

        links = []
        pattern = r'href=["\']([^"\']+)["\']'

        for match in re.finditer(pattern, html, re.IGNORECASE):
            link = match.group(1)

            # Skip anchors and javascript
            if link.startswith('#') or link.startswith('javascript:'):
                continue

            # Make absolute URL
            absolute_url = urljoin(base_url, link)
            links.append(absolute_url)

        return list(set(links))

    def close(self) -> None:
        """Close the session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
