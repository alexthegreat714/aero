"""
Content parsing utilities for Aero Agent.

Provides tools for parsing and structuring web content.
"""

import logging
import re
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedContent:
    """Container for parsed content."""

    title: Optional[str]
    text: str
    headings: list[str]
    paragraphs: list[str]
    lists: list[list[str]]
    tables: list[list[list[str]]]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "text_length": len(self.text),
            "headings_count": len(self.headings),
            "paragraphs_count": len(self.paragraphs),
            "lists_count": len(self.lists),
            "tables_count": len(self.tables),
        }


class ContentParser:
    """
    Content parser for web and document content.

    Provides:
    - HTML parsing
    - Text extraction
    - Structure detection
    - Table extraction

    Example:
        parser = ContentParser()
        content = parser.parse_html(html)
        print(content.title)
        for heading in content.headings:
            print(f"  - {heading}")
    """

    def __init__(self):
        """Initialize the parser."""
        self._logger = logging.getLogger("aero.parser")

    def parse_html(self, html: str) -> ParsedContent:
        """
        Parse HTML content.

        Args:
            html: HTML string

        Returns:
            ParsedContent with extracted elements
        """
        title = self._extract_title(html)
        text = self._extract_text(html)
        headings = self._extract_headings(html)
        paragraphs = self._extract_paragraphs(html)
        lists = self._extract_lists(html)
        tables = self._extract_tables(html)

        return ParsedContent(
            title=title,
            text=text,
            headings=headings,
            paragraphs=paragraphs,
            lists=lists,
            tables=tables,
        )

    def _extract_title(self, html: str) -> Optional[str]:
        """Extract title from HTML."""
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return self._clean_text(match.group(1))
        return None

    def _extract_text(self, html: str) -> str:
        """Extract plain text from HTML."""
        # Remove script and style
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', text)

        return self._clean_text(text)

    def _extract_headings(self, html: str) -> list[str]:
        """Extract headings (h1-h6) from HTML."""
        headings = []

        for level in range(1, 7):
            pattern = f'<h{level}[^>]*>(.*?)</h{level}>'
            for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
                text = self._clean_text(match.group(1))
                if text:
                    headings.append(text)

        return headings

    def _extract_paragraphs(self, html: str) -> list[str]:
        """Extract paragraphs from HTML."""
        paragraphs = []

        pattern = r'<p[^>]*>(.*?)</p>'
        for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
            text = self._clean_text(match.group(1))
            if text and len(text) > 20:  # Filter very short paragraphs
                paragraphs.append(text)

        return paragraphs

    def _extract_lists(self, html: str) -> list[list[str]]:
        """Extract lists (ul, ol) from HTML."""
        lists = []

        # Match ul and ol elements
        list_pattern = r'<[ou]l[^>]*>(.*?)</[ou]l>'
        item_pattern = r'<li[^>]*>(.*?)</li>'

        for list_match in re.finditer(list_pattern, html, re.IGNORECASE | re.DOTALL):
            list_content = list_match.group(1)
            items = []

            for item_match in re.finditer(item_pattern, list_content, re.IGNORECASE | re.DOTALL):
                text = self._clean_text(item_match.group(1))
                if text:
                    items.append(text)

            if items:
                lists.append(items)

        return lists

    def _extract_tables(self, html: str) -> list[list[list[str]]]:
        """Extract tables from HTML."""
        tables = []

        table_pattern = r'<table[^>]*>(.*?)</table>'
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'

        for table_match in re.finditer(table_pattern, html, re.IGNORECASE | re.DOTALL):
            table_content = table_match.group(1)
            rows = []

            for row_match in re.finditer(row_pattern, table_content, re.IGNORECASE | re.DOTALL):
                row_content = row_match.group(1)
                cells = []

                for cell_match in re.finditer(cell_pattern, row_content, re.IGNORECASE | re.DOTALL):
                    text = self._clean_text(cell_match.group(1))
                    cells.append(text)

                if cells:
                    rows.append(cells)

            if rows:
                tables.append(rows)

        return tables

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Decode common entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_metadata(self, html: str) -> dict:
        """
        Extract metadata from HTML.

        Args:
            html: HTML string

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        # Meta tags
        meta_pattern = r'<meta\s+(?:name|property)=["\']([^"\']+)["\']\s+content=["\']([^"\']+)["\']'
        for match in re.finditer(meta_pattern, html, re.IGNORECASE):
            name = match.group(1).lower()
            content = match.group(2)
            metadata[name] = content

        return metadata

    def summarize(self, text: str, max_length: int = 500) -> str:
        """
        Create a simple summary of text.

        Args:
            text: Input text
            max_length: Maximum summary length

        Returns:
            Summarized text
        """
        # Simple extractive summary - first few sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        summary = []
        length = 0

        for sentence in sentences:
            if length + len(sentence) > max_length:
                break
            summary.append(sentence)
            length += len(sentence) + 2  # Account for period and space

        return '. '.join(summary) + '.' if summary else text[:max_length]
