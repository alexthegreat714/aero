"""
Text preprocessing for Aero Agent RAG system.

Provides utilities for cleaning and normalizing text content,
with special handling for scientific documents.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedText:
    """Container for preprocessed text with extracted elements."""

    text: str
    equations: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class TextPreprocessor:
    """
    Text preprocessor for RAG documents.

    Provides:
    - Whitespace normalization
    - Boilerplate removal
    - Equation detection and extraction
    - Table detection
    - Hyphenation fixes

    Example:
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("Some text with $E=mc^2$ equation")
        print(result.text)
        print(result.equations)
    """

    def __init__(
        self,
        remove_boilerplate: bool = True,
        extract_equations: bool = True,
        fix_hyphens: bool = True,
        normalize_whitespace: bool = True,
    ):
        """
        Initialize the preprocessor.

        Args:
            remove_boilerplate: Remove common boilerplate text
            extract_equations: Extract LaTeX equations
            fix_hyphens: Fix broken hyphenated words
            normalize_whitespace: Normalize whitespace characters
        """
        self.remove_boilerplate = remove_boilerplate
        self.extract_equations = extract_equations
        self.fix_hyphens = fix_hyphens
        self.normalize_whitespace = normalize_whitespace

        # Common boilerplate patterns
        self._boilerplate_patterns = [
            r"Page \d+ of \d+",
            r"^\s*\d+\s*$",  # Page numbers only
            r"Copyright.*\d{4}",
            r"All rights reserved",
            r"Confidential",
            r"DRAFT",
            r"www\.\S+",
            r"http[s]?://\S+",
        ]

        # Equation patterns
        self._equation_patterns = [
            (r"\$\$(.+?)\$\$", "display"),  # Display math $$...$$
            (r"\$(.+?)\$", "inline"),  # Inline math $...$
            (r"\\begin\{equation\}(.+?)\\end\{equation\}", "equation"),
            (r"\\begin\{align\}(.+?)\\end\{align\}", "align"),
            (r"\\begin\{eqnarray\}(.+?)\\end\{eqnarray\}", "eqnarray"),
            (r"\\\[(.+?)\\\]", "display_bracket"),  # \[...\]
            (r"\\\((.+?)\\\)", "inline_paren"),  # \(...\)
        ]

        # Table markers
        self._table_patterns = [
            r"\\begin\{tabular\}(.+?)\\end\{tabular\}",
            r"\\begin\{table\}(.+?)\\end\{table\}",
            r"\|[^\n]+\|(?:\n\|[^\n]+\|)+",  # Markdown tables
        ]

    def preprocess(self, text: str) -> PreprocessedText:
        """
        Preprocess text content.

        Args:
            text: Raw text content

        Returns:
            PreprocessedText with cleaned content and extracted elements
        """
        result = PreprocessedText(text=text)

        # Extract equations before cleaning
        if self.extract_equations:
            text, equations = self._extract_equations(text)
            result.equations = equations

        # Extract tables
        text, tables = self._extract_tables(text)
        result.tables = tables

        # Fix hyphenation
        if self.fix_hyphens:
            text = self._fix_hyphenation(text)

        # Remove boilerplate
        if self.remove_boilerplate:
            text = self._remove_boilerplate(text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        result.text = text.strip()
        return result

    def _extract_equations(self, text: str) -> tuple[str, list[str]]:
        """Extract LaTeX equations from text."""
        equations = []
        equation_placeholder = " [EQUATION_{idx}] "

        for pattern, eq_type in self._equation_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                eq_text = match.strip() if isinstance(match, str) else match[0].strip()
                equations.append({
                    "text": eq_text,
                    "type": eq_type,
                    "index": len(equations),
                })

            # Replace with placeholder to preserve position info
            def replacer(m):
                idx = len(equations) - len(matches) + equations.index(next(
                    e for e in equations if e["text"] in m.group()
                ))
                return equation_placeholder.format(idx=idx)

            # Simple replacement for now
            text = re.sub(pattern, " [EQUATION] ", text, flags=re.DOTALL)

        return text, [eq["text"] for eq in equations]

    def _extract_tables(self, text: str) -> tuple[str, list[str]]:
        """Extract tables from text."""
        tables = []

        for pattern in self._table_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            tables.extend(matches)
            text = re.sub(pattern, " [TABLE] ", text, flags=re.DOTALL)

        return text, tables

    def _fix_hyphenation(self, text: str) -> str:
        """Fix broken hyphenated words across lines."""
        # Fix words split across lines with hyphen
        text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)

        # Fix soft hyphens
        text = text.replace("\u00ad", "")

        # Fix em-dash used as hyphen
        text = re.sub(r"(\w)—(\w)", r"\1-\2", text)

        return text

    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate text."""
        for pattern in self._boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove lines that are just punctuation or very short
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Keep lines with actual content
            if len(stripped) > 3 and not re.match(r"^[\W\d]+$", stripped):
                cleaned_lines.append(line)
            elif stripped == "":
                cleaned_lines.append("")  # Preserve paragraph breaks

        return "\n".join(cleaned_lines)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace various whitespace with regular space
        text = re.sub(r"[\t\r\f\v]+", " ", text)

        # Normalize multiple spaces to single space
        text = re.sub(r" +", " ", text)

        # Normalize multiple newlines to double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove spaces at start/end of lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text

    def detect_sections(self, text: str) -> list[dict]:
        """
        Detect section headers in text.

        Args:
            text: Text content

        Returns:
            List of detected sections with positions
        """
        sections = []

        # Common header patterns
        patterns = [
            # Numbered sections: "1. Introduction", "1.2.3 Methods"
            (r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]+)$", "numbered"),
            # Markdown headers: "# Title", "## Section"
            (r"^(#{1,6})\s+([^\n]+)$", "markdown"),
            # Uppercase headers: "INTRODUCTION"
            (r"^([A-Z][A-Z\s]{3,})$", "uppercase"),
            # Title case with colon: "Methods:"
            (r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):$", "title_colon"),
        ]

        lines = text.split("\n")
        for i, line in enumerate(lines):
            for pattern, section_type in patterns:
                match = re.match(pattern, line.strip(), re.MULTILINE)
                if match:
                    sections.append({
                        "line": i,
                        "text": line.strip(),
                        "type": section_type,
                        "level": self._get_section_level(match, section_type),
                    })
                    break

        return sections

    def _get_section_level(self, match: re.Match, section_type: str) -> int:
        """Determine section level from match."""
        if section_type == "numbered":
            # Count dots to determine level
            return match.group(1).count(".") + 1
        elif section_type == "markdown":
            return len(match.group(1))
        elif section_type == "uppercase":
            return 1
        elif section_type == "title_colon":
            return 2
        return 1

    def clean_for_embedding(self, text: str) -> str:
        """
        Clean text specifically for embedding generation.

        Removes elements that don't contribute to semantic meaning.

        Args:
            text: Text to clean

        Returns:
            Cleaned text suitable for embedding
        """
        # Remove equation placeholders
        text = re.sub(r"\[EQUATION(?:_\d+)?\]", "", text)
        text = re.sub(r"\[TABLE\]", "", text)

        # Remove citation markers
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)", "", text)

        # Remove figure/table references
        text = re.sub(r"(?:Figure|Fig\.|Table|Tab\.)\s*\d+", "", text, flags=re.IGNORECASE)

        # Normalize
        text = self._normalize_whitespace(text)

        return text.strip()


def preprocess_text(text: str, **kwargs) -> PreprocessedText:
    """
    Convenience function to preprocess text.

    Args:
        text: Text to preprocess
        **kwargs: Arguments passed to TextPreprocessor

    Returns:
        PreprocessedText result
    """
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.preprocess(text)


def clean_for_embedding(text: str) -> str:
    """
    Convenience function to clean text for embedding.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.clean_for_embedding(text)
