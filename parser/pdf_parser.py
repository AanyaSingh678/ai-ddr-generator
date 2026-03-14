"""
parser/pdf_parser.py

Handles PDF text and page-level metadata extraction for the DDR pipeline.
Each page is extracted individually so downstream modules can correlate
images and observations by page number.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path


import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """
    Represents extracted content from a single PDF page.

    Attributes:
        page_number:  1-indexed page number (matches human-readable references).
        text:         Full plain text content of the page.
        headings:     List of detected heading strings based on font size.
        word_count:   Automatically computed from text on initialization.
    """

    page_number: int
    text: str
    headings: list[str] = field(default_factory=list)
    word_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())

    def is_empty(self) -> bool:
        """Returns True if the page contains no meaningful text."""
        return self.text.strip() == ""

    def __repr__(self) -> str:
        return (
            f"PageContent(page={self.page_number}, "
            f"words={self.word_count}, "
            f"headings={len(self.headings)})"
        )


@dataclass
class ParsedDocument:
    """
    Represents a fully parsed PDF document.

    Attributes:
        source_path:  Path to the original PDF file.
        pages:        List of PageContent objects, one per page.
        total_pages:  Total number of pages in the document.
    """

    source_path: Path
    pages: list[PageContent] = field(default_factory=list)
    total_pages: int = 0

    @property
    def full_text(self) -> str:
        """Concatenated text from all pages, separated by newlines."""
        return "\n".join(page.text for page in self.pages)

    @property
    def non_empty_pages(self) -> list[PageContent]:
        """Returns only pages that contain actual text content."""
        return [page for page in self.pages if not page.is_empty()]

    def get_page(self, page_number: int) -> PageContent | None:
        """
        Retrieves a page by its 1-indexed page number.

        Args:
            page_number: The 1-indexed page number to look up.

        Returns:
            PageContent if found, None otherwise.
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def __repr__(self) -> str:
        return (
            f"ParsedDocument("
            f"source='{self.source_path.name}', "
            f"pages={self.total_pages}, "
            f"non_empty={len(self.non_empty_pages)})"
        )


class PDFParser:
    """
    Parses a PDF file and extracts structured page-level text content.

    Designed to work with UrbanRoof inspection and thermal reports.
    Each page is extracted individually with heading detection so that
    downstream modules (image mapper, observation extractor) can use
    page numbers and section headings to map findings correctly.

    Usage:
        parser = PDFParser("data/inspection_report.pdf")
        document = parser.parse()

        print(document.full_text)
        print(document.get_page(5))
    """

    # Minimum font size to treat a text span as a heading.
    # UrbanRoof reports use larger bold text for section headers.
    HEADING_FONT_SIZE_THRESHOLD: float = 11.0

    def __init__(self, pdf_path: str) -> None:
        self.pdf_path = Path(pdf_path)
        self._validate_path()

    def _validate_path(self) -> None:
        """
        Validates that the file exists and is a PDF.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            ValueError: If the file does not have a .pdf extension.
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: '{self.pdf_path}'. "
                f"Please ensure the file is placed in the /data folder."
            )
        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a .pdf file, got: '{self.pdf_path.suffix}'"
            )

    def _extract_headings(self, page: fitz.Page) -> list[str]:
        """
        Extracts likely section headings from a page using font size.

        Text spans with font size at or above HEADING_FONT_SIZE_THRESHOLD
        are treated as headings. This is reliable for UrbanRoof reports
        where section titles use noticeably larger fonts.

        Args:
            page: A PyMuPDF page object.

        Returns:
            List of heading text strings found on the page.
        """
        headings: list[str] = []

        try:
            blocks = page.get_text("dict")["blocks"]
        except Exception as e:
            logger.warning("Could not parse text dict on page: %s", e)
            return headings

        for block in blocks:
            if block.get("type") != 0:  # type 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_size: float = span.get("size", 0.0)
                    text: str = span.get("text", "").strip()
                    if font_size >= self.HEADING_FONT_SIZE_THRESHOLD and text:
                        headings.append(text)

        return headings

    def _parse_page(self, page: fitz.Page, page_number: int) -> PageContent:
        """
        Extracts text and headings from a single PDF page.

        Args:
            page:        A PyMuPDF page object.
            page_number: 1-indexed page number.

        Returns:
            A populated PageContent dataclass.
        """
        text = page.get_text()
        headings = self._extract_headings(page)

        logger.debug(
            "Page %d: %d characters, %d headings.",
            page_number,
            len(text),
            len(headings),
        )

        return PageContent(
            page_number=page_number,
            text=text,
            headings=headings,
        )

    def parse(self) -> ParsedDocument:
        """
        Parses the entire PDF and returns a structured ParsedDocument.

        Opens the document using a context manager to guarantee cleanup
        even if an error occurs during parsing. Each page is processed
        individually so page-level metadata is preserved.

        Returns:
            ParsedDocument containing all page content and metadata.

        Raises:
            RuntimeError: If the PDF is encrypted or the file is corrupted.
        """
        logger.info("Starting PDF parsing: '%s'", self.pdf_path.name)

        parsed_document = ParsedDocument(source_path=self.pdf_path)

        try:
            with fitz.open(self.pdf_path) as document:

                if document.is_encrypted:
                    raise RuntimeError(
                        f"PDF is encrypted and cannot be parsed: '{self.pdf_path.name}'"
                    )

                parsed_document.total_pages = document.page_count

                for index, page in enumerate(document):
                    page_number = index + 1  # convert to 1-indexed
                    page_content = self._parse_page(page, page_number)
                    parsed_document.pages.append(page_content)

        except fitz.FileDataError as e:
            raise RuntimeError(
                f"Could not open PDF '{self.pdf_path.name}'. "
                f"File may be corrupted. Details: {e}"
            ) from e

        logger.info(
            "Parsing complete: %d pages, %d non-empty.",
            parsed_document.total_pages,
            len(parsed_document.non_empty_pages),
        )

        return parsed_document

    def __repr__(self) -> str:
        return f"PDFParser(pdf_path='{self.pdf_path}')"