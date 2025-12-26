"""
PDF text extraction using PyMuPDF (primary) and pdfplumber (tables).

Extraction Strategy:
1. PyMuPDF for fast text extraction (90% of pages)
2. pdfplumber for pages with tables/complex layouts
3. OCR fallback for scanned pages (requires tesseract)
"""

from pathlib import Path
from typing import Iterator, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re
import fitz  # PyMuPDF
import pdfplumber
from rich.progress import Progress, TaskID


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_num: int                    # 1-indexed page number
    text: str                        # Extracted text content
    content_type: str                # 'text', 'table', 'code', 'mixed', 'ocr'
    has_images: bool                 # Whether page contains images
    has_tables: bool                 # Whether page contains tables
    word_count: int                  # Number of words extracted
    tables: List[List[List[str]]] = field(default_factory=list)  # Extracted tables


@dataclass
class PDFMetadata:
    """PDF document metadata."""
    filename: str
    filepath: str
    page_count: int
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    toc: List[Tuple[int, str, int]] = field(default_factory=list)  # (level, title, page)


class PDFExtractor:
    """
    Two-pass PDF extraction strategy:
    1. Classify pages by content type
    2. Use appropriate extractor for each page type

    PyMuPDF (fitz) - Fast, handles most pages
    pdfplumber - Better for tables and complex layouts
    OCR - Fallback for scanned/image-based pages
    """

    # Patterns to detect code blocks
    CODE_PATTERNS = [
        r'^\s*(public|private|protected|class|interface|void|static)\s+',  # Apex/Java
        r'^\s*(function|const|let|var|import|export)\s+',                   # JavaScript
        r'^\s*(SELECT|FROM|WHERE|INSERT|UPDATE|DELETE)\s+',                 # SOQL/SQL
        r'^\s*<[a-z]+[^>]*>',                                              # XML/HTML
        r'^\s*\{[\s\S]*\}',                                                 # JSON-like
    ]

    # Minimum characters for a page to be considered "has text"
    OCR_THRESHOLD = 100

    def __init__(self, pdf_path: Path, ocr_threshold: int = None):
        """
        Initialize PDF extractor.

        Args:
            pdf_path: Path to PDF file
            ocr_threshold: Minimum chars before OCR is used (default: 100)
        """
        self.pdf_path = Path(pdf_path)
        self.ocr_threshold = ocr_threshold or self.OCR_THRESHOLD

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        # Open documents
        self._doc = fitz.open(str(self.pdf_path))
        self._pdf_plumber = pdfplumber.open(str(self.pdf_path))

        # Classify all pages
        self._page_types = self._classify_pages()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close PDF documents."""
        if self._doc:
            self._doc.close()
        if self._pdf_plumber:
            self._pdf_plumber.close()

    @property
    def page_count(self) -> int:
        """Number of pages in the PDF."""
        return len(self._doc)

    def get_metadata(self) -> PDFMetadata:
        """Extract PDF metadata including table of contents."""
        metadata = self._doc.metadata or {}

        # Extract TOC (table of contents)
        toc = []
        try:
            raw_toc = self._doc.get_toc()
            for item in raw_toc:
                level, title, page = item[0], item[1], item[2]
                toc.append((level, title, page))
        except Exception:
            pass

        return PDFMetadata(
            filename=self.pdf_path.name,
            filepath=str(self.pdf_path.absolute()),
            page_count=self.page_count,
            title=metadata.get("title") or self.pdf_path.stem,
            author=metadata.get("author"),
            subject=metadata.get("subject"),
            toc=toc
        )

    def _classify_pages(self) -> Dict[int, str]:
        """
        Classify each page by content type.

        Returns:
            Dict mapping page number to content type
        """
        page_types = {}

        for page_num in range(len(self._doc)):
            page = self._doc[page_num]

            # Get basic text to check if page has content
            text = page.get_text("text")
            text_chars = len(text.strip())

            # Check for images
            images = page.get_images()
            has_images = len(images) > 0

            # Check for tables using pdfplumber
            plumber_page = self._pdf_plumber.pages[page_num]
            tables = plumber_page.find_tables()
            has_tables = len(tables) > 0

            # Determine page type
            if text_chars < self.ocr_threshold and has_images:
                # Scanned page - needs OCR
                page_types[page_num] = "ocr"
            elif has_tables:
                # Has tables - use pdfplumber
                page_types[page_num] = "table"
            elif self._has_code(text):
                # Contains code blocks
                page_types[page_num] = "code"
            else:
                # Regular text page
                page_types[page_num] = "text"

        return page_types

    def _has_code(self, text: str) -> bool:
        """Check if text contains code patterns."""
        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                return True
        return False

    def extract_page(self, page_num: int) -> PageContent:
        """
        Extract content from a single page.

        Args:
            page_num: 0-indexed page number

        Returns:
            PageContent with extracted text and metadata
        """
        page_type = self._page_types.get(page_num, "text")
        page = self._doc[page_num]
        plumber_page = self._pdf_plumber.pages[page_num]

        # Check for images
        has_images = len(page.get_images()) > 0

        # Extract based on page type
        if page_type == "ocr":
            text, tables = self._extract_with_ocr(page)
        elif page_type == "table":
            text, tables = self._extract_with_tables(plumber_page)
        else:
            text = self._extract_text(page)
            tables = []

        # Clean up text
        text = self._clean_text(text)

        return PageContent(
            page_num=page_num + 1,  # 1-indexed for human readability
            text=text,
            content_type=page_type,
            has_images=has_images,
            has_tables=len(tables) > 0,
            word_count=len(text.split()),
            tables=tables
        )

    def _extract_text(self, page: fitz.Page) -> str:
        """Extract text using PyMuPDF (fast path)."""
        # Use "text" mode for clean text extraction
        text = page.get_text("text")
        return text

    def _extract_with_tables(self, page) -> Tuple[str, List]:
        """Extract content using pdfplumber for better table handling."""
        # Extract tables
        tables = []
        for table in page.find_tables():
            try:
                extracted = table.extract()
                if extracted:
                    tables.append(extracted)
            except Exception:
                continue

        # Get text excluding table areas
        text = page.extract_text() or ""

        # If tables found, append formatted table text
        if tables:
            text += "\n\n"
            for i, table in enumerate(tables):
                text += self._format_table(table)
                text += "\n\n"

        return text, tables

    def _extract_with_ocr(self, page: fitz.Page) -> Tuple[str, List]:
        """Extract text using OCR for scanned pages."""
        try:
            import pytesseract
            from PIL import Image
            import io

            # Render page to image
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # Run OCR
            text = pytesseract.image_to_string(img)
            return text, []

        except ImportError:
            # Fallback if OCR not available
            return page.get_text("text"), []
        except Exception as e:
            # Return whatever text we can get
            return page.get_text("text") or f"[OCR Error: {e}]", []

    def _format_table(self, table: List[List[str]]) -> str:
        """Format extracted table as markdown."""
        if not table or not table[0]:
            return ""

        lines = []
        # Header
        header = " | ".join(str(cell or "") for cell in table[0])
        lines.append(f"| {header} |")

        # Separator
        sep = " | ".join("---" for _ in table[0])
        lines.append(f"| {sep} |")

        # Data rows
        for row in table[1:]:
            row_text = " | ".join(str(cell or "") for cell in row)
            lines.append(f"| {row_text} |")

        return "\n".join(lines)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Hyphenation
        text = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', text)  # Line breaks in sentences

        # Remove form feed and other control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        return text.strip()

    def extract_all(self, progress: Progress = None, task_id: TaskID = None) -> Iterator[PageContent]:
        """
        Extract all pages from the PDF.

        Args:
            progress: Optional Rich progress bar
            task_id: Optional task ID for progress updates

        Yields:
            PageContent for each page
        """
        for page_num in range(self.page_count):
            yield self.extract_page(page_num)

            if progress and task_id:
                progress.update(task_id, advance=1)

    def get_toc_structure(self) -> Dict[int, Dict]:
        """
        Get table of contents mapped to page numbers.

        Returns:
            Dict mapping page numbers to chapter/section info
        """
        metadata = self.get_metadata()
        toc_map = {}

        current_chapter = ""
        current_section = ""

        for level, title, page in metadata.toc:
            if level == 1:
                current_chapter = title
                current_section = ""
            elif level == 2:
                current_section = title

            # Map this page and subsequent pages until next TOC entry
            toc_map[page] = {
                "chapter": current_chapter,
                "section": current_section,
                "level": level,
                "title": title
            }

        # Fill in gaps - propagate chapter/section info
        if toc_map:
            last_info = {"chapter": "", "section": "", "level": 0, "title": ""}
            for page_num in range(1, self.page_count + 1):
                if page_num in toc_map:
                    last_info = toc_map[page_num]
                else:
                    toc_map[page_num] = last_info.copy()

        return toc_map
