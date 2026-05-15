"""Document loaders for supported file types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from docx import Document as DocxDocument
from pypdf import PdfReader

from app.utils.helpers import normalize_whitespace


@dataclass
class ExtractedPage:
    """Intermediate extracted page or section."""

    text: str
    page_number: Optional[int]


class BaseLoader:
    """Base contract for file loaders."""

    def load(self, path: Path) -> List[ExtractedPage]:
        raise NotImplementedError


class PDFLoader(BaseLoader):
    """Load text from PDF pages."""

    def load(self, path: Path) -> List[ExtractedPage]:
        reader = PdfReader(str(path))
        pages: List[ExtractedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = normalize_whitespace(page.extract_text() or "")
            if text:
                pages.append(ExtractedPage(text=text, page_number=index))
        return pages


class TextLoader(BaseLoader):
    """Load UTF-8 text files."""

    def load(self, path: Path) -> List[ExtractedPage]:
        text = normalize_whitespace(path.read_text(encoding="utf-8", errors="ignore"))
        return [ExtractedPage(text=text, page_number=None)] if text else []


class MarkdownLoader(TextLoader):
    """Markdown is treated as plain text."""


class DocxLoader(BaseLoader):
    """Load paragraph text from DOCX files."""

    def load(self, path: Path) -> List[ExtractedPage]:
        doc = DocxDocument(str(path))
        paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
        text = normalize_whitespace("\n\n".join(paragraphs))
        return [ExtractedPage(text=text, page_number=None)] if text else []


def get_loader(path: Path) -> BaseLoader:
    """Resolve the loader for a given file path."""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PDFLoader()
    if suffix in {".txt"}:
        return TextLoader()
    if suffix in {".md", ".markdown"}:
        return MarkdownLoader()
    if suffix == ".docx":
        return DocxLoader()
    raise ValueError(f"Unsupported file type: {suffix}")
