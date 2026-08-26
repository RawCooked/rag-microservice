"""Document parsing with Docling.

Docling converts PDF, Word, PowerPoint, Excel, HTML, and images (with
OCR) into structured markdown — layout-aware, so tables survive as
markdown tables instead of dissolving into word soup. That structure is
what the chunker downstream relies on.

The docling import happens inside the class so that importing rag_core
stays instant — the heavy ML models load only when a parser is created.
"""

from __future__ import annotations

from pathlib import Path

from rag_core.interfaces import PAGE_BREAK, ParsedDocument

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".md", ".html", ".csv",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
}


class DoclingParser:
    def __init__(self) -> None:
        from docling.document_converter import DocumentConverter

        self._converter = DocumentConverter()

    def parse(self, path: Path) -> ParsedDocument:
        result = self._converter.convert(path)
        markdown = result.document.export_to_markdown(page_break_placeholder=PAGE_BREAK)
        num_pages = markdown.count(PAGE_BREAK) + 1 if PAGE_BREAK in markdown else None
        return ParsedDocument(source=path.name, markdown=markdown, num_pages=num_pages)
