"""Structure-aware chunking of parsed markdown.

Instead of blindly cutting every N characters (which slices sentences
and tables in half), chunks follow the document's own structure:

- a new heading starts a new chunk, and the heading text is prepended
  to each chunk under it — so an embedded chunk carries its context
  ("Refund policy: ..." instead of a floating "within 30 days")
- table lines are never separated from each other
- chunks are capped at `max_chars`, splitting at paragraph boundaries
- page-break markers from the parser attribute each chunk to a page

Chunk IDs are deterministic (uuid5 of source + index), so re-ingesting
the same file overwrites its old chunks instead of duplicating them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from rag_core.interfaces import PAGE_BREAK, Chunk, ParsedDocument


def _split_blocks(markdown: str) -> list[str]:
    """Split markdown into blocks: paragraphs, headings, tables, page markers.

    Consecutive table lines stay together as one block.
    """
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped == PAGE_BREAK:
            flush()
            blocks.append(PAGE_BREAK)
        elif not stripped:
            flush()
        elif stripped.startswith("#"):
            flush()
            blocks.append(stripped)
        else:
            current.append(line)
    flush()
    return blocks


@dataclass
class MarkdownChunker:
    max_chars: int = 1200

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        has_pages = PAGE_BREAK in doc.markdown
        page = 1 if has_pages else None
        heading: str | None = None

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buf_page, buf_heading = page, heading

        def flush() -> None:
            text = "\n\n".join(buffer).strip()
            buffer.clear()
            if not text:
                return
            if buf_heading:
                text = f"{buf_heading}\n\n{text}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.source}:{len(chunks)}"))
            chunks.append(
                Chunk(id=chunk_id, text=text, source=doc.source, page=buf_page, heading=buf_heading)
            )

        for block in _split_blocks(doc.markdown):
            if block == PAGE_BREAK:
                page += 1  # type: ignore[operator]
                continue
            if block.startswith("#"):
                flush()
                heading = block.lstrip("#").strip()
                buf_page, buf_heading = page, heading
                continue
            if buffer and sum(len(b) for b in buffer) + len(block) > self.max_chars:
                flush()
            if not buffer:
                buf_page, buf_heading = page, heading
            buffer.append(block)
        flush()
        return chunks
