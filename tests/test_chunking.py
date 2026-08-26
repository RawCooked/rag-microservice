from rag_core.chunking import MarkdownChunker
from rag_core.interfaces import PAGE_BREAK, ParsedDocument


def make_doc(markdown: str) -> ParsedDocument:
    num_pages = markdown.count(PAGE_BREAK) + 1 if PAGE_BREAK in markdown else None
    return ParsedDocument(source="test.pdf", markdown=markdown, num_pages=num_pages)


def test_heading_starts_new_chunk_and_is_prepended():
    doc = make_doc("# Refunds\n\nWithin 30 days.\n\n# Shipping\n\nTakes one week.")
    chunks = MarkdownChunker().chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].heading == "Refunds"
    assert chunks[0].text.startswith("Refunds")  # context travels with the chunk
    assert "Within 30 days." in chunks[0].text
    assert chunks[1].heading == "Shipping"


def test_page_breaks_attribute_chunks_to_pages():
    doc = make_doc(f"First page text.\n\n{PAGE_BREAK}\n\n# Section\n\nSecond page text.")
    chunks = MarkdownChunker().chunk(doc)
    assert chunks[0].page == 1
    assert chunks[1].page == 2


def test_no_page_markers_means_no_page_numbers():
    chunks = MarkdownChunker().chunk(make_doc("Just some text."))
    assert chunks[0].page is None


def test_long_sections_split_at_paragraph_boundaries():
    paragraphs = "\n\n".join(f"Paragraph {i}. " + "x" * 300 for i in range(6))
    chunks = MarkdownChunker(max_chars=700).chunk(make_doc(f"# Big\n\n{paragraphs}"))
    assert len(chunks) > 1
    # every chunk keeps the section heading as context
    assert all(chunk.heading == "Big" for chunk in chunks)
    # no chunk is wildly over the cap (cap + one paragraph at most)
    assert all(len(chunk.text) < 1200 for chunk in chunks)


def test_table_lines_stay_together():
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    doc = make_doc(f"# Data\n\nIntro paragraph.\n\n{table}")
    chunks = MarkdownChunker().chunk(doc)
    joined = [chunk for chunk in chunks if "| 1 | 2 |" in chunk.text]
    assert len(joined) == 1
    assert "| 3 | 4 |" in joined[0].text  # table was not split


def test_chunk_ids_are_deterministic():
    doc = make_doc("# A\n\nSome text.")
    first = MarkdownChunker().chunk(doc)
    second = MarkdownChunker().chunk(doc)
    assert [c.id for c in first] == [c.id for c in second]
