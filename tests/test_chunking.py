from src.chunking import (
    chunk_document,
    split_by_hcl_blocks,
    split_by_markdown_headers,
    _find_matching_brace,
    _fallback_window,
    _parse_resource_type,
)


def test_split_by_markdown_headers_basic():
    text = "# Title\nintro text\n## Section A\nbody a\n## Section B\nbody b\n"
    sections = split_by_markdown_headers(text)
    assert [s["header"] for s in sections] == ["Title", "Section A", "Section B"]
    assert "body a" in sections[1]["text"]
    assert "body b" in sections[2]["text"]


def test_split_by_markdown_headers_no_headers_returns_single_section():
    text = "just plain text, no headers at all"
    sections = split_by_markdown_headers(text)
    assert len(sections) == 1
    assert sections[0]["header"] is None
    assert sections[0]["text"] == text


def test_split_by_markdown_headers_preserves_text_before_first_header():
    text = "preamble\n# First Header\nbody"
    sections = split_by_markdown_headers(text)
    assert sections[0]["header"] is None
    assert "preamble" in sections[0]["text"]
    assert sections[1]["header"] == "First Header"


def test_find_matching_brace_handles_nesting():
    text = 'resource "aws_security_group" "web" {\n  ingress {\n    port = 80\n  }\n}\nafter'
    open_index = text.index("{")
    close_index = _find_matching_brace(text, open_index)
    assert text[close_index] == "}"
    # the matched close brace must be the outer one, not the inner ingress block's
    assert text[close_index:].startswith("}\nafter")


def test_parse_resource_type_extracts_type_from_resource_block():
    block = 'resource "aws_instance" "web" {\n  ami = "x"\n}'
    assert _parse_resource_type(block) == "aws_instance"


def test_parse_resource_type_returns_none_for_non_resource_blocks():
    block = 'variable "instance_type" {\n  type = string\n}'
    assert _parse_resource_type(block) is None


def test_split_by_hcl_blocks_extracts_full_nested_block_intact():
    section = (
        'Some intro text.\n\n'
        'resource "aws_security_group" "web" {\n'
        '  ingress {\n'
        '    from_port = 80\n'
        '    to_port   = 80\n'
        '  }\n'
        '  egress {\n'
        '    from_port = 0\n'
        '  }\n'
        '}\n\n'
        'Trailing text after the block.'
    )
    blocks = split_by_hcl_blocks(section)
    hcl_blocks = [b for b in blocks if b["resource_type"] or "resource \"aws_security_group\"" in b["text"]]
    assert len(hcl_blocks) == 1
    block_text = hcl_blocks[0]["text"]
    assert block_text.startswith('resource "aws_security_group" "web" {')
    assert block_text.rstrip().endswith("}")
    assert "egress" in block_text
    assert block_text.count("{") == block_text.count("}")


def test_split_by_hcl_blocks_never_splits_a_block_midway():
    # a block containing text that could confuse a naive line-based splitter
    section = (
        'resource "aws_iam_role" "x" {\n'
        '  assume_role_policy = jsonencode({\n'
        '    Statement = [{ Effect = "Allow" }]\n'
        '  })\n'
        '}\n'
    )
    blocks = split_by_hcl_blocks(section)
    resource_blocks = [b for b in blocks if b["resource_type"] == "aws_iam_role"]
    assert len(resource_blocks) == 1
    assert resource_blocks[0]["text"].count("{") == resource_blocks[0]["text"].count("}")


def test_split_by_hcl_blocks_parses_multiple_distinct_types():
    section = (
        'resource "aws_instance" "web" {\n  ami = "x"\n}\n\n'
        'variable "region" {\n  type = string\n}\n\n'
        'output "ip" {\n  value = aws_instance.web.public_ip\n}\n'
    )
    blocks = split_by_hcl_blocks(section)
    resource_types = [b["resource_type"] for b in blocks if b["resource_type"]]
    assert resource_types == ["aws_instance"]
    # variable/output blocks are still captured as their own blocks, just with resource_type=None
    non_fallback_block_count = sum(
        1 for b in blocks if b["text"].startswith(("resource", "variable", "output"))
    )
    assert non_fallback_block_count == 3


def test_split_by_hcl_blocks_falls_back_to_windowing_when_no_hcl_present():
    section = "This is plain prose with no HCL blocks in it whatsoever."
    blocks = split_by_hcl_blocks(section)
    assert len(blocks) == 1
    assert blocks[0]["resource_type"] is None
    assert blocks[0]["text"] == section


def test_fallback_window_short_text_stays_one_chunk():
    text = "a short paragraph that is well under the window size"
    chunks = _fallback_window(text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_fallback_window_empty_text_returns_no_chunks():
    assert _fallback_window("   ") == []
    assert _fallback_window("") == []


def test_fallback_window_long_text_splits_with_overlap():
    words = [f"word{i}" for i in range(700)]
    text = " ".join(words)
    chunks = _fallback_window(text)
    assert len(chunks) > 1
    # every chunk should respect the configured window size
    for chunk in chunks:
        assert len(chunk["text"].split()) <= 300
    # consecutive chunks should overlap (share some words) rather than losing context at the boundary
    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    assert set(first_words) & set(second_words)


def test_chunk_document_combines_headers_and_hcl_blocks_with_metadata():
    text = (
        "# aws_instance\n\n"
        "Some description text.\n\n"
        '```hcl\nresource "aws_instance" "web" {\n  ami = "x"\n}\n```\n\n'
        "## Notes\n\nAdditional prose here about usage."
    )
    chunks = chunk_document(text, base_metadata={"source_path": "aws_instance.md", "source_authority": "official"})

    assert all(c["metadata"]["source_authority"] == "official" for c in chunks)
    assert all(c["metadata"]["source_path"] == "aws_instance.md" for c in chunks)

    resource_chunks = [c for c in chunks if c["metadata"]["resource_type"] == "aws_instance"]
    assert len(resource_chunks) == 1
    assert resource_chunks[0]["metadata"]["section_header"] == "aws_instance"

    notes_chunks = [c for c in chunks if c["metadata"]["section_header"] == "Notes"]
    assert len(notes_chunks) == 1
    assert notes_chunks[0]["metadata"]["resource_type"] is None


def test_chunk_document_skips_blank_chunks():
    text = "# Header\n\n\n\n## Next\n\nreal content"
    chunks = chunk_document(text, base_metadata={"source_path": "x", "source_authority": "community"})
    assert all(c["text"].strip() for c in chunks)