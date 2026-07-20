import re

MARKDOWN_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
HCL_BLOCK_START_RE = re.compile(
    r'^[ \t]*(resource|module|variable|output|data|provider)\s+("[^"]*"\s*)*\{',
    re.MULTILINE,
)

FALLBACK_WINDOW_WORDS = 300
FALLBACK_OVERLAP_WORDS = 50


def split_by_markdown_headers(text: str) -> list[dict]:
    matches = list(MARKDOWN_HEADER_RE.finditer(text))
    if not matches:
        return [{"header": None, "text": text}]

    sections = []
    if matches[0].start() > 0:
        sections.append({"header": None, "text": text[: matches[0].start()]})

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({"header": match.group(2).strip(), "text": text[start:end]})

    return sections


def _find_matching_brace(text: str, open_brace_index: int) -> int:
    depth = 0
    for i in range(open_brace_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def _parse_resource_type(block_text: str) -> str | None:
    match = re.match(r'\s*resource\s+"([^"]+)"', block_text)
    return match.group(1) if match else None


def split_by_hcl_blocks(section_text: str) -> list[dict]:
    blocks = []
    cursor = 0
    starts = list(HCL_BLOCK_START_RE.finditer(section_text))

    if not starts:
        return _fallback_window(section_text)

    for match in starts:
        if match.start() > cursor:
            leftover = section_text[cursor : match.start()]
            blocks.extend(_fallback_window(leftover))

        open_brace_index = section_text.index("{", match.start())
        close_brace_index = _find_matching_brace(section_text, open_brace_index)
        block_text = section_text[match.start() : close_brace_index + 1]
        blocks.append({"text": block_text, "resource_type": _parse_resource_type(block_text)})
        cursor = close_brace_index + 1

    if cursor < len(section_text):
        blocks.extend(_fallback_window(section_text[cursor:]))

    return blocks


def _fallback_window(text: str) -> list[dict]:
    words = text.split()
    if not words:
        return []
    if len(words) <= FALLBACK_WINDOW_WORDS:
        return [{"text": text.strip(), "resource_type": None}] if text.strip() else []

    chunks = []
    step = FALLBACK_WINDOW_WORDS - FALLBACK_OVERLAP_WORDS
    for start in range(0, len(words), step):
        window = words[start : start + FALLBACK_WINDOW_WORDS]
        if window:
            chunks.append({"text": " ".join(window), "resource_type": None})
        if start + FALLBACK_WINDOW_WORDS >= len(words):
            break
    return chunks


def chunk_document(text: str, base_metadata: dict) -> list[dict]:
    chunks = []
    for section in split_by_markdown_headers(text):
        for block in split_by_hcl_blocks(section["text"]):
            if not block["text"].strip():
                continue
            metadata = {
                **base_metadata,
                "section_header": section["header"],
                "resource_type": block["resource_type"],
            }
            chunks.append({"text": block["text"], "metadata": metadata})
    return chunks