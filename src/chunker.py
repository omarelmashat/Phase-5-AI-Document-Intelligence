DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size

        if end < length:
            break_point = text.rfind(" ", start, end)
            if break_point > start:
                end = break_point

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_document(
    text: str,
    source_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        {
            "id": f"{source_id}_chunk_{i}",
            "text": piece,
            "source": source_id,
            "chunk_index": i,
        }
        for i, piece in enumerate(pieces)
    ]
