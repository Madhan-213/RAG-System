from app.chunking.recursive_chunker import RecursiveTextChunker


def test_recursive_chunker_respects_chunk_size():
    chunker = RecursiveTextChunker(chunk_size=12, chunk_overlap=2)
    text = " ".join(f"token{i}" for i in range(30))

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 14 for chunk in chunks)
