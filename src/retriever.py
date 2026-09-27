from src.embeddings import get_query_embedding
from src.vector_store import DocumentVectorStore, VectorStoreError


DEFAULT_TOP_K = 4


class RetrievalError(Exception):
    pass


def retrieve_relevant_chunks(
    question: str,
    vector_store: DocumentVectorStore,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if vector_store is None:
        raise ValueError("Vector store cannot be None.")

    try:
        query_embedding = get_query_embedding(
            question.strip()
        )

        return vector_store.search(
            query_embedding,
            top_k=top_k,
        )

    except VectorStoreError as error:
        raise RetrievalError(
            f"Failed to retrieve relevant chunks: {error}"
        ) from error

    except Exception as error:
        raise RetrievalError(
            f"Failed to create query embedding: {error}"
        ) from error


def format_retrieved_context(
    results: list[dict],
) -> str:

    if not results:
        return ""

    parts = []

    for index, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})

        source = metadata.get(
            "source",
            "unknown",
        )

        text = result.get(
            "text",
            "",
        ).strip()

        if text:
            parts.append(
                f"[Source {index}: {source}]\n{text}"
            )

    return "\n\n".join(parts)

