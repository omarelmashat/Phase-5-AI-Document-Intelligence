import config
from src.api_client import APIClientError, embed_texts
from src.utils import embed_in_batches


class EmbeddingError(Exception):
    pass


def get_embeddings(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    if not texts:
        return []
    try:
        return embed_in_batches(
            lambda batch: embed_texts(batch, input_type=input_type),
            texts,
            batch_size=config.EMBED_BATCH_SIZE,
        )
    except APIClientError as error:
        raise EmbeddingError(str(error)) from error


def get_query_embedding(text: str) -> list[float]:
    return get_embeddings([text], input_type="search_query")[0]
