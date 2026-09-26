
import os

import numpy as np
import cohere
from huggingface_hub import InferenceClient

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cohere").lower()
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class EmbeddingError(Exception):

def _cohere_client() -> cohere.ClientV2:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise EmbeddingError("Missing COHERE_API_KEY environment variable.")
    return cohere.ClientV2(api_key=api_key)


def _hf_client() -> InferenceClient:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise EmbeddingError("Missing HF_API_KEY environment variable.")
    return InferenceClient(model=HF_EMBED_MODEL, token=api_key)


def _embed_with_cohere(texts: list[str], input_type: str) -> list[list[float]]:
    client = _cohere_client()
    try:
        response = client.embed(
            texts=texts,
            model=COHERE_EMBED_MODEL,
            input_type=input_type,
            embedding_types=["float"],
        )
        return [list(vector) for vector in response.embeddings.float]
    except EmbeddingError:
        raise
    except Exception as error:
        raise EmbeddingError(f"Cohere embedding request failed: {error}") from error


def _embed_with_huggingface(texts: list[str]) -> list[list[float]]:
    client = _hf_client()
    try:
        vectors = np.array(client.feature_extraction(texts))
        if vectors.ndim == 3:
            vectors = vectors.mean(axis=1)
        return vectors.tolist()
    except EmbeddingError:
        raise
    except Exception as error:
        raise EmbeddingError(f"Hugging Face embedding request failed: {error}") from error


def get_embeddings(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    if not texts:
        return []

    if EMBEDDING_PROVIDER == "cohere":
        return _embed_with_cohere(texts, input_type)
    if EMBEDDING_PROVIDER == "huggingface":
        return _embed_with_huggingface(texts)

    raise EmbeddingError(
        f"Unknown EMBEDDING_PROVIDER: '{EMBEDDING_PROVIDER}'. Use 'cohere' or 'huggingface'."
    )


def get_query_embedding(text: str) -> list[float]:
    return get_embeddings([text], input_type="search_query")[0]
