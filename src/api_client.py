
import config
from src.utils import retry_call


class APIClientError(Exception):
    """User-presentable failure from the AI provider."""


_clients = {}


def _cohere():
    if "cohere" not in _clients:
        if not config.COHERE_API_KEY:
            raise APIClientError("COHERE_API_KEY is not set. Add it to your .env file.")
        import cohere
        _clients["cohere"] = cohere.ClientV2(api_key=config.COHERE_API_KEY,
                                             timeout=config.API_TIMEOUT_SECONDS)
    return _clients["cohere"]


def _hf(model: str):
    key = f"hf:{model}"
    if key not in _clients:
        if not config.HF_API_KEY:
            raise APIClientError("HF_API_KEY is not set. Add it to your .env file.")
        from huggingface_hub import InferenceClient
        _clients[key] = InferenceClient(model=model, token=config.HF_API_KEY,
                                        timeout=config.API_TIMEOUT_SECONDS)
    return _clients[key]


def _chat_once(prompt: str) -> str:
    if config.AI_PROVIDER == "cohere":
        response = _cohere().chat(
            model=config.COHERE_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return "".join(part.text for part in response.message.content if hasattr(part, "text"))
    if config.AI_PROVIDER == "huggingface":
        response = _hf(config.HF_CHAT_MODEL).chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=max(config.LLM_TEMPERATURE, 0.01),
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content or ""
    raise APIClientError(f"Unknown AI_PROVIDER '{config.AI_PROVIDER}'. Use 'cohere' or 'huggingface'.")


def get_ai_response(prompt: str) -> str:
    """Send a prompt, return the model's text. Retries transient failures;
    raises APIClientError with a readable message if it still fails."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    try:
        text = retry_call(_chat_once, prompt, exceptions=(Exception,))
    except APIClientError:
        raise
    except Exception as error:
        raise APIClientError(f"The AI service could not be reached ({error}). Please try again.") from error
    if not text or not text.strip():
        raise APIClientError("The AI service returned an empty response. Please try again.")
    return text.strip()


def embed_texts(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """Embedding entry point for Role 2 (embeddings.py can delegate here)."""
    if not texts:
        return []
    try:
        if config.EMBEDDING_PROVIDER == "cohere":
            def call():
                r = _cohere().embed(texts=texts, model=config.COHERE_EMBED_MODEL,
                                    input_type=input_type, embedding_types=["float"])
                return [list(v) for v in r.embeddings.float]
        elif config.EMBEDDING_PROVIDER == "huggingface":
            def call():
                import numpy as np
                vectors = np.array(_hf(config.HF_EMBED_MODEL).feature_extraction(texts))
                if vectors.ndim == 3:
                    vectors = vectors.mean(axis=1)
                return vectors.tolist()
        else:
            raise APIClientError(f"Unknown EMBEDDING_PROVIDER '{config.EMBEDDING_PROVIDER}'.")
        return retry_call(call)
    except APIClientError:
        raise
    except Exception as error:
        raise APIClientError(f"Embedding service failed ({error}). Please try again.") from error
