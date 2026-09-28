
import logging
import os
import re
import shutil
import tempfile
import time
import uuid

import config

logger = logging.getLogger("docintel")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def retry_call(func, *args, retries=None, delay=None, exceptions=(Exception,), **kwargs):
    """Call func(*args, **kwargs); on failure retry with exponential backoff.
    Re-raises the last error once retries are used up."""
    retries = config.API_MAX_RETRIES if retries is None else retries
    delay = config.API_RETRY_DELAY_SECONDS if delay is None else delay
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except exceptions as error:
            attempt += 1
            if attempt > retries:
                logger.error("Giving up on %s after %d attempts: %s",
                             getattr(func, "__name__", func), attempt, error)
                raise
            wait = delay * (2 ** (attempt - 1))
            logger.warning("%s failed (%s). Retry %d/%d in %.1fs",
                           getattr(func, "__name__", func), error, attempt, retries, wait)
            time.sleep(wait)


def safe_call(func, *args, default=None, error_message="Something went wrong.", **kwargs):
    try:
        return func(*args, **kwargs), None
    except Exception as error:  # noqa: BLE001 - this is the crash barrier
        logger.exception("safe_call caught error in %s", getattr(func, "__name__", func))
        return default, f"{error_message} ({error})"

def create_session_dir() -> str:
    return tempfile.mkdtemp(prefix=config.TEMP_ROOT_PREFIX)


def save_upload(session_dir: str, filename: str, data: bytes) -> str:

    ext = os.path.splitext(os.path.basename(filename))[1].lower()
    path = os.path.join(session_dir, f"upload_{uuid.uuid4().hex[:8]}{ext}")
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def cleanup_session_dir(session_dir) -> None:
    if session_dir and os.path.isdir(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)

def trim_history(history: list[dict], limit: int | None = None) -> list[dict]:
    limit = config.HISTORY_LIMIT if limit is None else limit
    return history[-limit:] if limit > 0 else []


def format_history(history: list[dict]) -> str:
    lines = []
    for message in trim_history(history):
        who = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{who}: {message['content']}")
    return "\n".join(lines)


def with_history(ai_response_function, history: list[dict]):
    context = format_history(history)
    if not context:
        return ai_response_function

    def call(prompt: str) -> str:
        return ai_response_function(
            "Earlier conversation about this same document (use it only to understand "
            "what the new question refers to; facts must still come from the document "
            f"context):\n{context}\n\n{prompt}"
        )

    return call

class ThresholdedStore:


    def __init__(self, store, max_distance: float | None = None):
        self._store = store
        self._max = config.MAX_DISTANCE if max_distance is None else max_distance

    def search(self, query_embedding, top_k=config.TOP_K):
        results = self._store.search(query_embedding, top_k=top_k)
        return [r for r in results if r.get("distance") is None or r["distance"] <= self._max]

    def __getattr__(self, name):
        return getattr(self._store, name)


def embed_in_batches(embed_function, texts: list[str], batch_size: int | None = None) -> list[list[float]]:

    batch_size = batch_size or config.EMBED_BATCH_SIZE
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(retry_call(embed_function, batch))
    return vectors
def _norm(value: str) -> str:
    value = str(value).lower()
    value = re.sub(r"(?<=\d),(?=\d)", "", value)  # 1,200 -> 1200
    return re.sub(r"[\s]+", " ", value).strip()


def _flatten(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _flatten(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _flatten(v)
    elif value is not None:
        yield value


def validate_fields(fields: dict, source_text: str) -> dict:
    haystack = _norm(source_text)
    checked = {}
    for name, value in fields.items():
        parts = [str(p) for p in _flatten(value) if str(p).strip()]
        if not parts:
            continue
        verified = all(_norm(p) in haystack for p in parts)
        checked[name] = {"value": value, "verified": verified}
    return checked
