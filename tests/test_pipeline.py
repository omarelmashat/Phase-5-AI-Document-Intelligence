"""End-to-end pipeline with sample documents and a mocked AI (no network)."""
import hashlib
import os

import pytest

import config
from src import api_client, utils
from src.chunker import chunk_document, chunk_text
from src.classifier import classify_document
from src.document_loader import load_document
from src.field_extractor import extract_fields
from src.qa_chain import answer_question
from src.summarizer import summarize_document
from src.vector_store import DocumentVectorStore
from conftest import SAMPLES


def fake_embed(texts, input_type="search_document"):
    """Deterministic bag-of-words hash embedding: similar words -> close vectors."""
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().split():
            v[int(hashlib.md5(w.strip('.,:').encode()).hexdigest(), 16) % 64] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr("src.retriever.get_query_embedding", lambda q: fake_embed([q], "search_query")[0])
    text = load_document(os.path.join(SAMPLES, "sample_invoice.pdf"), "sample_invoice.pdf")
    chunks = chunk_document(text, "invoice", chunk_size=200, chunk_overlap=40)
    s = DocumentVectorStore(collection_name="test_" + os.urandom(4).hex())
    s.add_chunks(chunks, utils.embed_in_batches(fake_embed, [c["text"] for c in chunks], batch_size=3))
    return s


def test_chunker_overlap_and_validation():
    chunks = chunk_text("word " * 500, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1 and all(len(c) <= 200 for c in chunks)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, chunk_overlap=10)
    assert chunk_text("   ") == []


def test_qa_grounded_prompt_contains_document_text(store):
    seen = {}

    def ai(prompt):
        seen["prompt"] = prompt
        return "The total is 930.00 EGP."

    result = answer_question("What is the total amount due?", store, ai)
    assert "930.00" in seen["prompt"]
    assert result["answer"].startswith("The total")
    assert result["sources"]


def test_unrelated_question_gets_not_found_without_calling_ai(store):
    guarded = utils.ThresholdedStore(store, max_distance=0.0)  # nothing is close enough

    def ai(prompt):
        raise AssertionError("AI must not be called when nothing relevant was retrieved")

    result = answer_question("Who won the world cup?", guarded, ai)
    assert "couldn't find" in result["answer"].lower()
    assert result["sources"] == []


def test_ai_failure_becomes_message_not_crash(store):
    def ai(prompt):
        raise ConnectionError("network down")

    result, error = utils.safe_call(answer_question, "total amount?", store, ai,
                                    error_message="Sorry, try again.")
    assert result is None and "Sorry, try again." in error


def test_new_document_does_not_leak_into_old_one(store):
    other = DocumentVectorStore(collection_name="test_" + os.urandom(4).hex())
    assert other.is_empty() and not store.is_empty()
    store.reset()
    assert store.is_empty()


def test_summary_classifier_and_fields_with_mock_ai():
    text = "Invoice INV-1 total 50.00"
    assert summarize_document(text, lambda p: " A short summary. ") == "A short summary."
    assert classify_document(text, lambda p: "Invoice") == "invoice"
    assert classify_document(text, lambda p: "spaceship") == "other"
    assert extract_fields(text, lambda p: '```json\n{"invoice_number": "INV-1"}\n```') == {"invoice_number": "INV-1"}
    with pytest.raises(Exception):
        extract_fields(text, lambda p: "not json")


def test_summarizer_uses_map_reduce_for_long_documents(monkeypatch):
    import src.summarizer as summarizer
    monkeypatch.setattr(summarizer, "MAP_REDUCE_THRESHOLD_CHARS", 50)
    monkeypatch.setattr(summarizer, "SECTION_CHUNK_SIZE", 40)
    monkeypatch.setattr(summarizer, "SECTION_CHUNK_OVERLAP", 5)
    calls = []

    def ai(prompt):
        calls.append(prompt)
        return "combined summary" if "SECTION_SUMMARIES" in prompt else "partial summary"

    result = summarizer.summarize_document("word " * 40, ai)
    assert result == "combined summary"
    assert len(calls) > 2


def test_embeddings_delegates_to_shared_api_client(monkeypatch):
    import src.embeddings as embeddings
    seen = []

    def fake_embed_texts(texts, input_type="search_document"):
        seen.append((tuple(texts), input_type))
        return [[1.0, 2.0]] * len(texts)

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)
    assert embeddings.get_embeddings(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
    assert embeddings.get_query_embedding("q") == [1.0, 2.0]
    assert seen[0][1] == "search_document" and seen[1][1] == "search_query"


def test_api_client_retries_then_returns(monkeypatch):
    monkeypatch.setattr(config, "API_RETRY_DELAY_SECONDS", 0)
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("slow")
        return "  hello  "

    monkeypatch.setattr(api_client, "_chat_once", flaky)
    assert api_client.get_ai_response("hi") == "hello"


def test_api_client_reports_persistent_failure(monkeypatch):
    monkeypatch.setattr(config, "API_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(api_client, "_chat_once", lambda p: (_ for _ in ()).throw(TimeoutError("x")))
    with pytest.raises(api_client.APIClientError, match="could not be reached"):
        api_client.get_ai_response("hi")


def test_api_client_missing_key_message(monkeypatch):
    monkeypatch.setattr(config, "AI_PROVIDER", "cohere")
    monkeypatch.setattr(config, "COHERE_API_KEY", "")
    api_client._clients.clear()
    with pytest.raises(api_client.APIClientError, match="COHERE_API_KEY"):
        api_client._chat_once("hi")


def test_api_client_rejects_empty_prompt():
    with pytest.raises(ValueError):
        api_client.get_ai_response("  ")
