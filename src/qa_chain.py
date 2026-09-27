from src.prompts import build_qa_prompt
from src.retriever import retrieve_relevant_chunks, format_retrieved_context


class QAError(Exception):
    pass


def answer_question(
    question: str,
    vector_store,
    ai_response_function,
    top_k: int = 4,
) -> dict:
    """
    Answer a user's question using only retrieved document context.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if vector_store is None:
        raise ValueError("Vector store cannot be None.")

    if ai_response_function is None:
        raise ValueError("AI response function cannot be None.")

    # Retrieve relevant document chunks.
    retrieved_chunks = retrieve_relevant_chunks(
        question=question,
        vector_store=vector_store,
        top_k=top_k,
    )

    # No relevant information was found.
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find relevant information in the document.",
            "sources": [],
        }

    # Convert retrieved chunks into context for the AI.
    context = format_retrieved_context(retrieved_chunks)

    # Build the grounded Q&A prompt.
    prompt = build_qa_prompt(
        question=question,
        context=context,
    )

    # Ask the AI model.
    try:
        answer = ai_response_function(prompt)
    except Exception as error:
        raise QAError(
            f"Failed to generate an answer: {error}"
        ) from error

    if not answer:
        raise QAError("The AI returned an empty answer.")

    # Keep source information so the UI can show where
    # the answer came from.
    sources = []

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})

        sources.append({
            "source": metadata.get("source", ""),
            "chunk_index": metadata.get("chunk_index", ""),
            "distance": chunk.get("distance"),
        })

    return {
        "answer": answer.strip(),
        "sources": sources,
    }
