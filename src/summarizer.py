from src.prompts import build_summary_prompt


class SummarizationError(Exception):
    pass


def summarize_document(
    text: str,
    ai_response_function,
) -> str:
    """
    Summarize a document using only the document text.
    """

    if not text or not text.strip():
        raise ValueError("Document text cannot be empty.")

    if ai_response_function is None:
        raise ValueError("AI response function cannot be None.")

    prompt = build_summary_prompt(text)

    try:
        summary = ai_response_function(prompt)
    except Exception as error:
        raise SummarizationError(
            f"Failed to generate document summary: {error}"
        ) from error

    if not summary or not summary.strip():
        raise SummarizationError(
            "The AI returned an empty summary."
        )

    return summary.strip()
