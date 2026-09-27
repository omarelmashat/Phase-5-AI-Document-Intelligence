from src.prompts import build_classification_prompt


class ClassificationError(Exception):
    pass


ALLOWED_DOCUMENT_TYPES = {
    "contract",
    "invoice",
    "report",
    "resume",
    "letter",
    "legal_document",
    "financial_document",
    "academic_document",
    "other",
}


def classify_document(
    text: str,
    ai_response_function,
) -> str:
    """
    Classify a document based only on its content.
    """

    if not text or not text.strip():
        raise ValueError("Document text cannot be empty.")

    if ai_response_function is None:
        raise ValueError("AI response function cannot be None.")

    prompt = build_classification_prompt(text)

    try:
        result = ai_response_function(prompt)
    except Exception as error:
        raise ClassificationError(
            f"Failed to classify document: {error}"
        ) from error

    if not result or not result.strip():
        raise ClassificationError(
            "The AI returned an empty classification."
        )

    document_type = result.strip().lower()

    # Remove accidental formatting from the AI response.
    document_type = document_type.replace("`", "").strip()

    if document_type not in ALLOWED_DOCUMENT_TYPES:
        return "other"

    return document_type
