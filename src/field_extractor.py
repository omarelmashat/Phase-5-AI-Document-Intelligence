
import json

from src.prompts import build_field_extraction_prompt


class FieldExtractionError(Exception):
    pass


def extract_fields(
    text: str,
    ai_response_function,
) -> dict:
    """
    Extract structured fields from a document.
    """

    if not text or not text.strip():
        raise ValueError("Document text cannot be empty.")

    if ai_response_function is None:
        raise ValueError("AI response function cannot be None.")

    prompt = build_field_extraction_prompt(text)

    try:
        result = ai_response_function(prompt)
    except Exception as error:
        raise FieldExtractionError(
            f"Failed to extract document fields: {error}"
        ) from error

    if not result or not result.strip():
        raise FieldExtractionError(
            "The AI returned an empty result."
        )

    cleaned_result = result.strip()

    if cleaned_result.startswith("```"):
        cleaned_result = cleaned_result.replace("```json", "", 1)
        cleaned_result = cleaned_result.replace("```", "")
        cleaned_result = cleaned_result.strip()

    try:
        extracted_fields = json.loads(cleaned_result)
    except json.JSONDecodeError as error:
        raise FieldExtractionError(
            "The AI returned invalid JSON."
        ) from error

    if not isinstance(extracted_fields, dict):
        raise FieldExtractionError(
            "Extracted fields must be returned as a JSON object."
        )

    return extracted_fields


