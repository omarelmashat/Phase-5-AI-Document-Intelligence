from src.chunker import chunk_text
from src.prompts import build_summary_prompt, build_combine_summaries_prompt

MAP_REDUCE_THRESHOLD_CHARS = 6000
SECTION_CHUNK_SIZE = 4000
SECTION_CHUNK_OVERLAP = 200


class SummarizationError(Exception):
    pass


def _summarize_once(text: str, ai_response_function) -> str:
    prompt = build_summary_prompt(text)
    try:
        summary = ai_response_function(prompt)
    except Exception as error:
        raise SummarizationError(f"Failed to generate document summary: {error}") from error
    if not summary or not summary.strip():
        raise SummarizationError("The AI returned an empty summary.")
    return summary.strip()


def summarize_document(text: str, ai_response_function) -> str:
    if not text or not text.strip():
        raise ValueError("Document text cannot be empty.")

    if ai_response_function is None:
        raise ValueError("AI response function cannot be None.")

    if len(text) <= MAP_REDUCE_THRESHOLD_CHARS:
        return _summarize_once(text, ai_response_function)

    sections = chunk_text(text, chunk_size=SECTION_CHUNK_SIZE, chunk_overlap=SECTION_CHUNK_OVERLAP)
    section_summaries = [_summarize_once(section, ai_response_function) for section in sections]
    combined = "\n\n".join(f"Section {i + 1}: {s}" for i, s in enumerate(section_summaries))

    prompt = build_combine_summaries_prompt(combined)
    try:
        final_summary = ai_response_function(prompt)
    except Exception as error:
        raise SummarizationError(f"Failed to combine section summaries: {error}") from error
    if not final_summary or not final_summary.strip():
        raise SummarizationError("The AI returned an empty summary.")
    return final_summary.strip()
