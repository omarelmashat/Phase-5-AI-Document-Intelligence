"""
Prompt templates for the AI Document Intelligence System.
"""


# =========================================================
# Q&A PROMPT
# =========================================================

QA_SYSTEM_PROMPT = """
You are an AI document assistant.

Use ONLY the information provided in the document context.

Rules:
1. Do not use outside knowledge.
2. Do not guess or invent facts.
3. If the answer is not in the context, say:
"I couldn't find this information in the document."
4. Give a clear and concise answer.
"""


def build_qa_prompt(question: str, context: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context or not context.strip():
        raise ValueError("Document context cannot be empty.")

    return (
        QA_SYSTEM_PROMPT
        + "\n\nDOCUMENT CONTEXT:\n"
        + context.strip()
        + "\n\nUSER QUESTION:\n"
        + question.strip()
        + "\n\nANSWER:\n"
    )


# =========================================================
# SUMMARIZATION PROMPT
# =========================================================

SUMMARY_SYSTEM_PROMPT = """
You are an AI document summarization assistant.

Summarize ONLY information contained in the document.
Do not add outside information or invent facts.
"""


def build_summary_prompt(context: str) -> str:
    if not context or not context.strip():
        raise ValueError("Document context cannot be empty.")

    return (
        SUMMARY_SYSTEM_PROMPT
        + "\n\nDOCUMENT:\n"
        + context.strip()
        + "\n\nSUMMARY:\n"
    )


# =========================================================
# CLASSIFICATION PROMPT
# =========================================================

CLASSIFICATION_SYSTEM_PROMPT = """
You are a document classification assistant.

Classify the document into exactly ONE category:

contract
invoice
report
resume
letter
legal_document
financial_document
academic_document
other

Use only evidence from the document.
If it does not clearly match, return "other".
"""


def build_classification_prompt(context: str) -> str:
    if not context or not context.strip():
        raise ValueError("Document context cannot be empty.")

    return (
        CLASSIFICATION_SYSTEM_PROMPT
        + "\n\nDOCUMENT:\n"
        + context.strip()
        + "\n\nDOCUMENT TYPE:\n"
    )


# =========================================================
# FIELD EXTRACTION PROMPT
# =========================================================

FIELD_EXTRACTION_SYSTEM_PROMPT = """
You are an information extraction assistant.

Extract only information explicitly present in the document.

Possible fields include:
names, dates, amounts, organizations, parties,
addresses, invoice numbers, contract numbers,
email addresses, and phone numbers.

If a field is not present, use null.
Return valid JSON.
"""


def build_field_extraction_prompt(context: str) -> str:
    if not context or not context.strip():
        raise ValueError("Document context cannot be empty.")

    return (
        FIELD_EXTRACTION_SYSTEM_PROMPT
        + "\n\nDOCUMENT:\n"
        + context.strip()
        + "\n\nEXTRACTED FIELDS:\n"
    )

