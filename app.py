
import hashlib
import os
import uuid

import streamlit as st

import config
from src import utils
from src.api_client import get_ai_response, embed_texts
from src.chunker import chunk_document
from src.classifier import classify_document
from src.document_loader import load_document
from src.field_extractor import extract_fields
from src.qa_chain import answer_question
from src.summarizer import summarize_document
from src.vector_store import DocumentVectorStore

st.set_page_config(page_title="AI Document Intelligence", page_icon="📄", layout="wide")

STATE_DEFAULTS = {
    "file_hash": None, "session_dir": None, "text": None, "doc_type": None,
    "store": None, "history": [], "summary": None, "fields": None,
}


def init_state():
    for key, value in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, value if not isinstance(value, list) else [])


def reset_document():

    utils.cleanup_session_dir(st.session_state.get("session_dir"))
    store = st.session_state.get("store")
    if store is not None:
        store.reset()
    for key, value in STATE_DEFAULTS.items():
        st.session_state[key] = value if not isinstance(value, list) else []


def process_upload(uploaded) -> None:
    data = uploaded.getvalue()
    reset_document()
    st.session_state.file_hash = hashlib.sha256(data).hexdigest()
    session_dir = utils.create_session_dir()
    st.session_state.session_dir = session_dir
    path = utils.save_upload(session_dir, uploaded.name, data)

    with st.status("Processing document...", expanded=True) as status:
        st.write("Reading text (OCR is used automatically for scans)...")
        text, error = utils.safe_call(load_document, path, uploaded.name,
                                      error_message="Could not read this file.")
        if error:
            status.update(label="Could not process the file", state="error")
            st.error(error)
            utils.cleanup_session_dir(session_dir)
            st.session_state.session_dir = None
            st.session_state.file_hash = None
            return
        st.session_state.text = text

        st.write("Identifying document type...")
        doc_type, error = utils.safe_call(
            classify_document, text[:config.CLASSIFY_CHARS], get_ai_response,
            default="other", error_message="Could not classify the document.")
        st.session_state.doc_type = doc_type
        if error:
            st.warning(error)

        st.write("Indexing for search...")
        chunks = chunk_document(text, source_id=os.path.splitext(uploaded.name)[0],
                                chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        store = DocumentVectorStore(collection_name=f"doc_{uuid.uuid4().hex[:12]}")
        vectors, error = utils.safe_call(
            utils.embed_in_batches, embed_texts, [c["text"] for c in chunks],
            error_message="The embedding service failed, so questions can't be answered right now.")
        if error:
            status.update(label="Indexing failed", state="error")
            st.error(error)
        else:
            store.add_chunks(chunks, vectors)
            st.session_state.store = store
            status.update(label=f"Ready - {len(chunks)} passages indexed", state="complete")
    utils.cleanup_session_dir(session_dir)
    st.session_state.session_dir = None


def render_summary():
    if st.button("Generate summary"):
        with st.spinner("Summarizing..."):
            text = st.session_state.text[:config.MAX_PROMPT_CHARS]
            summary, error = utils.safe_call(summarize_document, text, get_ai_response,
                                             error_message="Could not generate the summary.")
        if error:
            st.error(error)
        else:
            st.session_state.summary = summary
    if st.session_state.summary:
        st.markdown(st.session_state.summary)


def render_fields():
    if st.button("Extract key fields"):
        with st.spinner("Extracting..."):
            text = st.session_state.text[:config.MAX_PROMPT_CHARS]
            fields, error = utils.safe_call(extract_fields, text, get_ai_response,
                                            error_message="Could not extract fields.")
        if error:
            st.error(error)
        else:
            st.session_state.fields = utils.validate_fields(fields, st.session_state.text)
    checked = st.session_state.fields
    if checked is not None:
        if not checked:
            st.info("No structured fields were found in this document.")
        for name, item in checked.items():
            icon = "✅" if item["verified"] else "⚠️"
            st.write(f"{icon} **{name}**: {item['value']}")
        if any(not i["verified"] for i in checked.values()):
            st.caption("⚠️ = value not found word-for-word in the document. Double-check it.")


def render_chat():
    if st.session_state.store is None:
        st.warning("The document isn't indexed, so questions are unavailable. Try uploading again.")
        return
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about this document")
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    ask = utils.with_history(get_ai_response, st.session_state.history)
    guarded_store = utils.ThresholdedStore(st.session_state.store)
    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            result, error = utils.safe_call(
                answer_question, question, guarded_store, ask, top_k=config.TOP_K,
                error_message="Sorry, I couldn't get an answer just now. Please try again.")
        answer = error if error else result["answer"]
        st.markdown(answer)
        if result and result["sources"]:
            with st.expander("Sources"):
                for src in result["sources"]:
                    st.caption(f"passage {src['chunk_index']} (distance {src['distance']:.2f})")
    if not error:
        st.session_state.history += [{"role": "user", "content": question},
                                     {"role": "assistant", "content": answer}]


def main():
    init_state()
    st.title("📄 AI Document Intelligence")
    st.caption("Upload a PDF, Word file or scanned image, then ask questions about it. "
               "Answers come only from your document. Files are deleted after processing.")

    if not (config.COHERE_API_KEY if config.AI_PROVIDER == "cohere" else config.HF_API_KEY):
        st.error("No API key found. Copy `.env.example` to `.env` and add your key, then restart.")
        st.stop()

    uploaded = st.file_uploader("Upload a document", type=[e.lstrip(".") for e in sorted(config.ALLOWED_EXTENSIONS)])
    if uploaded is not None:
        if hashlib.sha256(uploaded.getvalue()).hexdigest() != st.session_state.file_hash:
            process_upload(uploaded)
    elif st.session_state.file_hash:
        reset_document()  # user removed the file

    if not st.session_state.text:
        st.info("No document loaded yet.")
        return

    st.success(f"Document type: **{st.session_state.doc_type}**")
    if st.sidebar.button("Clear conversation"):
        st.session_state.history = []
    tab_chat, tab_summary, tab_fields = st.tabs(["💬 Ask", "📝 Summary", "🔎 Key fields"])
    with tab_chat:
        render_chat()
    with tab_summary:
        render_summary()
    with tab_fields:
        render_fields()


main()
