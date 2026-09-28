# AI Document Intelligence System

Upload a PDF, Word document, or scanned image, get a summary and key fields, and ask questions in plain language.

The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant passages from the uploaded document and generate answers based only on the retrieved content. Follow-up questions maintain their conversation context.

**TechMaster Academy — Final Phase / Graduation Project — Group 4**

## How It Works

```text
Upload
  ↓
Validate File
  ↓
Extract Text
(PDF/DOCX Parser or OCR Fallback)
  ↓
Classify Document
  ↓
Chunk
  ↓
Generate Embeddings
  ↓
Vector Store (ChromaDB)
  ↓
Retrieve Relevant Passages
  ↓
LLM Generates Grounded Answer
  ↓
Chat / Summary / Key Fields
```

## Project Structure

| File / Folder            | Purpose                                              |
| ------------------------ | ---------------------------------------------------- |
| `src/document_loader.py` | File validation and document text extraction         |
| `src/ocr.py`             | OCR processing for scanned documents                 |
| `src/chunker.py`         | Splits extracted text into chunks                    |
| `src/embeddings.py`      | Generates embeddings for document chunks and queries |
| `src/vector_store.py`    | Stores and searches embeddings using ChromaDB        |
| `src/prompts.py`         | AI prompts used by the system                        |
| `src/retriever.py`       | Retrieves relevant document passages                 |
| `src/qa_chain.py`        | Question-answering pipeline                          |
| `src/summarizer.py`      | Document summarization                               |
| `src/classifier.py`      | Document type classification                         |
| `src/field_extractor.py` | Extracts important fields from documents             |
| `src/api_client.py`      | Centralized AI/API client with retry handling        |
| `src/utils.py`           | Shared utilities and error handling                  |
| `app.py`                 | Streamlit application and user interface             |
| `config.py`              | Centralized project configuration                    |
| `tests/`                 | Automated tests                                      |
| `sample_docs/`           | Sample documents used for testing                    |

## Key Design Points

### One AI Client

All chat and embedding API calls go through:

```text
src/api_client.py
```

This provides a centralized interface for AI services and handles retries for temporary API failures.

### Centralized Configuration

Project settings are maintained in:

```text
config.py
```

Environment-specific values can be overridden using `.env`.

Important settings include:

* `CHUNK_SIZE`
* `TOP_K`
* `MAX_DISTANCE`
* Upload limits
* AI provider configuration
* Retrieval thresholds

### Graceful Error Handling

Pipeline failures are handled through `safe_call` and reported clearly in the user interface instead of causing the application to crash.

### Grounded Answers

The question-answering system uses retrieved document passages as its knowledge source.

Chunks that are farther than `MAX_DISTANCE` from the question are discarded. If no sufficiently relevant passages are found, the system returns a **"not found"** response instead of sending unrelated content to the model.

### Field Verification

Extracted field values are checked against the original document text.

A field is marked as **verified** only when its value appears in the document.

### Privacy

Uploaded files are stored temporarily and deleted after text extraction.

When a new document is uploaded, the previous document's vectors and chat context are cleared.

## Technologies Used

* **Python 3.11+**
* **Streamlit** — Web application interface
* **ChromaDB** — Vector database
* **Tesseract OCR** — OCR for scanned documents
* **PDF/DOCX processing** — Document text extraction
* **Embedding model/API** — Text embeddings
* **LLM API** — Question answering, summarization, classification, and field extraction
* **Pytest** — Automated testing
* **python-dotenv** — Environment configuration

## Setup

### Requirements

The project requires:

* **Python 3.11+**
* **Tesseract OCR** installed on the system
* Python dependencies from `requirements.txt`
* An API key for the configured AI provider

Tesseract is a system installation and is **not a pip package**.

### Windows

Install Tesseract from a Windows distribution and add it to the system PATH.

### macOS

```bash
brew install tesseract
```

### Ubuntu/Debian

```bash
sudo apt install tesseract-ocr
```

### Create a Virtual Environment

#### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

On Windows, you can also copy `.env.example` manually and rename it to `.env`.

Then add the required API key and configuration values to `.env`.

**Never commit `.env` to the repository.**

### Run the Application

```bash
streamlit run app.py
```

## How to Use

1. Start the Streamlit application.
2. Upload a supported PDF, Word document, or scanned document.
3. The system validates the uploaded file.
4. Text is extracted using a document parser or OCR when necessary.
5. The document is classified and divided into chunks.
6. Embeddings are generated and stored in ChromaDB.
7. The user can:

   * Ask questions about the document.
   * Generate a summary.
   * Extract important fields.
8. The system retrieves relevant passages before generating an answer.
9. Follow-up questions maintain the current document and conversation context.

## Tests

Run the automated test suite using:

```bash
python -m pytest -q
```

Tests use the files in:

```text
sample_docs/
```

and mocked AI responses, so they do not require an API key or internet connection.

The tests cover:

* Empty files
* Corrupted files
* Oversized uploads
* Unsupported uploads
* Scanned PDFs and OCR fallback
* Document retrieval
* "Not found" answers
* API failures and retries
* Field validation

## Settings

The main settings are available in:

```text
config.py
```

Environment-specific settings can be configured through:

```text
.env
```

The most important settings to tune are:

| Setting        | Purpose                                   |
| -------------- | ----------------------------------------- |
| `CHUNK_SIZE`   | Controls the size of document chunks      |
| `TOP_K`        | Controls the number of retrieved passages |
| `MAX_DISTANCE` | Sets the maximum retrieval distance       |

## Limits — Version 1

* One document per session
* Maximum **20 MB** upload size
* Maximum **20 PDF pages**
* Runs locally
* Temporary document storage
* Session-based vector storage
* No cloud deployment

## Project Goal

The goal of the project is to provide an AI-powered document intelligence system that combines **document processing, OCR, embeddings, vector search, and LLM-based question answering** in a single application.

The system is designed to make documents easier to understand and interact with while keeping generated answers grounded in the uploaded document.

