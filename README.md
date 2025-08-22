# Legal Document Assistant

A powerful Streamlit application for interactive Q&A and clause extraction from multiple legal documents (PDF, DOCX, and TXT). This tool leverages OpenAI's GPT models and LangChain to enable efficient document understanding, clause discovery, and conversational exploration of legal materials.

---

## Features

- **Multi-document Upload:** Load and analyze several legal documents at once (PDF, DOCX, TXT).
- **Conversational Q&A:** Ask questions in natural language and receive answers grounded in the uploaded documents.
- **Clause Extraction:** Find and extract key legal clauses (e.g., termination, liability) using customizable keyword search.
- **Source Tracking:** See which document snippets were used to answer your question.
- **Chat History:** Review your previous questions and answers within the session.
- **Secure:** Your OpenAI API key is used securely and never stored.

---

## How It Works

1. **Upload Documents:** Add one or more legal documents via the web interface.
2. **Enter OpenAI API Key:** Provide your OpenAI API key for access to GPT-3.5 or GPT-4.
3. **Ask Questions:** Type legal questions related to the uploaded documents and receive precise, document-based answers.
4. **Extract Clauses:** Enter keywords (e.g., "confidentiality", "indemnity") to extract relevant clauses from all documents.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/legal-document-assistant.git
cd legal-document-assistant
```

### 2. Install Requirements

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

#### Requirements include:
- `streamlit`
- `langchain`
- `openai`
- `faiss-cpu`
- `unstructured`
- `pypdf`
- `python-docx`

### 3. Set up your OpenAI API Key

You can obtain an API key from [OpenAI Dashboard](https://platform.openai.com/account/api-keys).

You will be prompted in the sidebar to enter your API key when you run the app.

### 4. Run the Application

```bash
streamlit run legal_chat.py
```

### 5. Open in your browser

Navigate to the local address displayed in your terminal (e.g., http://localhost:8501).

---

## Usage

- **Document Upload:** Drag and drop your legal files in the upload widget.
- **Clause Extraction:** Input keywords (comma-separated) to extract relevant clauses from documents.
- **Q&A:** Type questions about the documents in the chat box and receive precise, context-based answers.

---
