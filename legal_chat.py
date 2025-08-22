import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.memory import ConversationBufferMemory
import tempfile
import re


def load_document(uploaded_file):
    suffix = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if suffix == "pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix in ["doc", "docx"]:
        loader = UnstructuredWordDocumentLoader(tmp_path)
    elif suffix == "txt":
        loader = TextLoader(tmp_path, encoding="utf-8")
    else:
        st.error(f"Unsupported file type: {suffix}")
        return None

    docs = loader.load()
    # Filter out empty documents
    docs = [doc for doc in docs if doc.page_content.strip() != ""]
    if len(docs) == 0:
        st.warning(f"The uploaded file '{uploaded_file.name}' contains no readable text.")
    return docs

def combine_documents(docs_list):
    combined = []
    for docs in docs_list:
        combined.extend(docs)
    return combined

def build_conversational_qa_chain(documents, openai_api_key, model_name="gpt-4"):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    if len(chunks) == 0:
        st.error("Error: No chunks were created from the document. Please check your document content.")
        st.stop()

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=model_name,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a legal expert assistant. Answer user questions based only on the legal document provided. "
                   "If the answer is not in the document, respond with 'The document does not contain that information.'"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": prompt},
        get_chat_history=lambda h: h
    )

    return qa_chain

def extract_clauses(documents, keywords):
    clauses = []
    pattern = re.compile("|".join([re.escape(k) for k in keywords]), re.IGNORECASE)
    for doc in documents:
        text = doc.page_content
        if pattern.search(text):
            for match in pattern.finditer(text):
                start = max(match.start() - 150, 0)
                end = min(match.end() + 150, len(text))
                snippet = text[start:end].strip()
                clauses.append(snippet)
    return clauses



st.set_page_config(page_title="Legal Document Assistant - MultiDoc + Chat + Clauses", layout="wide")
st.title("Legal Document Assistant")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
model_name = st.sidebar.selectbox("Choose OpenAI Model", ["gpt-4", "gpt-3.5-turbo"])

uploaded_files = st.file_uploader(
    " Upload one or more legal documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files and openai_api_key and openai_api_key.startswith("sk-"):
    with st.spinner(" Loading documents..."):
        docs_list = []
        for file in uploaded_files:
            loaded = load_document(file)
            if loaded:
                docs_list.append(loaded)
        combined_docs = combine_documents(docs_list)

    if len(combined_docs) == 0:
        st.warning("No readable text found in the uploaded documents.")
        st.stop()

    current_doc_names = [file.name for file in uploaded_files]
    if ("qa_chain" not in st.session_state) or (st.session_state.get("doc_names") != current_doc_names):
        st.session_state.qa_chain = build_conversational_qa_chain(combined_docs, openai_api_key, model_name)
        st.session_state.doc_names = current_doc_names
        st.session_state.combined_docs = combined_docs
        st.session_state.chat_history = []

    st.success(f"Loaded {len(combined_docs)} document chunks from {len(uploaded_files)} files.")

    st.sidebar.markdown("### 🔎 Clause Extraction")
    clause_keywords = st.sidebar.text_area(
        "Enter keywords for clause extraction (comma separated):",
        value="termination,liability,confidentiality,indemnity"
    )
    keywords_list = [k.strip() for k in clause_keywords.split(",") if k.strip()]

    if st.sidebar.button("Extract Clauses"):
        with st.spinner(" Extracting clauses..."):
            clauses = extract_clauses(st.session_state.combined_docs, keywords_list)
            if clauses:
                st.sidebar.markdown(f"### Found {len(clauses)} clause snippets:")
                for i, clause in enumerate(clauses, 1):
                    st.sidebar.markdown(f"**Clause {i}:**")
                    st.sidebar.write(clause)
            else:
                st.sidebar.info("No clauses found for the given keywords.")

    user_question = st.text_input("Ask a question about the uploaded documents:")

    if user_question:
        with st.spinner("Generating answer..."):
            result = st.session_state.qa_chain({"question": user_question})
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            st.session_state.chat_history.append({"question": user_question, "answer": answer})

        st.markdown("### Answer:")
        st.write(answer)

        with st.expander("Chat History", expanded=True):
            for i, chat in enumerate(st.session_state.chat_history[::-1]):
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.markdown("---")

        if source_docs:
            st.markdown("---")
            st.markdown("### Source Document Chunks:")
            for i, doc in enumerate(source_docs):
                page = doc.metadata.get("page", "N/A")
                st.write(f"**Chunk {i+1}** (Page {page}):")
                st.code(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

else:
    if not openai_api_key:
        st.sidebar.warning("Please enter your OpenAI API key.", icon="")
    elif uploaded_files and not openai_api_key.startswith("sk-"):
        st.sidebar.warning("Invalid OpenAI API key format.", icon="")
    else:
        st.info("Upload legal documents and enter your OpenAI API key to get started.")
