import os
from pathlib import Path

import streamlit as st

from app.embeddings.embedding_model import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrievers.retriever import RetrieverService
from app.prompts.prompt import PromptService
from app.llm.llm import LLMService
from app.chains.rag_chain import RAGChain
from app.loaders.pdf_loader import PDFLoader
from app.config.settings import TOP_K, PDF_DIR
from app.text_splitters.splitter import DocumentSplitter

# =========================================================
# Streamlit Page Config
# =========================================================

st.set_page_config(
    page_title="Enterprise Multi PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# =========================================================
# Claude-like CSS Theme
# =========================================================

st.markdown("""
<style>
    /* Overall page */
    .stApp {
        background-color: #FAF9F6;
    }

    /* Center the whole chat like Claude's centered column */
    .block-container {
        max-width: 780px;
        padding-top: 2.5rem;
        padding-bottom: 8rem;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Title */
    .chat-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2D2A26;
        margin-bottom: 0.2rem;
    }
    .chat-subtitle {
        font-size: 0.92rem;
        color: #8A8578;
        margin-bottom: 1.8rem;
    }

    /* Chat message bubbles */
    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.4rem 0;
    }

    /* User bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) {
        border-radius: 14px;
    }

    /* Chat input container — pinned bottom, rounded like Claude */
    div[data-testid="stChatInput"] {
        max-width: 780px;
        margin: 0 auto;
        border-radius: 22px;
        border: 1px solid #E3E0D8;
        background-color: #FFFFFF;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    div[data-testid="stChatInput"] textarea {
        font-size: 0.95rem;
    }

    /* Uploaded file pill styling */
    .file-pill {
        display: inline-block;
        background-color: #F0EEE6;
        color: #5C5850;
        border-radius: 12px;
        padding: 4px 12px;
        font-size: 0.82rem;
        margin: 4px 6px 0 0;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        margin-top: 8rem;
        color: #A39E90;
    }
    .empty-state h2 {
        font-size: 1.4rem;
        color: #4A463F;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Cache Embedding Model / LLM / Vector Store
# =========================================================

@st.cache_resource
def load_embedding_model():
    embedding_service = EmbeddingService()
    return embedding_service.get_embeddings()


@st.cache_resource
def load_llm():
    llm_service = LLMService()
    return llm_service.get_llm()


@st.cache_resource
def load_vector_store(_embedding_model):
    vector_store = FAISSVectorStore(_embedding_model)
    vector_store.load_vectorstore()
    return vector_store


embedding_model = load_embedding_model()
vector_store = load_vector_store(embedding_model)
llm = load_llm()

retriever = RetrieverService(vector_store).get_retriever(TOP_K)
prompt = PromptService().get_prompt()
rag_chain = RAGChain(retriever, prompt, llm)

# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()

# =========================================================
# Header
# =========================================================

st.markdown('<div class="chat-title">🤖 Enterprise Multi PDF RAG Chatbot</div>', unsafe_allow_html=True)

# =========================================================
# Empty State (no messages yet)
# =========================================================

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h2>How can I help you today?</h2>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# Display Chat History
# =========================================================

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# =========================================================
# Chat Input (with built-in attach icon, pinned to bottom)
# =========================================================

prompt_input = st.chat_input(
    "Ask your question...",
    accept_file="multiple",
    file_type=["pdf"],
)

if prompt_input:
    user_text = prompt_input.text
    uploaded_files = prompt_input.files if prompt_input.files else []

    # ---------------------------------------------------
    # Handle Uploaded PDFs
    # ---------------------------------------------------
    if uploaded_files:
        pdf_dir = Path(PDF_DIR)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        new_files = []

        for uploaded_file in uploaded_files:
            save_path = pdf_dir / uploaded_file.name

            if uploaded_file.name in st.session_state.uploaded_names:
                continue

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.uploaded_names.add(uploaded_file.name)
            new_files.append(uploaded_file.name)

        if new_files:
            with st.spinner("Indexing uploaded PDF(s)..."):
                loader = PDFLoader(PDF_DIR)
                documents = loader.load_documents()

                splitter = DocumentSplitter()
                chunks = splitter.split_documents(documents)

                vector_store.load_vectorstore()
                vector_store.add_documents(chunks)
                vector_store.save_vectorstore()

            pills = "".join(f'<span class="file-pill">📎 {name}</span>' for name in new_files)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Added {len(new_files)} file(s) to the knowledge base:<br>{pills}",
            })

    # ---------------------------------------------------
    # Handle User Question
    # ---------------------------------------------------
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_text)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                result = rag_chain.invoke(user_text)
                st.markdown(result["answer"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
        })
    else:
        st.rerun()