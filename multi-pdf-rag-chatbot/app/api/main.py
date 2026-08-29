from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.embeddings.embedding_model import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrievers.retriever import RetrieverService
from app.llm.llm import LLMService
from app.prompts.prompt import PromptService
from app.chains.rag_chain import RAGChain
from app.config.settings import (
    TOP_K,
    VECTOR_DB_EXISTS,
)

app = FastAPI(
    title="Multi PDF RAG Chatbot API",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str


# --------------------------------------------------
# RAG Initialization
# --------------------------------------------------

rag_chain = None
retriever_service = None


def initialize_rag():

    global rag_chain
    global retriever_service

    print("Initializing RAG...")

    # Embedding
    embedding_service = EmbeddingService()
    embedding_model = embedding_service.get_embeddings()

    print("Embedding Model Ready")

    # Vector Store
    vector_store = FAISSVectorStore(embedding_model)

    if not VECTOR_DB_EXISTS:
        raise RuntimeError(
            "FAISS vector database does not exist. "
            "Create the vector database first using your existing main.py."
        )

    vector_store.load_vectorstore()

    print("Vector Store Loaded")

    # Retriever
    retriever_service = RetrieverService(vector_store)

    retriever = retriever_service.get_retriever(TOP_K)

    print("Retriever Ready")

    # LLM
    llm_service = LLMService()
    llm = llm_service.get_llm()

    print("LLM Ready")

    # Prompt
    prompt_service = PromptService()
    prompt = prompt_service.get_prompt()

    print("Prompt Ready")

    # RAG Chain
    rag_chain = RAGChain(
        retriever,
        prompt,
        llm,
    )

    print("RAG Chain Ready")


# --------------------------------------------------
# Startup
# --------------------------------------------------

@app.on_event("startup")
def startup_event():
    initialize_rag()


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "message": "RAG API is running",
    }


# --------------------------------------------------
# Chat Endpoint
# --------------------------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    if rag_chain is None:
        return {
            "error": "RAG chain is not initialized"
        }

    question = request.question.strip()

    if not question:
        return {
            "error": "Question cannot be empty"
        }

    response = rag_chain.invoke(question)

    sources = []

    for doc in response["sources"]:

        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        sources.append(
            {
                "source": Path(source).name if source else None,
                "page": page + 1 if page is not None else None,
            }
        )

    return {
        "answer": response["answer"],
        "sources": sources,
    }