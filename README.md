Multi-PDF RAG Chatbot 🤖

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload one or more PDF documents and ask natural-language questions about their content. Answers are generated strictly from the retrieved context — if the answer isn't in your documents, the bot says "I don't know." instead of guessing.

Built with LangChain, HuggingFace sentence-transformer embeddings, a FAISS vector store, and Google Gemini as the LLM. Ships with three ways to use it: a CLI, a FastAPI backend, and a Streamlit chat UI.

✨ Features
📄 Multi-PDF ingestion — loads every PDF in a data folder (via PyMuPDF) or lets you upload new ones straight from the chat UI
✂️ Smart chunking — RecursiveCharacterTextSplitter splits documents into overlapping chunks for better retrieval
🔎 FAISS vector store — persists locally, supports incremental additions without rebuilding from scratch
🎯 MMR retrieval — Maximal Marginal Relevance search for diverse, non-redundant context, plus similarity-scored source lookup
🧠 Grounded answers — a strict prompt template forces the LLM to answer only from retrieved context
🖥️ Three interfaces:
main.py — interactive command-line chat loop
app/api/main.py — FastAPI REST API (/chat, /health)
streamlit_app.py — polished, Claude-style chat UI with drag-and-drop PDF uploads
📝 Logging built in via a shared logger
🏗️ Project Structure
multi-pdf-rag-chatbot/
├── app/
│   ├── api/
│   │   └── main.py            # FastAPI app (chat + health endpoints)
│   ├── chains/
│   │   └── rag_chain.py       # LangChain RAG pipeline (retrieve → prompt → LLM)
│   ├── config/
│   │   └── settings.py        # Loads configuration from .env
│   ├── embeddings/
│   │   └── embedding_model.py # HuggingFace embedding wrapper
│   ├── llm/
│   │   └── llm.py             # Google Gemini LLM wrapper
│   ├── loaders/
│   │   └── pdf_loader.py      # Loads all PDFs from a directory
│   ├── logging/
│   │   └── logger.py          # Shared application logger
│   ├── prompts/
│   │   └── prompt.py          # Strict context-only prompt template
│   ├── retrievers/
│   │   └── retriever.py       # MMR retriever + similarity-scored search
│   ├── text_splitters/
│   │   └── splitter.py        # Recursive character chunking
│   └── vectorstore/
│       └── faiss_store.py     # Create / load / save / update FAISS index
├── data/
│   ├── pdfs/                  # Source PDF documents go here
│   └── vector_db/             # Persisted FAISS index (auto-generated)
├── logs/
│   └── app.log
├── main.py                    # CLI entry point
├── streamlit_app.py           # Streamlit chat UI entry point
└── requirements.txt
⚙️ Requirements
Python 3.10+
A Google Gemini API key (used for the LLM)

Key dependencies (see requirements.txt for the full list):

langchain, langchain-community, langchain-core, langchain-huggingface, langchain-text-splitters
langchain-google-genai
faiss-cpu
sentence-transformers, transformers, accelerate, torch
pymupdf
fastapi, uvicorn
streamlit
python-dotenv, pydantic
🚀 Setup
Clone the repository
bash
   git clone https://github.com/chetan164/RAG---CHATBOT.git
   cd RAG---CHATBOT/multi-pdf-rag-chatbot
Create a virtual environment and install dependencies
bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
Create a .env file in multi-pdf-rag-chatbot/ with the following variables:
env
   PROJECT_NAME=Multi PDF RAG Chatbot

   # LLM
   LLM_MODEL=gemini-1.5-flash
   GEMINI_API_KEY=your_google_gemini_api_key

   # Embeddings
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

   # Chunking
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200

   # Retrieval
   TOP_K=3

   # Paths (relative to project root)
   PDF_DIR=data/pdfs
   VECTOR_DB=data/vector_db
Add your PDFs to data/pdfs/ (skip this if you plan to upload files through the Streamlit UI instead).
▶️ Usage
Option 1 — CLI
bash
python main.py

On first run (no existing vector DB), it loads and chunks every PDF in data/pdfs/, builds the FAISS index, and saves it to data/vector_db/. On later runs, it loads the saved index directly. Type your question at the prompt, or exit to quit.

Option 2 — Streamlit chat UI
bash
streamlit run streamlit_app.py

Open the local URL Streamlit prints (usually http://localhost:8501). You can chat right away if a vector store already exists, or drag-and-drop new PDFs into the chat input to add them to the knowledge base on the fly.

Option 3 — FastAPI backend
bash
uvicorn app.api.main:app --reload

Requires a vector store to already exist (created via main.py or the Streamlit app first).

Endpoints:

Method	Endpoint	Description
GET	/health	Health check
POST	/chat	Ask a question, returns answer + sources

Example request:

bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'

Example response:

json
{
  "answer": "...",
  "sources": [
    { "source": "example.pdf", "page": 3 }
  ]
}
🧠 How It Works
Load — all PDFs in data/pdfs/ are parsed page-by-page with PyMuPDF.
Split — pages are recursively split into overlapping chunks (CHUNK_SIZE / CHUNK_OVERLAP).
Embed — chunks are embedded using a HuggingFace sentence-transformer model.
Store — embeddings are indexed in a local FAISS vector store and persisted to disk.
Retrieve — on each question, the retriever pulls the top-k most relevant chunks using MMR search for diversity.
Generate — the retrieved context and question are passed to Gemini through a strict prompt that only allows context-grounded answers.
📌 Notes
If the vector store already exists, both the CLI and API load it instead of rebuilding — delete data/vector_db/ to force a fresh rebuild.
The Streamlit app supports incrementally adding new PDFs without rebuilding the whole index.
The prompt is intentionally strict: if the answer isn't in the retrieved context, the model responds with I don't know. rather than hallucinating.
