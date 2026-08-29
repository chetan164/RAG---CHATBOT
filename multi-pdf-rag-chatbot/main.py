from app.embeddings.embedding_model import EmbeddingService
from app.logging.logger import logger
from app.text_splitters.splitter import DocumentSplitter
from app.loaders.pdf_loader import PDFLoader
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrievers.retriever import RetrieverService
from app.llm.llm import LLMService
from app.prompts.prompt import PromptService
from app.chains.rag_chain import RAGChain
from pathlib import Path

from app.config.settings import (
    PROJECT_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    PDF_DIR,
    VECTOR_DB_EXISTS,
)


def main():

    logger.info("Application Started")

    print("=" * 60)
    print(PROJECT_NAME)
    print("=" * 60)

    print(f"Embedding Model : {EMBEDDING_MODEL}")
    print(f"Chunk Size      : {CHUNK_SIZE}")
    print(f"Chunk Overlap   : {CHUNK_OVERLAP}")
    print(f"Top K           : {TOP_K}")

    # -------------------------
    # Embedding Model
    # -------------------------

    embedding_service = EmbeddingService()
    embedding_model = embedding_service.get_embeddings()

    print("Embedding Model Ready")

    vector_store = FAISSVectorStore(embedding_model)

    # -------------------------
    # Load or Create Vector DB
    # -------------------------

    if VECTOR_DB_EXISTS:

        faiss_db = vector_store.load_vectorstore()

        print("Existing Vector Store Loaded")

    else:

        print("Vector DB Not Found")
        print("Creating New Vector Store...")

        # Load PDF
        loader = PDFLoader(PDF_DIR)
        documents = loader.load_documents()

        print(f"Total Pages Loaded : {len(documents)}")

        # Split
        splitter = DocumentSplitter()
        chunks = splitter.split_documents(documents)

        print(f"Total Chunks : {len(chunks)}")

        # Create Vector Store
        faiss_db = vector_store.create_vectorstore(chunks)

        # Save Vector Store
        vector_store.save_vectorstore()

        print("New Vector Store Created & Saved")

    # -------------------------
    # Retriever
    # -------------------------

    retriever_service = RetrieverService(vector_store)

    retriever = retriever_service.get_retriever(TOP_K)

    print("Retriever Search Kwargs:", retriever.search_kwargs)

    print("Retriever Ready")

    # -------------------------
    # LLM
    # -------------------------

    llm_service = LLMService()

    llm = llm_service.get_llm()

    print("LLM Ready")

    # -------------------------
    # Prompt
    # -------------------------

    prompt_service = PromptService()

    prompt = prompt_service.get_prompt()
    #memory_service = ChatMemory()

    #memory = memory_service.get_memory()

    print("Prompt Ready")

    # -------------------------
    # RAG Chain
    # -------------------------

    rag_chain = RAGChain(
        retriever,
        prompt,
        llm,
)

    print("RAG Chain Ready")

    # -------------------------
    # Chat Loop
    # -------------------------

    while True:
        question = input("\nAsk Question (type 'exit' to quit): ")

        if question.lower() == "exit":
           break

    # Retrieve Documents
        results = retriever_service.get_relevant_documents_with_score(
    question,
    TOP_K
)

        print("\nRetrieved Sources:\n")

        for i, (doc, score) in enumerate(results, start=1):

            source = Path(doc.metadata.get("source")).name
            page = doc.metadata.get("page")

            print(f"{i}. {source} (Page {page + 1})")
            similarity = 1 / (1 + score)
            print(f"Similarity : {similarity:.2%}")
            print("-" * 50)

    # Generate Answer
        print("\nGenerating Answer...")

        response = rag_chain.invoke(question)
        
        print("\nAnswer:\n")
        print(response)
        


        print("\nDone!")

if __name__ == "__main__":
    main()