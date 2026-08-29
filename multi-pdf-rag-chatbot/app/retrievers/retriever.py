
from langchain_core.vectorstores import VectorStoreRetriever
from app.vectorstore.faiss_store import FAISSVectorStore


class RetrieverService:

    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store

    def get_retriever(self, k=3) -> VectorStoreRetriever:

        return self.vector_store.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": 10,
                "lambda_mult": 0.5,
            },
        )

    def get_relevant_documents_with_score(self, question, k=3):

        return self.vector_store.vectorstore.similarity_search_with_score(
            question,
            k=k,
        )
