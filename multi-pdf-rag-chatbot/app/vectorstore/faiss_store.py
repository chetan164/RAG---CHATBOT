from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config.settings import VECTOR_DB_DIR
from app.logging.logger import logger


class FAISSVectorStore:
    """
    Creates and manages the FAISS vector database.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vectorstore = None

    # -------------------------
    # Create New Vector Store
    # -------------------------

    def create_vectorstore(self, documents: list[Document]):

        logger.info("Creating FAISS Vector Store...")

        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model,
        )

        logger.info("FAISS Vector Store Created Successfully")

        return self.vectorstore

    # -------------------------
    # Add Documents
    # -------------------------

    def add_documents(self, documents: list[Document]):

        logger.info("Adding New Documents To Existing Vector Store...")

        self.vectorstore.add_documents(documents)

        logger.info("Documents Added Successfully")

    # -------------------------
    # Save Vector Store
    # -------------------------

    def save_vectorstore(self):

        logger.info("Saving FAISS Vector Store...")

        self.vectorstore.save_local(str(VECTOR_DB_DIR))

        logger.info("Vector Store Saved")

    # -------------------------
    # Load Existing Vector Store
    # -------------------------

    def load_vectorstore(self):

        logger.info("Loading Existing Vector Store...")

        self.vectorstore = FAISS.load_local(
            str(VECTOR_DB_DIR),
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )

        logger.info("Vector Store Loaded")

        return self.vectorstore