from langchain_huggingface import HuggingFaceEmbeddings

from app.config.settings import EMBEDDING_MODEL
from app.logging.logger import logger


class EmbeddingService:
    """
    Creates HuggingFace Embedding model.
    """

    def __init__(self):

        logger.info("Loading Embedding Model...")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        logger.info("Embedding Model Loaded Successfully")

    def get_embeddings(self):

        return self.embedding_model