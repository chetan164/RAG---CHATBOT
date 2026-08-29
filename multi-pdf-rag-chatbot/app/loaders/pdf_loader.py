from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from app.logging.logger import logger


class PDFLoader:
    """
    Load all PDF files from a folder.
    """

    def __init__(self, pdf_directory: Path):
        self.pdf_directory = pdf_directory

    def load_documents(self) -> list[Document]:

        documents = []

        pdf_files = list(self.pdf_directory.glob("*.pdf"))

        if len(pdf_files) == 0:
            logger.warning("No PDF Found")
            return documents

        for pdf in pdf_files:
            print(f"Loading: {pdf.name}")


            logger.info(f"Loading {pdf.name}")

            loader = PyMuPDFLoader(str(pdf))

            docs = loader.load()

            documents.extend(docs)

        logger.info(f"Total Pages : {len(documents)}")

        return documents