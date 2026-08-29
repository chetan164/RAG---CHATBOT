from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import LLM_MODEL
from app.logging.logger import logger

load_dotenv()


class LLMService:

    def __init__(self):

        logger.info("Loading Gemini LLM...")

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
            max_output_tokens=512,
        )

        logger.info("LLM Loaded Successfully")

    def get_llm(self):
        return self.llm