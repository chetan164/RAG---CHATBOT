from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain:

    def __init__(self, retriever, prompt, llm):

        self.retriever = retriever

        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, question):

        # Retrieve documents for source information
        docs = self.retriever.invoke(question)

        # Generate answer using retrieved context + Gemini
        answer = self.chain.invoke(question)

        return {
            "answer": answer,
            "sources": docs,
        }