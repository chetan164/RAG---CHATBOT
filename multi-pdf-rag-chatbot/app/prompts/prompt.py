from langchain_core.prompts import PromptTemplate

class PromptService:

    def __init__(self):

        template = """
You are a Question Answering Assistant.

Strict Rules:
1. Answer ONLY from the provided context.
2. If the answer is NOT explicitly present in the context, respond with exactly:

I don't know.

3. Do NOT use your own knowledge.
4. Do NOT guess.
5. Do NOT explain why you don't know.
6. Do NOT generate any additional text.

Context:
{context}

Question:
{question}

Answer:
"""

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

    def get_prompt(self):
        return self.prompt