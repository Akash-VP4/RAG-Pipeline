from src.vector_store import ChromaVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv


class RAGSearch:

    def __init__(self, vector_store: ChromaVectorStore):
        """Initialize vector_store, call initialize_llm()"""

        self.vector_store = vector_store
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """This will load llm"""
        try:
            load_dotenv()
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY")
            )
            print(f"[DEBUG]: loaded llm: {self.llm}")

        except Exception as e:
            print(f"[ERROR]: Failed to load llm:{e}")

    def search(self, query: str):
        """This will search for context in vector db and with llm generate response

        Args:
          query: quetion

        """

        results = self.vector_store.search(query)
        print("[DEBUG]: Context extracted")
        context: str = (
            "\n\n".join([doc for doc in results["documents"][0]]) if results else ""
        )

        if not context:
            return "No relevent context found!"

        prompt = f""" Anwer for the following qution by using the context given below
                
                \n Context:{context}

                \n question:{query}
        """

        response = self.llm.invoke([prompt.format(context=context, query=query)])

        return response.content
