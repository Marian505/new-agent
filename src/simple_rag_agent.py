
import bs4
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class SimpleRagAgent:
    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        self._model = model
        self._llm = init_chat_model(model, temperature=0.0)
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self._vector_store = InMemoryVectorStore(self._embeddings)
        self._system_prompt = """
            You are a helpful assistant.
            You have access to a tool that retrieves context from documents.
            Always use the tool to answer user question.
        """

    def load_data(self, web_paths: tuple[str]) -> int:
        bs4_strainer = bs4.SoupStrainer(
            class_=("post-title", "post-header", "post-content")
        )
        loader = WebBaseLoader(
            web_paths=web_paths,
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_docs = splitter.split_documents(docs)

        doc_ids = self._vector_store.add_documents(documents=all_docs)
        return len(doc_ids)

    def get_agent(self):
        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information related to the query."""
            retrieved_docs = self._vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                f"Source:{doc.metadata}\nContent:\n{doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        return create_agent(
            model=self._model,
            tools=[retrieve_context],
            system_prompt=self._system_prompt,
        )
