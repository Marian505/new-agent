
from bs4 import SoupStrainer
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters.base import TextSplitter

load_dotenv()

class SimpleRagAgent:
    def __init__(
        self,
        model: BaseChatModel = None,
        embeddings: GoogleGenerativeAIEmbeddings= None,
        vector_store: VectorStore = None,
        splitter: TextSplitter = None,
        system_prompt: str = """
        You are a helpful assistant.
        You have access to a tool that retrieves context from documents.
        Always use the tool to answer user question.
        """
    ):
        self._model = model or ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.0)
        self._embeddings = embeddings or GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self._vector_store = vector_store or InMemoryVectorStore(self._embeddings)
        self._splitter = splitter or RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self._system_prompt = system_prompt

    async def load_data(self, web_paths: list[str]) -> int:
        bs4_strainer = SoupStrainer(
            class_=("post-title", "post-header", "post-content")
        )
        loader = WebBaseLoader(
            web_paths=web_paths,
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()
        all_docs = self._splitter.split_documents(docs)
        doc_ids = await self._vector_store.aadd_documents(documents=all_docs)
        return len(doc_ids)

    def get_agent(self):

        @tool
        async def retrieve_context(query: str) -> str:
            """Retrieve information related to the query."""
            retrieved_docs = await self._vector_store.asimilarity_search(query, k=2)
            serialized = "\n\n".join(
                f"Source:{doc.metadata}\nContent:\n{doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized

        return create_agent(
            model=self._model,
            tools=[retrieve_context],
            system_prompt=self._system_prompt,
        )

simpleRagAgent = SimpleRagAgent()
agent = simpleRagAgent.get_agent()