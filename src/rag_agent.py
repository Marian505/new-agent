import os
import bs4
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = InMemoryVectorStore(embeddings)


class RagAgentState(AgentState):
    pass

# TODO: loading WIP, add load by tool
def load_web(web_paths: tuple[str]) -> int:
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=web_paths,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_docs = splitter.split_documents(docs)

    doc_ids = vector_store.add_documents(documents=all_docs)
    return len(doc_ids)

# TODO: loading WIP, add load by tool
def load_pdfs(pdf_paths: list[str]) -> int:
    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["pdf_id"] = os.path.basename(pdf_path)
        all_docs.extend(chunks)

    doc_ids = vector_store.add_documents(documents=all_docs)
    return len(doc_ids)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information related to the query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Id:{doc.id}\nSource:{doc.metadata}\nContent:\n{doc.page_content}"
        for doc in retrieved_docs
    )
    print(f"retrieve_context query: {query}")
    return serialized, retrieved_docs

prompt = (
    "You have access to a tool that retrieves context from documents."
    "Always use the tool to help answer user queries."
)

agent = create_agent(
    model=init_chat_model("claude-sonnet-4-5-20250929", model_provider="anthropic", temperature=0.0),
    tools=[retrieve_context],
    system_prompt=prompt
)
