import base64
import os
from io import BytesIO
import asyncio
from typing import Any
import bs4
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_agent
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.runtime import Runtime
from pypdf import PdfReader
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.messages import HumanMessage

load_dotenv()

model=init_chat_model("claude-sonnet-4-5-20250929", temperature=0.0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = InMemoryVectorStore(embeddings)

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

def _base64_to_pdf(file):
    base64_data = file.split("base64,", 1)[1]
    pdf_bytes = base64.b64decode(base64_data)
    pdf_file = BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    return reader.pages


@before_agent 
async def load_pdfs(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state['messages'][-1].content

    if isinstance(last_message, str):
        return None

    if isinstance(last_message, list):
        files = [item.get("file").get("file_data") for item in last_message if item.get("type") == "file" and item.get("file").get("file_data") != None]

        docs = []
        for file in files:
            # TODO: check it, maybe it is not necessary
            pages = await asyncio.to_thread(_base64_to_pdf, file)         
            for page_num, page in enumerate(pages):
                text = page.extract_text()
                docs.append(Document(page_content=text, metadata={"page": page_num, "source": "base64_pdf"}))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs[::20]) # first 20
        doc_ids = vector_store.add_documents(documents=chunks)

        return_message = next((item["text"] for item in last_message if item.get("type") == "text"), None)
        
        # TODO: maybe add ai message, pdf is loaded, update retriver to retrive only source related data by filter
        return {"messages": state['messages'][:-1] + [HumanMessage(content=return_message)]}

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

system_prompt = (
    "You have access to a tool that retrieves context from documents."
    "Always use the tool to help answer user queries."
)

system_prompt_tool = """
    You are a helpful assistant.
    You have access to a tool for loading PDF from user message.
    If user explicitly ask for load PDF use tool. Otherwise ignore PDF in messages.
    You have access to a tool that retrieves context from loaded PDF documents.
    If user explicitly ask to get context from tool call retrieves context tool. 
    Otherwise do not use retrieves context tool.
"""

agent = create_agent(
    model=model,
    tools=[retrieve_context],
    # middleware=
    #     [
    #         load_pdfs,
    #         ToolCallLimitMiddleware(
    #             tool_name="retrieve_context",
    #             run_limit=3,
    #         )
    #     ],
    system_prompt=system_prompt
)
