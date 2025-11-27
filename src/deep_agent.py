import os
from typing import Literal

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_community.agent_toolkits import FileManagementToolkit
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
file_tools = FileManagementToolkit(root_dir=str(os.getcwd() + "/files")).get_tools()

def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

subagents = [
    {
        "name": "web-search-agent",
        "description": "Agent with access to websearch tool",
        "system_prompt": "Search the internet for information and collect them to response.",
        "tools": [internet_search],
    }
]

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    system_prompt="You are a helpful assistant. Use subagents for specialized tasks.",
    tools=file_tools,
    # backend=FilesystemBackend(root_dir=str(os.getcwd() + "/file_system/user_id_1"), virtual_mode=False)
    subagents=subagents
)
