import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.agents import AgentState
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_tavily import TavilySearch
from langchain_anthropic import ChatAnthropic

load_dotenv()

fast_model=ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0)
file_tools = FileManagementToolkit(root_dir=str(os.getcwd() + "/files")).get_tools()
search = TavilySearch(max_results=3)

subagents = [
    {
        "name": "web-search-agent",
        "description": "Agent with access to websearch tool",
        "system_prompt": "Search the internet for information and collect them to response.",
        "tools": [search],
    }
]


system_prompt="You are a helpful assistant. Use subagents for specialized tasks."

agent = create_deep_agent(
    model=fast_model,
    system_prompt=system_prompt,
    tools=file_tools,
    subagents=subagents
)
