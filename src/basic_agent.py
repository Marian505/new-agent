from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

load_dotenv()

search = TavilySearch(max_results=3)

system_prompt = "You are a helpful assistant with web search tool."

agent = create_agent(
    model = "claude-sonnet-4-5-20250929",
    tools=[search],
    system_prompt=system_prompt
)
