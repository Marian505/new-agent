from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain_tavily import TavilySearch
from langchain_anthropic import ChatAnthropic

load_dotenv()

fast_model=ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0)

search = TavilySearch(max_results=3)

system_prompt = "You are a helpful assistant with web search tool."

class CustomState(AgentState):
    pass

agent = create_agent(
    model=fast_model,
    tools=[search],
    system_prompt=system_prompt,
    state_schema=CustomState
)
