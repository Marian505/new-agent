from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch


load_dotenv()

search = TavilySearch(max_results=3)

agent = create_agent(
    model=init_chat_model("claude-sonnet-4-5-20250929", temperature=0.0),
    tools=[search],
)
