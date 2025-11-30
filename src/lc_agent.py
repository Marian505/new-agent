from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.runtime import Runtime

load_dotenv()

search = TavilySearch(max_results=3)

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict | None:
    # print(f"\nlog_before_model Sate:\n{state['messages'][0].content[1]['type']}")
    # print(f"\nlog_before_model mime_type:\n{state['messages'][0].content[1]['file']}")
    # print(f"\nCompleted request for user:\n{runtime}")
    return None

@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict | None:
    # print(f"\nlog_after_model Sate:\n{state}")
    # print(f"Completed request for user: {runtime}")
    # print(f"Completed request for user: {runtime.context.user_name}")
    return None


agent = create_agent(
    model=init_chat_model("claude-sonnet-4-5-20250929", temperature=0.0),
    tools=[search],
    middleware=[log_before_model, log_after_model]
)
