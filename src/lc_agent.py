from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model, dynamic_prompt, ModelRequest, wrap_model_call, \
    ModelResponse
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.runtime import Runtime
from langchain.tools import tool, ToolRuntime
from pydantic import BaseModel, Field

load_dotenv()

class CustomState(AgentState):
    user_preferences: dict

@dataclass
class Context:
    user_name: str

search = TavilySearch(max_results=3)

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name
    system_prompt = f"You are a helpful assistant with web search tool. Address the user as {user_name}."
    return system_prompt

@before_model
def log_before_model(state: CustomState, runtime: Runtime[Context]) -> dict | None:
    # print(f"Sate: {state}")
    print(f"Processing request for user: {runtime.context.user_name}")
    return None

@after_model
def log_after_model(state: CustomState, runtime: Runtime) -> dict | None:
    # print(f"Sate: {state}")
    # print(f"Completed request for user: {runtime}")
    print(f"Completed request for user: {runtime.context.user_name}")
    return None

class SimpleResponse(BaseModel):
    """Simple response for early conversation."""
    answer: str = Field(description="A brief answer")

class DetailedResponse(BaseModel):
    """Detailed response for established conversation."""
    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of reasoning")
    confidence: float = Field(description="Confidence score 0-1")

# TODO: does not work yet
@wrap_model_call
def state_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select output format based on State."""
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)

    # if message_count < 3:
    #     return handler(request.override(response_format=SimpleResponse))
    # else:
    return handler(request.override(response_format=DetailedResponse))

@tool
def check_authentication(
    runtime: ToolRuntime
) -> str:
    """Check if user is authenticated."""
    # Read from State: check current auth status
    current_state = runtime.state
    is_authenticated = current_state.get("authenticated", False)

    if is_authenticated:
        return "User is authenticated"
    else:
        return "User is not authenticated"

agent = create_agent(
    model=init_chat_model("claude-sonnet-4-5-20250929", temperature=0.0),
    tools=[search, check_authentication],
    middleware=[log_before_model, log_after_model, dynamic_system_prompt],
    context_schema=Context,
    response_format=DetailedResponse,
    state_schema=CustomState
)
