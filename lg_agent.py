import uuid
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore


load_dotenv()

class State(MessagesState):
    messages: Annotated[list, add_messages]


search = TavilySearch(max_results=3)
model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.0)
tools = [search]
model_with_tools = model.bind_tools(tools)
tool_node = ToolNode(tools)


def call_model(
        state: MessagesState,
        config: RunnableConfig,
        *,
        store: BaseStore,
):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = last_message.content.lower()
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}


def should_continue(state: State):
    """Determines the next step based on the model's last message."""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")
agent = workflow.compile()

