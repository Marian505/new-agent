from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_tavily import TavilySearch
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()

search = TavilySearch(max_results=3)
model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.0)
tools = [search]
model_with_tools = model.bind_tools(tools)


def call_model(state: MessagesState):
    system_msg = f"You are a helpful assistant talking to the user."
    response = model_with_tools.invoke([SystemMessage(content=system_msg)] + state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState):
    """Determines the next step based on the model's last message."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

tool_node = ToolNode(tools)

graph = StateGraph(MessagesState)

graph.add_node("call_model", call_model)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "call_model")
graph.add_conditional_edges(
    "call_model",
    should_continue,
    ["tool_node", END]
)
graph.add_edge("tool_node", "call_model")

agent = graph.compile()

