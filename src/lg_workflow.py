from operator import or_
from typing import NotRequired

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict, Literal, Annotated

load_dotenv()

fast_model = init_chat_model("claude-haiku-4-5-20251001")
smart_model = init_chat_model("claude-sonnet-4-5-20250929")
premium_model = init_chat_model("claude-opus-4-5-20251101")

class State(TypedDict):
    user_prompt: str | None
    enhanced_prompt: Annotated[dict | None, or_]
    approved_prompt: NotRequired[str | None]
    response: NotRequired[str | None]
    model_type: Literal["fast", "smart", "premium"]

def fast_enhance_prompt(state: State):
    node_prompt="""
        Analyze user prompt ant proposes an enhanced version. 
        Maximal length of the enhanced prompt is 30 words. 
        Return just the enhanced string prompt. No new lines
    """
    enhanced_prompt = fast_model.invoke(
        [
            HumanMessage(content=state['user_prompt']),
            SystemMessage(content=node_prompt)
        ]
    )

    return {"enhanced_prompt": {"fast": enhanced_prompt.content}}

def smart_enhance_prompt(state: State):
    node_prompt="""
        Analyze user prompt ant proposes an enhanced version. 
        Maximal length of the enhanced prompt is 30 words. 
        Return just the enhanced string prompt. No new lines
    """
    enhanced_prompt = smart_model.invoke(
        [
            HumanMessage(content=state['user_prompt']),
            SystemMessage(content=node_prompt)
        ]
    )

    return {"enhanced_prompt": {"smart": enhanced_prompt.content}}

def premium_enhance_prompt(state: State):
    node_prompt = """
        Analyze user prompt ant proposes an enhanced version. 
        Maximal length of the enhanced prompt is 30 words. 
        Return just the enhanced string prompt. No new lines
    """
    enhanced_prompt = premium_model.invoke(
        [
            HumanMessage(content=state['user_prompt']),
            SystemMessage(content=node_prompt)
        ]
    )

    return {"enhanced_prompt": {"premium": enhanced_prompt.content}}

def choose_prompt(state: State):
    edited_prompt = interrupt({
        "question": "Choose enhanced prompt.",
        "details": state["enhanced_prompt"]
    })

    if edited_prompt == "smart":
        return {"approved_prompt": state["enhanced_prompt"]["smart"]}
    elif edited_prompt == "premium":
        return {"approved_prompt": state["enhanced_prompt"]["premium"]}
    else:
        return {"approved_prompt": state["enhanced_prompt"]["fast"]}

def condition(state: State):
    if state["model_type"] == "fast":
        return "fast_call_llm"
    elif state["model_type"] == "smart":
        return "smart_call_llm"
    elif state["model_type"] == "premium":
        return "premium_call_llm"

def fast_call_llm(state: State):
    llm_prompt = "You are a helpful assistant."
    response = fast_model.invoke(f"{llm_prompt} user prompt: {state['approved_prompt']}")
    return {"response": response.content}

def smart_call_llm(state: State):
    llm_prompt = "You are a helpful assistant."
    response = smart_model.invoke(f"{llm_prompt} user prompt: {state['approved_prompt']}")
    return {"response": response.content}

def premium_call_llm(state: State):
    llm_prompt = "You are a helpful assistant."
    response = premium_model.invoke(f"{llm_prompt} user prompt: {state['approved_prompt']}")
    return {"response": response.content}


workflow = StateGraph(State)

workflow.add_node("fast_enhance_prompt", fast_enhance_prompt)
workflow.add_node("smart_enhance_prompt", smart_enhance_prompt)
workflow.add_node("premium_enhance_prompt", premium_enhance_prompt)

workflow.add_node("choose_prompt", choose_prompt)

workflow.add_node("fast_call_llm", fast_call_llm)
workflow.add_node("smart_call_llm", smart_call_llm)
workflow.add_node("premium_call_llm", premium_call_llm)


workflow.add_edge(START, "fast_enhance_prompt")
workflow.add_edge(START, "smart_enhance_prompt")
workflow.add_edge(START, "premium_enhance_prompt")

workflow.add_edge("fast_enhance_prompt", "choose_prompt")
workflow.add_edge("smart_enhance_prompt", "choose_prompt")
workflow.add_edge("premium_enhance_prompt", "choose_prompt")

workflow.add_conditional_edges(
    "choose_prompt",
    condition,
    {
        "fast_call_llm": "fast_call_llm",
        "smart_call_llm": "smart_call_llm",
        "premium_call_llm": "premium_call_llm"
    }
)

workflow.add_edge("fast_call_llm", END)
workflow.add_edge("smart_call_llm", END)
workflow.add_edge("premium_call_llm", END)

graph = workflow.compile()

