from basic_agent import agent
from langchain.messages import HumanMessage
from rich.pretty import pprint
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from langgraph.checkpoint.memory import InMemorySaver


def test_basic_agent():
    result = agent.invoke({"messages": [HumanMessage("What is Java?")]})

    assert result["messages"] is not None
    tool_massages = [message for message in result["messages"] if message.__class__.__name__ == "ToolMessage"]
    assert len(tool_massages) == 0

def test_basic_agent_web_search():
    result = agent.invoke({"messages": [HumanMessage("What is black hole? Search on web.")]})
    
    assert result["messages"] is not None
    tool_massage = next(message for message in result["messages"] if message.__class__.__name__ == "ToolMessage")
    assert tool_massage is not None
    assert tool_massage.name == "tavily_search"

def test_trajectory_quality():
    evaluator = create_trajectory_llm_as_judge(  
        model="claude-haiku-4-5-20251001",
        prompt=TRAJECTORY_ACCURACY_PROMPT,  
    )  
    
    result = agent.invoke({"messages": [HumanMessage("What is black hole? Search on web.")]})

    evaluation = evaluator(outputs=result["messages"])
    assert evaluation["score"] is True

def test_basic_agent_no_mem(): 
    result = agent.invoke({"messages": [HumanMessage("My name is Majo?")]})
    result2 = agent.invoke({"messages": [HumanMessage("What is my name?")]})

    last_msg = result2["messages"][-1]
    assert "majo" not in last_msg.content.lower(), "Agent should not remember name."

def test_basic_agent_mem():
    config = {"configurable": {"thread_id": "thread_id_1"}}
    agent.checkpointer = InMemorySaver()
    result = agent.invoke({"messages": [HumanMessage("My name is Majo?")]}, config=config)
    result2 = agent.invoke({"messages": [HumanMessage("What is my name?")]}, config=config)

    last_msg = result2["messages"][-1]
    assert "majo" in last_msg.content.lower(), "Agent should remember name."
