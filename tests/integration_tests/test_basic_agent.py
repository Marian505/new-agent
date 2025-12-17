import pytest

from langchain.messages import HumanMessage
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from langgraph.checkpoint.memory import InMemorySaver

from basic_agent import agent

@pytest.mark.asyncio
async def test_basic_agent():
    result = await agent.ainvoke({"messages": [HumanMessage("What is Java? Do not use search tool.")]})

    assert result["messages"] is not None
    tool_massages = [message for message in result["messages"] if message.__class__.__name__ == "ToolMessage"]
    assert len(tool_massages) == 0

@pytest.mark.asyncio
async def test_basic_agent_web_search():
    result = await agent.ainvoke({"messages": [HumanMessage("What is black hole? Search on web.")]})
    
    assert result["messages"] is not None
    tool_massage = next(message for message in result["messages"] if message.__class__.__name__ == "ToolMessage")
    assert tool_massage is not None
    assert tool_massage.name == "tavily_search"

@pytest.mark.asyncio
async def test_trajectory_quality():
    evaluator = create_trajectory_llm_as_judge(  
        model="claude-haiku-4-5-20251001",
        prompt=TRAJECTORY_ACCURACY_PROMPT,  
    )  
    
    result = await agent.ainvoke({"messages": [HumanMessage("What is black hole? Search on web.")]})

    evaluation = evaluator(outputs=result["messages"])
    assert evaluation["score"] is True

@pytest.mark.asyncio
async def test_basic_agent_no_mem(): 
    _result = await agent.ainvoke({"messages": [HumanMessage("My name is Majo?")]})
    result2 = await agent.ainvoke({"messages": [HumanMessage("What is my name?")]})

    last_msg = result2["messages"][-1]
    assert "majo" not in last_msg.content.lower(), "Agent should not remember name."

@pytest.mark.asyncio
async def test_basic_agent_mem():
    config = {"configurable": {"thread_id": "thread_id_1"}}
    agent.checkpointer = InMemorySaver()
    _result = await agent.ainvoke({"messages": [HumanMessage("My name is Majo?")]}, config=config)
    result2 = await agent.ainvoke({"messages": [HumanMessage("What is my name?")]}, config=config)

    last_msg = result2["messages"][-1]
    assert "majo" in last_msg.content.lower(), "Agent should remember name."
