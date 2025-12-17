import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import testing as t
import pytest_asyncio
from rich.pretty import pprint # noqa: F401

from lg_agent import graph

@pytest_asyncio.fixture
async def agent():
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)

@pytest.mark.asyncio
@pytest.mark.langsmith(test_suite_name="test_lg_agent")
async def test_lg_agent(agent):
    config = {"configurable": {"thread_id": "test_workflow"}}
    inputs = {"messages": [HumanMessage(content="What is java?")]}
    result = await agent.ainvoke(inputs, config=config)

    t.log_inputs(inputs)
    t.log_outputs(result)
    t.log_reference_outputs({"messages": "here should be reference_trajectory"})

    assert result is not None
    tool_massage = [message for message in result["messages"] if message.__class__.__name__ == "ToolMessage"]
    assert len(tool_massage) == 0

@pytest.mark.asyncio
@pytest.mark.langsmith(test_suite_name="test_lg_agent_web_search")
async def test_lg_agent_web_search(agent):
    config = {"configurable": {"thread_id": "test_workflow"}}
    inputs = {"messages": [HumanMessage(content="Search on what is java?")]}
    result = await agent.ainvoke(inputs, config=config)

    t.log_inputs(inputs)
    t.log_outputs(result)
    t.log_reference_outputs({"messages": "here should be reference_trajectory"})
    
    assert result is not None
    tool_massage = next(message for message in result["messages"] if message.__class__.__name__ == "ToolMessage")
    assert tool_massage.name == "tavily_search"

