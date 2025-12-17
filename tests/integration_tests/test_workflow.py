import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import testing as t
from lg_workflow import State, workflow
from langgraph.types import Command
import pytest_asyncio


@pytest_asyncio.fixture
async def graph():
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)

@pytest.mark.asyncio
@pytest.mark.langsmith(test_suite_name="test_fast_workflow")
async def test_fast_workflow(graph):
    config = {"configurable": {"thread_id": "test_workflow"}}
    state = State(user_prompt="What is java?", enhanced_prompt={}, model_type="fast")
    result = await graph.ainvoke(state, config=config)

    assert "__interrupt__" in result

    result  = await graph.ainvoke(Command(resume="fast"), config=config)
    t.log_inputs(state)
    t.log_outputs(result)

    assert "user_prompt" in result
    assert "approved_prompt" in result
    assert "enhanced_prompt" in result
    assert "response" in result
    assert "model_type" in result
    assert "fast" in result['model_type']

@pytest.mark.asyncio
@pytest.mark.langsmith(test_suite_name="test_smart_workflow")
async def test_smart_workflow(graph):
    config = {"configurable": {"thread_id": "test_workflow"}}
    state = State(user_prompt="What is java?", enhanced_prompt={}, model_type="smart")
    result = await graph.ainvoke(state, config=config)

    assert "__interrupt__" in result

    result  = await graph.ainvoke(Command(resume="smart"), config=config)
    t.log_inputs(state)
    t.log_outputs(result)

    assert "user_prompt" in result
    assert "approved_prompt" in result
    assert "enhanced_prompt" in result
    assert "response" in result
    assert "model_type" in result
    assert "smart" in result['model_type']

# this test is too long and expensive
# @pytest.mark.asyncio
# @pytest.mark.langsmith(test_suite_name="test_premium_workflow")
# async def test_premium_workflow(graph):
#     config = {"configurable": {"thread_id": "test_workflow"}}
#     state = State(user_prompt="What is java?", enhanced_prompt={}, model_type="premium")
#     result = await graph.ainvoke(state, config=config)

#     assert "__interrupt__" in result

#     result  = await graph.ainvoke(Command(resume="premium"), config=config)
#     t.log_inputs(state)
#     t.log_outputs(result)

#     assert "user_prompt" in result
#     assert "approved_prompt" in result
#     assert "enhanced_prompt" in result
#     assert "response" in result
#     assert "model_type" in result
#     assert "premium" in result['model_type']

