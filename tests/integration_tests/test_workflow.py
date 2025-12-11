import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import testing as t
from lg_workflow import State, workflow
from langgraph.types import Command


@pytest.fixture
def graph():
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


@pytest.mark.langsmith(test_suite_name="test_fast_workflow")
def test_fast_workflow(graph):
    config = {"configurable": {"thread_id": "test_workflow"}}
    state = State(user_prompt="What is java?", enhanced_prompt={}, model_type="fast")
    result = graph.invoke(state, config=config)

    assert "__interrupt__" in result

    result  = graph.invoke(Command(resume="fast"), config=config)
    t.log_inputs(state)
    t.log_outputs(result)

    assert "user_prompt" in result
    assert "approved_prompt" in result
    assert "enhanced_prompt" in result
    assert "response" in result
    assert "model_type" in result
    assert "fast" in result['model_type']

@pytest.mark.langsmith(test_suite_name="test_smart_workflow")
def test_smart_workflow(graph):
    config = {"configurable": {"thread_id": "test_workflow"}}
    state = State(user_prompt="What is java?", enhanced_prompt={}, model_type="smart")
    result = graph.invoke(state, config=config)

    assert "__interrupt__" in result

    result  = graph.invoke(Command(resume="smart"), config=config)
    t.log_inputs(state)
    t.log_outputs(result)

    assert "user_prompt" in result
    assert "approved_prompt" in result
    assert "enhanced_prompt" in result
    assert "response" in result
    assert "model_type" in result
    assert "smart" in result['model_type']

# this test is too long and expensive
# @pytest.mark.langsmith(test_suite_name="test_premium_workflow")
# def test_premium_workflow(graph):
#     config = {"configurable": {"thread_id": "test_workflow"}}
#     state = State(user_prompt="What is java?", enhanced_prompt={}, model_type="premium")
#     result = graph.invoke(state, config=config)

#     assert "__interrupt__" in result

#     result  = graph.invoke(Command(resume="premium"), config=config)
#     t.log_inputs(state)
#     t.log_outputs(result)

#     assert "user_prompt" in result
#     assert "approved_prompt" in result
#     assert "enhanced_prompt" in result
#     assert "response" in result
#     assert "model_type" in result
#     assert "premium" in result['model_type']

