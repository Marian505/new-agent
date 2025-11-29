import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import testing as t

from lg_agent import graph


@pytest.mark.langsmith(test_suite_name="test_lg_agent")
def test_lg_agent():
    checkpointer = InMemorySaver()
    agent = graph.compile(checkpointer=checkpointer)
    config = {
        "configurable": {
            "thread_id": "test_workflow"
        }
    }

    agent_input = {"messages": [HumanMessage(content="What is java?")]}
    result = agent.invoke(agent_input, config=config)

    t.log_inputs(agent_input)
    t.log_outputs(result)
    assert result is not None

@pytest.mark.langsmith(test_suite_name="test_lg_agent_web_search")
def test_lg_agent_web_search():
    checkpointer = InMemorySaver()
    agent = graph.compile(checkpointer=checkpointer)
    config = {
        "configurable": {
            "thread_id": "test_workflow"
        }
    }

    agent_input = {"messages": [HumanMessage(content="Search on what is java?")]}
    result = agent.invoke(agent_input, config=config)

    t.log_inputs(agent_input)
    t.log_outputs(result)
    assert result is not None
    assert "tavily_search" in result['messages'][1].tool_calls[0]['name']

