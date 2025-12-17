import pytest

from langsmith import testing as t
from langchain_core.messages import HumanMessage

from lc_agent import agent

@pytest.mark.asyncio
@pytest.mark.langsmith(test_suite_name="test_lc_agent")
async def test_lc_agent():
    inputs = {"messages": [HumanMessage(content="What is java?")]}
    result = await agent.ainvoke(inputs)

    # print(result["structured_response"])
    t.log_inputs(inputs)
    t.log_outputs(result)
    t.log_reference_outputs({"messages": "here should be reference_trajectory"})

    assert True
