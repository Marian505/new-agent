import pytest

from langsmith import testing as t
from langchain_core.messages import HumanMessage

from lc_agent import agent

@pytest.mark.langsmith(test_suite_name="test_lc_agent")
def test_lc_agent():
    inputs = {"messages": [HumanMessage("What is java?")]}
    result = agent.invoke(inputs)

    # print(result["structured_response"])
    t.log_inputs(inputs["messages"])
    t.log_outputs(result["messages"])
    t.log_reference_outputs({"messages": "here should be reference_trajectory"})

    assert True
