import pytest

from langsmith import testing as t
from langchain_core.messages import HumanMessage

from lc_agent import agent, Context

@pytest.mark.langsmith
def test_lc_agent():
    result = agent.invoke(
        {
            "messages": [HumanMessage("What is java?")],
            "user_preferences": {"style": "technical", "verbosity": "detailed"},
         },
        context=Context(user_name="John Smith")
    )

    print(result["structured_response"])
    t.log_inputs({})
    t.log_outputs({"messages": result["messages"]})
    t.log_reference_outputs({"messages": "here should be reference_trajectory"})

    assert True
