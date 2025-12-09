import pytest
from langsmith import Client
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from langchain_core.messages import HumanMessage, AIMessage

from lc_agent import agent, Context

@pytest.mark.langsmith
def test_trajectory_quality():
    evaluator = create_trajectory_llm_as_judge(
        model="claude-sonnet-4-5-20250929",
        prompt=TRAJECTORY_ACCURACY_PROMPT,
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="What's the weather in Seattle?")]},
        context=Context(user_name="John Smith")
    )

    evaluation = evaluator(outputs=result["messages"])
    # print(f"evaluation: {evaluation}")
    assert evaluation["score"] is True

def test_evaluate():
    client = Client()

    dataset = client.create_dataset("my_dataset4")
    client.create_example(
        inputs={"messages": [HumanMessage(content="What's the weather in Java?")]},
        outputs={"messages": [AIMessage("Java is programming language.")]},
        dataset_id=dataset.id
    )

    trajectory_evaluator = create_trajectory_llm_as_judge(
        model="claude-sonnet-4-5-20250929",
        prompt=TRAJECTORY_ACCURACY_PROMPT,
    )

    def run_agent(inputs):
        """Your agent function that returns trajectory messages."""
        return agent.invoke(inputs, context=Context(user_name="Majo"))["messages"]

    experiment_results = client.evaluate(
        run_agent,
        data="my_dataset4",
        evaluators=[trajectory_evaluator]
    )

    print(f"experiment_results: {experiment_results}")