import asyncio
import pytest


from langchain_core.messages import HumanMessage
from langsmith import AsyncClient, Client
from openevals import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from langsmith.evaluation import aevaluate

from rag_agent import load_web, agent

async def _data_init(project_name: str, dataset_name: str):
    aclient = AsyncClient()
    client= Client()
    inputs = [
        "What is an LLM-powered autonomous agent?",
        "What are the three main functional components of an LLM-powered agent?",
        "How does short-term memory differ from long-term memory in these agents?",
        "Why do LLM agents need external tools?",
        "What role does planning play in an LLM-powered agent?",
    ]
    outputs = [
        "An LLM-powered autonomous agent is a system that uses a large language model as its central controller to perceive tasks, decide on actions, and interact with tools and memory in order to achieve user-defined goals.",
        "The three main functional components are planning, memory, and tool use, working together around the LLM which acts as the agents brain.",
        "Short-term memory corresponds to in-context learning within the models context window, while long-term memory is implemented as an external store (such as a vector database) that can retain and retrieve information over extended periods.",
        "LLM agents need external tools to access up-to-date information, run code, and interface with external systems, overcoming the limitations of static model weights.",
        "Planning lets the agent decompose complex tasks into smaller subgoals, reflect on previous steps, and iteratively refine its actions to improve final results.",
    ]

    project_names = [p.name for p in client.list_projects()]
    if project_name not in project_names:
        client.create_project(project_name)

    dataset_names = [d.name for d in await aclient.list_datasets()]
    if dataset_name not in dataset_names:
        dataset = client.create_dataset(
            dataset_name=dataset_name, description="QA dataset for RAG"
        )
    else:
        dataset = next(d for d in await aclient.list_datasets() if d.name == dataset_name)

    tasks=[
        aclient.create_example(inputs=inp, outputs=outp,dataset_id=dataset.id)
        for inp, outp in zip([{"question": q} for q in inputs], [{"answer": a} for a in outputs])
    ]
    await asyncio.gather(*tasks)

@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_rag_agent_call_tool():
    web_paths = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    count_stored_docs = await load_web(web_paths)
    print(f"loaded docs: {count_stored_docs}")
    assert count_stored_docs > 0

    result = await agent.ainvoke({"messages": [HumanMessage(content="What is Self-Reflection?")]})

    tool_massage = next(message for message in result["messages"] if message.__class__.__name__ == "ToolMessage")
    assert tool_massage is not None
    assert tool_massage.name == "retrieve_context"


async def call_agent(agent, inputs: dict):
    messages = [{"role": "user", "content": inputs["question"]}]
    result = await agent.ainvoke({"messages": messages})

    tool_massages = [
        message
        for message in result["messages"]
        if message.__class__.__name__ == "ToolMessage"
    ]
    if tool_massages:
        context = tool_massages[0].content
        return {"answer": result["messages"][-1].content, "context": context}
    else:
        return {"answer": result["messages"][-1].content}

@pytest.mark.asyncio
async def test_rag_correctness():
    # load data to rag
    web_paths = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    count_stored_docs = await load_web(web_paths)
    print(f"loaded docs: {count_stored_docs}")
    assert count_stored_docs > 0

    # set dataset if not exist
    dataset_name = "rag_agent_dataset"
    project_name = "rag_agent_project"
    _data_init(project_name, dataset_name)

    # set evaluator
    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="claude-haiku-4-5-20251001",
        feedback_key="correctness"
    )

    async def target(inputs):
        return await call_agent(agent, inputs)

    # evaluate
    await aevaluate(
        target,
        data=dataset_name,
        evaluators=[correctness_evaluator],
        experiment_prefix="test-rag-qa-oai",
        metadata={"variant": "stuff website context into gpt-3.5-turbo"},
    )

