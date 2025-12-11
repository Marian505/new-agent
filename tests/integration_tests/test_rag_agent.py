import os
import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import Client
from openevals import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
import requests
from bs4 import BeautifulSoup

from rag_agent import load_web, agent

url = "https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
text = [p.text for p in soup.find_all("p")]
full_text = "\n".join(text)

def test_rag_agent_call_tool():
    web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
    count_stored_docs = load_web(web_paths)

    print(f"loaded docs: {count_stored_docs}")
    assert count_stored_docs > 0

    result = agent.invoke({"messages": [HumanMessage(content="What is Self-Reflection?")]})

    tool_massage = next(message for message in result["messages"] if message.__class__.__name__ == "ToolMessage")
    assert tool_massage is not None
    assert tool_massage.name == "retrieve_context"


def call_agent(inputs: dict):

    system_msg = (
        f"Answer user questions in 2-3 sentences about this context:\n\n\{full_text}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": inputs["question"]},
    ]
    result = agent.invoke({"messages": messages})

    return {"answer": result["messages"][-1].content}


def test_rag():

    inputs = [
        "What is an LLM-powered autonomous agent?",
        "What are the three main functional components of an LLM-powered agent?",
        "How does short-term memory differ from long-term memory in these agents?",
        "Why do LLM agents need external tools?",
        "What role does planning play in an LLM-powered agent?"
    ]

    outputs = [
        "An LLM-powered autonomous agent is a system that uses a large language model as its central controller to perceive tasks, decide on actions, and interact with tools and memory in order to achieve user-defined goals.",
        "The three main functional components are planning, memory, and tool use, working together around the LLM which acts as the agents brain.",
        "Short-term memory corresponds to in-context learning within the models context window, while long-term memory is implemented as an external store (such as a vector database) that can retain and retrieve information over extended periods.",
        "LLM agents need external tools to access up-to-date information, run code, and interface with external systems, overcoming the limitations of static model weights.",
        "Planning lets the agent decompose complex tasks into smaller subgoals, reflect on previous steps, and iteratively refine its actions to improve final results."
    ]

    client = Client()
    dataset_name = "rag_test"

    # project = client.create_project("RAGTest")
    # dataset = client.create_dataset(
    #     dataset_name=datases_name,
    #     description="QA dataset for RAG"
    # )
    # client.create_examples(
    #     inputs=[{"question": q} for q in inputs],
    #     outputs=[{"answer": a} for a in outputs],
    #     dataset_id=dataset.id
    # )


    cot_qa_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="claude-haiku-4-5-20251001",
        feedback_key="cot_qa"
    )

    experiment_results = client.evaluate(
        call_agent,
        data=dataset_name,
        evaluators=[cot_qa_evaluator],
        experiment_prefix="test-rag-qa-oai",
        metadata={"variant": "stuff website context into gpt-3.5-turbo"},
    )

