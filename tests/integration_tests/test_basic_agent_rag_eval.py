from typing import Annotated, TypedDict
from langsmith import Client

# from langsmith.run_helpers import traceable
from openevals import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
import requests
from bs4 import BeautifulSoup
import pytest
from langchain.chat_models import init_chat_model
from langsmith.schemas import Run, Example

from basic_agent import agent


url = "https://lilianweng.github.io/posts/2023-06-23-agent"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
text = [p.text for p in soup.find_all("p")]
full_text = "\n".join(text)
fast_model = init_chat_model("claude-haiku-4-5-20251001", temperature=0.0)


@pytest.fixture
def client() -> Client:
    return Client()

# TODO: does not work, I do not know why, yet
# @traceable(project_name="rag-curom-eval", run_type="llm")
def call_agent(inputs: dict):
    system_msg = (f"Answer user questions in 2-3 sentences about this context:\n\n{full_text}")
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": inputs["question"]},
    ]
    result = agent.invoke({"messages": messages})
    return {"answer": result["messages"][-1].content, "context": system_msg}


def data_init(client, project_name: str, dataset_name: str):
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

    dataset_names = [d.name for d in client.list_datasets()]
    if dataset_name not in dataset_names:
        dataset = client.create_dataset(
            dataset_name=dataset_name, description="QA dataset for RAG"
        )
    else:
        dataset = next(d for d in client.list_datasets() if d.name == dataset_name)

    client.create_examples(
        inputs=[{"question": q} for q in inputs],
        outputs=[{"answer": a} for a in outputs],
        dataset_id=dataset.id,
    )


def test_rag(client):
    dataset_name = "rag_test"
    cot_qa_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="claude-haiku-4-5-20251001",
        feedback_key="cot_qa",
    )
    client.evaluate(
        call_agent,
        data=dataset_name,
        evaluators=[cot_qa_evaluator],
        experiment_prefix="test-rag-qa-oai",
        metadata={
            "variant": "stuff website context into LLM Powered Autonomous Agents"
        },
    )


def is_answered(run: Run, example: Example):
    student_answer = run.outputs.get("answer")
    if not student_answer:
        return {"key": "is_answered", "score": 0}
    else:
        return {"key": "is_answered", "score": 1}


def test_custom_eval(client):
    dataset_name = "rag_test_eval"
    project_name = "rag-curom-eval"

    data_init(client, project_name, dataset_name)

    client.evaluate(
        call_agent,
        data=dataset_name,
        evaluators=[is_answered],
        experiment_prefix="test-rag-qa-oai-custom-eval",
        metadata={
            "variant": "stuff website context into LLM Powered Autonomous Agents"
        },
    )


#################################################
def groundedness(outputs: dict) -> dict:
    """A simple evaluator for RAG answer groundedness."""

    class GroundedGrade(TypedDict):
        explanation: Annotated[str, ..., "Explain your reasoning for the score"]
        grounded: Annotated[
            int,
            ...,
            "Provide the score on if the answer hallucinates from the documents",
        ]

    grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
    Grounded:
    Grouded value can be 0 - 10.
    A grounded value of 10 means that the student's answer meets all of the criteria.
    A grounded value of 0 means that the student'A answer does not meet all of the criteria.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

    grounded_llm = fast_model.with_structured_output(GroundedGrade)
    answer = f"FACTS: {outputs['context']}\n\nSTUDENT ANSWER: {outputs['answer']}"

    grade = grounded_llm.invoke(
        [
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return {
        "key": "groundedness",
        "score": grade["grounded"],
        "comment": grade["explanation"],
    }


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """An evaluator for RAG answer accuracy"""

    class CorrectnessGrade(TypedDict):
        explanation: Annotated[str, ..., "Explain your reasoning for the score"]
        correct: Annotated[bool, ..., "true if the answer is correct, false otherwise."]

    correctness_instructions = """
    You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
    (1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
    (3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

    Correctness:
    A correctness value of true means that the student's answer meets all of the criteria.
    A correctness value of false means that the student's answer does not meet all of the criteria.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

    grounded_llm = fast_model.with_structured_output(CorrectnessGrade)

    answer = f"""
        QUESTION: {inputs["question"]}\n\n
        GROUND TRUTH ANSWER: {reference_outputs["answer"]}\n\n
        STUDENT ANSWER: {outputs["answer"]}"""

    grade = grounded_llm.invoke(
        [
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answer},
        ]
    )

    return {
        "key": "correctness",
        "score": grade["correct"],
        "comment": grade["explanation"],
    }


def relevance(inputs: dict, outputs: dict) -> dict:
    """A simple evaluator for RAG answer helpfulness."""

    class RelevanceGrade(TypedDict):
        explanation: Annotated[str, ..., "Explain your reasoning for the score"]
        relevant: Annotated[
            int, ..., "Provide the score on whether the answer addresses the question"
        ]

    relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
    (1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
    (2) Ensure the STUDENT ANSWER helps to answer the QUESTION

    Relevance:
    Relevance value can be 0 - 10.
    A relevance value of 10 means that the student's answer meets all of the criteria.
    A relevance value of 0 means that the student's answer does not meet all of the criteria.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

    grounded_llm = fast_model.with_structured_output(RelevanceGrade)

    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"

    grade = grounded_llm.invoke(
        [
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )

    return {
        "key": "relevance",
        "score": grade["relevant"],
        "comment": grade["explanation"],
    }


def retrieval_relevance(inputs: dict, outputs: dict) -> dict:
    """An evaluator for document relevance"""

    class RetrievalRelevanceGrade(TypedDict):
        explanation: Annotated[str, ..., "Explain your reasoning for the score"]
        relevant: Annotated[
            int,
            ...,
            "10 if the retrieved documents are relevant to the question, 0 otherwise",
        ]

    retrieval_relevance_instructions = """
    You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
    (1) You goal is to identify FACTS that are completely unrelated to the QUESTION
    (2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
    (3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

    Relevance:
    Relevance value can be 0 - 10.
    A relevance value of 10 means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
    A relevance value of 0 means that the FACTS are completely unrelated to the QUESTION.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

    grounded_llm = fast_model.with_structured_output(RetrievalRelevanceGrade)

    answer = f"FACTS: {outputs['context']}\nQUESTION: {inputs['question']}"

    grade = grounded_llm.invoke(
        [
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )

    return {
        "key": "retrieval_relevance",
        "score": grade["relevant"],
        "comment": grade["explanation"],
    }

def test_rag_hall(client):
    client.evaluate(
        call_agent,
        data="rag_test",
        evaluators=[groundedness, correctness, relevance, retrieval_relevance],
        experiment_prefix="test-rag-all",
        metadata={
            "variant": "stuff website context into LLM Powered Autonomous Agents"
        },
    )
