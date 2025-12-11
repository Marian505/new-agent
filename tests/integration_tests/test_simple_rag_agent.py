from typing import Annotated, TypedDict
from langsmith import Client

import pytest
from langchain.chat_models import init_chat_model
from rich.pretty import pprint
from simple_rag_agent import SimpleRagAgent

fast_model = None

@pytest.fixture
def client() -> Client:
    return Client()

@pytest.fixture(scope="session", autouse=True)
def setup_fast_model():
    global fast_model
    fast_model = init_chat_model("claude-haiku-4-5-20251001", temperature=0.0)

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
        relevant: Annotated[int, ..., "10 if the retrieved documents are relevant to the question, 0 otherwise"]

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


def call_agent(agent, inputs: dict):
    messages = [{"role": "user", "content": inputs["question"]}]
    result = agent.invoke({"messages": messages})

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


def test_rag_model_1(client):
    simpleRagAgent = SimpleRagAgent()
    agent = simpleRagAgent.get_agent()

    simpleRagAgent.load_data(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent",)
    )

    client.evaluate(
        lambda inputs: call_agent(agent, inputs),
        data="rag_test",
        evaluators=[groundedness, correctness, relevance, retrieval_relevance],
        experiment_prefix="test-simple-rag-model-1",
        metadata={
            "variant": "stuff website context into LLM Powered Autonomous Agents"
        },
    )


def test_rag_model_2(client):
    simpleRagAgent = SimpleRagAgent(model="claude-haiku-4-5-20251001")
    agent = simpleRagAgent.get_agent()

    simpleRagAgent.load_data(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent",)
    )

    client.evaluate(
        lambda inputs: call_agent(agent, inputs),
        data="rag_test",
        evaluators=[groundedness, correctness, relevance, retrieval_relevance],
        experiment_prefix="test-simple-rag-model-2",
        metadata={
            "variant": "stuff website context into LLM Powered Autonomous Agents"
        },
    )