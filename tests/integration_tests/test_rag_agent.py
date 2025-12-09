from langchain_core.messages import HumanMessage

from rag_agent import load_web, agent


def test_rag_agent_call_tool():
    web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
    count_stored_docs = load_web(web_paths)

    print(f"loaded docs: {count_stored_docs}")
    assert count_stored_docs > 0

    result = agent.invoke({"messages": [HumanMessage(content="What is Self-Reflection?")]})

    tool_massage = next(message for message in result["messages"] if message.__class__.__name__ == "ToolMessage")
    assert tool_massage is not None
    assert tool_massage.name == "retrieve_context"
