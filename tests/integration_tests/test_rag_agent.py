from langchain_core.messages import HumanMessage

from rag_agent import load_web, agent


def test_rag_agent_call_tool():
    web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
    count_stored_docs = load_web(web_paths)

    print(f"loaded docs: {count_stored_docs}")
    assert count_stored_docs > 0

    result = agent.invoke({"messages": [HumanMessage(content="What is Self-Reflection?")]})

    for message in result['messages']:
        print(f"\n{type(message)}:\n{message}\n\n")

    print(f"\ntool_calls: {result['messages'][1].tool_calls}")
    assert "retrieve_context" in result['messages'][1].tool_calls[0]['name']
