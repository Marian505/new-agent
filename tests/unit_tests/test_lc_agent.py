from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolCall, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv("./.env")

def test_invoke():
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[ToolCall(name="foo", args={"bar": "baz"}, id="call_1")]),
        "bar"
    ]))

    result = model.invoke("hello")
    print(f"result: {result}")

    assert result is not None

def test_mem():
    agent = create_agent(
        model=init_chat_model("claude-sonnet-4-5-20250929"),
        checkpointer=InMemorySaver()
    )

    config = {
        "configurable": {
            "thread_id": "thread_id_1",
        }
    }

    agent.invoke({"messages": [HumanMessage(content="I live in Sydney, Australia.")]}, config=config)
    result = agent.invoke({"messages": [HumanMessage(content="What's my local time?")]}, config=config)
    print(f"result: {result['messages'][-1].content}")
    assert True