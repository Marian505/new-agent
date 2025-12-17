import aiofiles
from cv_agent import ContextSchema, agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from rich.pretty import pprint # noqa: F401
import pytest
import utils

@pytest.mark.asyncio
async def test_cv_agent():
    # TODO: multimodal propt
    agent.checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "thread_id_1"}}

    path = await utils.get_cwd_async() + "/tests/integration_tests/data/Marian_Cakajda_CV.pdf"
    async with aiofiles.open(path, "rb") as pdf_file:
        pdf_bytes = await pdf_file.read()
    pdf_base64_bytes = await utils.b64encode(pdf_bytes)
    pdf_base64_str = pdf_base64_bytes.decode('ascii')

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze my CV."},
            {"type": "file", "base64": pdf_base64_str, "mime_type": "application/pdf"},
        ]
    }

    _result = await agent.ainvoke(
        {"messages": [message]},        
        config=config
    )
    # pprint(result["messages"][-1].content)

    _result2 = await agent.ainvoke(
        {"messages": [HumanMessage(content="Suggest enhancements to improve my CV.")]}, 
        config=config
    )
    # pprint(result2["messages"][-1])

    pdf_path = await utils.get_cwd_async() + "/tests/integration_tests/data/GenCV.pdf"
    _result3 = await agent.ainvoke(
        {"messages": [HumanMessage(content="Implement enhancents to CV. Put all informations on one page. Generate new CV by tool. pdf_path")]}, 
        config=config,
        context=ContextSchema(pdf_path=pdf_path)
    )
    # pprint(result3["messages"][-1])

    assert await utils.is_path_exists(pdf_path)