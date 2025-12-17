import os
import logging
from typing import Any
from dotenv import load_dotenv
import httpx 

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import FileManagementToolkit
from langgraph.runtime import Runtime
from langchain.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

fast_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0)
smart_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.0)
file_tools = FileManagementToolkit(root_dir=str(os.getcwd() + "/files")).get_tools()

@tool
async def html_to_pdf(html: str) -> str:
    """Tool for create PDF from HTML.
    Function parameter has to be just valid HTML in string.
    """
    try:
        async with httpx.AsyncClient() as client:
            _resposne = await client.post(
                "https://pdf.weakpass.org/api/html-to-pdf",
                content=html,
                headers={"Content-Type": "text/html"}
            ) 
        return "PDF generated successfully"
    except Exception as e:
        return f"PDF generation failed: {str(e)}"

@before_agent
async def prompt_enhance(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Enhance user prompt before agent runs."""
    if not state.get("messages"):
        return None
    
    prompt = state["messages"][-1].content
    prompt_len = len(prompt)
    
    instructions = f"""
    Enhance this user prompt:
    - Fix typos and grammar
    - Add proper capitalization  
    - Include punctuation (.,?!)
    - Min length: {prompt_len*2}, Max: {prompt_len*3}
    
    Original: {prompt}
    """
    
    result = await fast_model.ainvoke(instructions)
    enhanced_prompt = result.content
    
    return {"messages": state["messages"][:-1] + [
        type(state["messages"][-1])(content=enhanced_prompt)
    ]}

system_prompt = """
You CV assistant. 
You will read and analyze PDF on user demand. 
Generate valid HTML And generate PDF on user demand.
For PDF generation use tool generate_pdf_from_html on user demand.
"""
agent = create_agent(
    model=fast_model,
    tools=file_tools + [html_to_pdf],
    middleware=[prompt_enhance],
    system_prompt=system_prompt
)
