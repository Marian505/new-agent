import os
import logging

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model, before_agent, \
    SummarizationMiddleware, HumanInTheLoopMiddleware, PIIMiddleware
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_tavily import TavilySearch
from langgraph.runtime import Runtime
# from xhtml2pdf import pisa
from langchain.tools import tool
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

search = TavilySearch(max_results=3)
file_tools = FileManagementToolkit(root_dir=str(os.getcwd() + "/files")).get_tools()
model=init_chat_model("claude-haiku-4-5-20251001", temperature=0.0)
smart_model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.0)

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict | None:
    # print(f"\nlog_before_model Sate:\n{state}")
    print(f"\nCompleted request for user:\n{runtime}")
    return None

@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict | None:
    # print(f"\nlog_after_model Sate:\n{state}")
    print(f"\nCompleted request for user:\n{runtime}")
    return None

class EnhancedPropmpt(BaseModel):
    original_prompt: str
    enhanced_prompt: str

structured_model = model.with_structured_output(EnhancedPropmpt)

@before_agent 
def prompt_enhance(state: AgentState, runtime: Runtime):
    prompt = state["messages"][-1].content
    prompt_len = len(prompt)
    model_instructions = f"""
        Enhance added prompt, fix typos, add capital letters, and special chars like '.,?!', correct gramar.
        Minimal enhanced prompt length is: {prompt_len*2}
        Maximal enhanced prompt length is: {prompt_len*3}
        Prompt: {prompt}
    """
    result = structured_model.invoke(model_instructions)

    state["messages"][-1].content = result.enhanced_prompt
    return None

# @tool
# def simple_pdf_from_html(html_content: str):
#     """Tool for create PDF from HTML. And save the PDF to the folder.
#     Function parameter has to be just valid HTML in string.
#     """
#     # TODO: works, but replace some web service pdf generator, pyppeteer is too havy run in langsmith deployment
#     with open(os.getcwd() + "/files/resume.pdf", "w+b") as pdf_file:
#         pisa.CreatePDF(html_content, dest=pdf_file)

#     return "PDF generated successfully."


system_prompt="""
    You CV assistant. 
    You will read and analyze PDF on user demand. 
    Generate valid HTML And generate PDF on user demand.
    For PDF generation use tool generate_pdf_from_html on user demand.
"""
agent = create_agent(
    model=model,
    tools=file_tools,
    middleware=[
        prompt_enhance
        # log_before_model,
        # PIIMiddleware("ip", strategy="mask", apply_to_input=True),
        # PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]", strategy="redact", apply_to_input=True),
        # SummarizationMiddleware(
        #     model="claude-haiku-4-5-20251001",
        #     trigger=("tokens", 10000),
        #     keep=("messages", 3),
        # ),
        # HumanInTheLoopMiddleware(
        #     interrupt_on={
        #         "simple_pdf_from_html": True,
        #     }
        # ),
    ],
    system_prompt=system_prompt
)
