from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
import httpx 
import base64
from rich.pretty import pprint
import aiofiles
from dataclasses import dataclass


load_dotenv()

fast_model=init_chat_model("claude-haiku-4-5-20251001", temperature=0.0)
smart_model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.0)

@dataclass
class ContextSchema:
    pdf_path: str | None

@tool
async def html_to_pdf(html: str, runtime: ToolRuntime[ContextSchema]) -> str:
    """Tool for create PDF from HTML.
    Function parameter has to be just valid HTML in string.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://pdf.weakpass.org/api/html-to-pdf",
            content=html,
            headers={"Content-Type": "text/html"}
        ) 
        if response.status_code != 200:
            raise ValueError(f"PDF API failed: {response.status_code}")

    # TODO: no prod version, for ptod store in S3
    if runtime.context.pdf_path is not None:
        pdf_path = runtime.context.pdf_path
        pdf_bytes = base64.b64decode(response.json()['pdf_base64'])
        async with aiofiles.open(pdf_path, 'wb') as f:
            await f.write(pdf_bytes)
    
    return f"PDF generated, response code: {response.status_code}"

system_prompt="""
    You are CV assistant. 
    You will read and analyze user CV in PDF. 
    Generate valid HTML and generate PDF on user demand.
    For PDF generation use tool html_to_pdf on user demand.
"""
agent = create_agent(
    model=fast_model,
    tools=[html_to_pdf],
    # middleware=[
    #     SummarizationMiddleware(
    #         model=smart_model,
    #         trigger=("tokens", 10000),
    #         keep=("messages", 3),
    #     )
    # ],
    system_prompt=system_prompt,
    context_schema=ContextSchema
)
