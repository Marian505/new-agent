import asyncio
import os
import logging

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.runtime import Runtime
from pyppeteer import launch

load_dotenv()

logger = logging.getLogger(__name__)

search = TavilySearch(max_results=3)
file_tools = FileManagementToolkit(root_dir=str(os.getcwd() + "/files")).get_tools()

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict | None:
    # print(f"\nlog_before_model Sate:\n{state['messages'][0].content[1]['type']}")
    # print(f"\nlog_before_model mime_type:\n{state['messages'][0].content[1]['file']}")
    print(f"\nCompleted request for user:\n{runtime}")
    return None

@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict | None:
    # print(f"\nlog_after_model Sate:\n{state}")
    print(f"\nCompleted request for user:\n{runtime}")
    # print(f"Completed request for user: {runtime.context.user_name}")
    return None


IS_LOCAL = os.getenv("ENVIRONMENT", "local").lower() == "local"
BROWSER_PATH = os.getenv("BROWSER_EXECUTABLE_PATH")

_browser = None
_browser_lock = asyncio.Lock()

async def get_browser():
    """Get or create browser instance."""
    global _browser
    
    async with _browser_lock:
        if _browser is not None:
            return _browser
        
        try:
            if IS_LOCAL and BROWSER_PATH:
                logger.info(f"Local mode: Launching browser from {BROWSER_PATH}")
                _browser = await launch(
                    executablePath=BROWSER_PATH,
                    headless=True,
                    args=["--no-sandbox"]
                )
            else:
                logger.info("Production mode: Auto-detecting browser")
                _browser = await launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-gpu"]
                )
            
            logger.info("Browser launched successfully")
            return _browser
            
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            raise

def _prepare_pdf_directory():
    """Synchronous helper for file operations."""
    current_dir = os.getcwd()
    pdf_dir = os.path.join(current_dir, "files")
    os.makedirs(pdf_dir, exist_ok=True)
    return os.path.join(pdf_dir, "output.pdf")

@tool
async def generate_pdf_from_html(html_content: str) -> str:
    """Tool for create PDF from HTML. And save the PDF to the folder.
    Function parameter has to be just valid HTML in string.
    """
    try:
        pdf_path = await asyncio.to_thread(_prepare_pdf_directory)

        browser = await get_browser()
        page = await browser.newPage()
        await page.setContent(html_content)
        await page.pdf({'path': pdf_path, 'format': 'A4', 'margin': {'top': '10mm', 'bottom': '10mm'}})
        await browser.close()

        return "PDF generated successfully."
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return f"Error: {str(e)}"


print(f"lc pid: {os.getpid()}")
system_prompt="""
    You CV assistant. 
    You will read and analyze PDF on user demand. 
    Generate valid HTML And generate PDF on user demand.
    For PDF generation use tool generate_pdf_from_html on user demand.
"""
agent = create_agent(
    model=init_chat_model("claude-haiku-4-5-20251001", temperature=0.0),
    tools=[generate_pdf_from_html] + file_tools,
    middleware=[log_before_model, log_after_model],
    system_prompt=system_prompt
)
