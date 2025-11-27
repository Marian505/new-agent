from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()

agent = create_agent(
    model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.0)
)
