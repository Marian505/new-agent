from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

agent = create_agent(
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
)
