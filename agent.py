import os
import requests

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.tools import tool


# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================

ENV_PATH = r"C:\Users\Manoj\Desktop\LangChain-Agent\research\.env"

load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# =========================
# CHECK KEYS
# =========================

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is missing")

if not WEATHERSTACK_API_KEY:
    raise RuntimeError("WEATHERSTACK_API_KEY is missing")


# =========================
# TAVILY SEARCH TOOL
# =========================

search_tool = TavilySearch(
    max_results=2,
    tavily_api_key=TAVILY_API_KEY
)


# =========================
# WEATHER TOOL
# =========================

@tool
def get_weather_data(city: str) -> str:
    """Get the current weather information for a given city."""

    url = (
        "https://api.weatherstack.com/current"
        f"?access_key={WEATHERSTACK_API_KEY}"
        f"&query={city}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}."

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']} Celsius\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


# =========================
# GEMINI
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)


# =========================
# SYSTEM PROMPT
# =========================

system_prompt = """
You are a helpful research assistant.

Use the search tool when you need current or
factual information.

Use the weather tool when the user asks for
current weather information.

You can use multiple tools when necessary
to complete a multi-step task.

Give a clear and concise final answer.
"""


# =========================
# CREATE AGENT
# =========================

tools = [
    search_tool,
    get_weather_data
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


# =========================
# FUNCTION FOR STREAMLIT
# =========================

def run_agent(user_query: str) -> str:

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_query
            }
        ]
    })

    content = response["messages"][-1].content

    # Gemini may return content as a list of blocks
    if isinstance(content, list):

        text_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        return "\n".join(text_parts)

    return str(content)