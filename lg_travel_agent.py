import asyncio
import os
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables from .env file.
# override=True ensures .env values take precedence over any already-exported shell vars.
load_dotenv(override=True)
OKAHU_API_KEY = os.getenv("OKAHU_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# MONOCLE_EXPORTER controls where traces are sent (e.g. "file", "okahu", or "file,okahu").
MONOCLE_EXPORTER = os.getenv("MONOCLE_EXPORTER")

# Enable Monocle tracing — instruments all agent/tool invocations automatically.
# Every call to the supervisor and sub-agents will emit OpenTelemetry spans,
# visible in the Okahu Cloud UI or a local trace file depending on MONOCLE_EXPORTER.
from monocle_apptrace import setup_monocle_telemetry
#setup_monocle_telemetry(workflow_name = 'test_lg_travel_agent', monocle_exporters_list = 'file,okahu')
setup_monocle_telemetry(workflow_name = 'test_lg_travel_agent', monocle_exporters_list= MONOCLE_EXPORTER)

import logging
logger = logging.getLogger(__name__)

# The weather MCP server runs as a separate process on this port.
# The agent connects to it at runtime via HTTP — start weather-mcp-server.py first.
DEFAULT_PORT = 8007
port = int(os.getenv("PORT", DEFAULT_PORT))

# Token cap applied to every LLM call. Keeps demo costs bounded.
# Override via environment variable if longer responses are needed.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))

def model_factory():
    """Create a ChatOpenAI model instance with the global max token setting.

    Called once per agent so each sub-agent gets its own isolated LLM instance.
    Using gpt-4o for strong instruction-following across all three specialist agents.
    """
    return ChatOpenAI(model="gpt-4o", max_tokens=MAX_OUTPUT_TOKENS)

# --- Stub booking tools ---
# These are demo implementations that return a success string immediately.
# In a real system, replace the function bodies with actual booking API calls.

@tool("okahu_demo_lg_tool_book_hotel", description="Book a hotel for a stay")
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

@tool("okahu_demo_lg_tool_book_flight", description="Book a flight from one airport to another")
def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

# --- Weather tool via MCP (Model Context Protocol) ---
# The weather tool is NOT defined here — it lives in weather-mcp-server.py and is
# fetched dynamically over HTTP. This keeps the weather service decoupled and
# independently runnable, and exercises the MCP tool-discovery mechanism.
async def get_mcp_tools():
    """Get tools from the monocle MCP server.

    Connects to the weather MCP server at localhost:{port} and retrieves
    its tool definitions. The agent treats these exactly like locally defined tools.
    Raises RuntimeError with a clear message if the server is not running.
    """
    client = MultiServerMCPClient(
        {
            "weather": {
                "url": f"http://localhost:{port}/weather/mcp/",
                "transport": "streamable_http",  # HTTP-based MCP transport (not stdio)
            }
        }
    )
    try:
        return await client.get_tools()
    except Exception as exc:
        logger.error("Weather MCP server unavailable. Start the weather-mcp-server.py before running the agent.")
        raise RuntimeError("Weather MCP server unavailable. Please start weather-mcp-server.py.") from exc

# --- Agent graph construction ---
# Three specialist ReAct agents sit under a single supervisor.
# The supervisor interprets the user's full request and delegates each part
# to the appropriate sub-agent. Sub-agents are narrowly scoped by their
# system prompts so they ignore tasks outside their domain.
#
# Delegation flow:
#   User prompt
#       └─► Supervisor (gpt-4o)
#               ├─► flight_assistant  ──► book_flight tool
#               ├─► hotel_assistant   ──► book_hotel tool
#               └─► weather_agent     ──► demo_get_weather (MCP → HTTP → weather server)
async def setup_agents(return_all_agents: bool = False):
    # Fetch weather tools from the MCP server before building the graph.
    # This will raise if the weather server is not already running.
    weather_tools = await get_mcp_tools()

    # Flight booking sub-agent: handles only flight reservations.
    # Instructed to extract origin/destination and ignore unrelated requests.
    flight_assistant = create_react_agent(
    model=model_factory(),
        tools=[book_flight],
        prompt="You are a flight booking assistant. You only handle flight booking. Just handle that part from what the user says, ignore other parts of the requests.",
        name="okahu_demo_lg_agent_air_travel_assistant"
    )

    # Hotel booking sub-agent: handles only hotel reservations.
    # Will only book if explicitly asked — avoids false positives on ambiguous requests.
    hotel_assistant = create_react_agent(
    model=model_factory(),
        tools=[book_hotel],
        prompt="You are a hotel booking assistant. You only handle hotel booking. Book hotel if the user explicitly asks, just handle that part from what the user says, ignore other parts of the requests.",
        name="okahu_demo_lg_agent_lodging_assistant"
    )

    # Weather sub-agent: uses tools fetched live from the MCP server.
    # Extracts the city name from the user query before calling the weather tool.
    weather_agent = create_react_agent(
    model=model_factory(),
        tools=weather_tools,
        prompt="You are a weather information assistant. Please use the tool available to you for checking weather. Extract city name from the user query and pass it to the weather tool, and ignore other parts of the requests.",
        name="okahu_demo_lg_agent_weather_assistant"
    )

    # Supervisor: routes the user's request to whichever sub-agents are needed.
    # A single user message can trigger multiple sub-agents (e.g. book flight + hotel + check weather).
    # compile() finalizes the LangGraph state machine so it can be invoked.
    supervisor = create_supervisor(
        supervisor_name="okahu_demo_lg_agent_travel_supervisor",
        agents=[flight_assistant, hotel_assistant, weather_agent],
    model=model_factory(),
        prompt=(
            "You manage a hotel booking assistant and a"
            "flight booking assistant. Assign work to them. Each assistant is skilled in their own area ONLY and cannot do other tasks. "
            "If the user asks for weather information, delegate to the weather assistant."
        )
    ).compile()

    # return_all_agents=True is used by the test harness to get direct handles
    # to each sub-agent for targeted span-level assertions.
    if return_all_agents:
        return supervisor, flight_assistant, hotel_assistant, weather_agent
    return supervisor

# --- Main entry point for running a single request ---
async def run_agent(request: str):
    """Build the agent graph and invoke it with the user's request.

    Prints and returns the final message content from the supervisor.
    If the weather MCP server is not running, prints the error and exits cleanly.
    """
    try:
        supervisor = await setup_agents()
    except RuntimeError as exc:
        print(exc)
        return

    # ainvoke sends the user message into the graph and waits for full completion.
    # The supervisor may call multiple sub-agents before producing the final message.
    chunk = await supervisor.ainvoke(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": request
            }
        ]
    })

    # The last message in the list is the supervisor's final consolidated response.
    print(chunk["messages"][-1].content)
    return chunk["messages"][-1].content

if __name__ == "__main__":
    # CLI mode: prompt the user interactively, then run the agent once.
    # For batch or API usage, import and call run_agent() directly.
    logging.basicConfig(level=logging.WARN)
    request = input("\nI am a travel booking agent. How can I assist you with your travel plans? (You can ask me to book flights, hotels, or check the weather at any location.): \n")
    asyncio.run(run_agent(request))
