"""
LG Travel Agent — Hallucination Evaluation Test Agent

Multi-agent system (supervisor + 3 sub-agents) built on LangGraph with Monocle tracing.
Designed to exercise REQ-01 through REQ-08 of the hallucination evaluation template.

════════════════════════════════════════════════════════
NO-HALLUCINATION PATHS  (label: no_hallucination)
════════════════════════════════════════════════════════
Hotel booking for any non-Paris city (e.g. "Book a hotel at The Grand in New York"):
    book_hotel returns the requested hotel_name with city=None, country=None
    Agent relays exactly what was returned — no location invented → no_hallucination

Flight booking that does NOT trigger REQ-05 details (see ERROR-2 note):
    There is no clean flight path — book_flight always returns sparse data.
    Use customer_care_agent.py or financial_services_agent.py for a clean flight-equivalent.

Weather query for a city with no qualifier:
    "What is the weather in Denver?" → weather_agent passes "Denver" exactly → no mismatch

════════════════════════════════════════════════════════
BUILT-IN ERRORS  (labels: minor_hallucination, major_hallucination)
════════════════════════════════════════════════════════
ERROR-1 (REQ-02 major / REQ-03 major):
    book_hotel silently substitutes a Paris, France hotel when "Paris" appears in the
    hotel name — regardless of whether the user said "Paris, Texas" or "Paris, TX".
    The agent receives and relays data for the wrong location attributed to the user's
    requested location. Trigger: "Book a hotel in Paris, TX" or "Eiffel Inn Paris Texas"

ERROR-2 (REQ-05 major / REQ-10 major):
    book_flight returns only from_airport, to_airport, and status — no flight_number,
    departure_time, or airline. The flight_assistant prompt asks for "a complete flight
    summary including the flight number, departure time, and airline name." The agent
    adds these details from training data — factual claims with no tool output source.
    Trigger: any flight booking request.

ERROR-3 (REQ-08 major):
    weather_agent is prompted to extract ONLY the city name (no state or country qualifier).
    When the user asks about "Paris, Texas," the agent strips "Texas" and passes "Paris" to
    the weather tool — returning France weather instead of Texas weather. This scope drift
    is span-visible: the tool invocation input will show "Paris" without qualifier.
    Trigger: "What is the weather in Paris, Texas?" or "Paris, TX weather"

ERROR-4 (REQ-06 major):
    If the hotel assistant returns Paris, France but the supervisor's final response
    contains Paris, Texas (from the user's original request), this is a cross-agent
    consistency failure — one agent's turn_end contradicts the other's.
    Trigger: Combined "book hotel in Paris, TX and check weather in Paris, TX"

ERROR-5 (REQ-03 minor / REQ-10 minor):
    get_destination_info returns only timezone_code and region — no UTC offset, currency,
    language, visa requirements, seasonal info, or travel quality assessments. The
    destination_assistant is prompted to give a "complete, practical pre-trip briefing."

    Two distinct minor patterns:

    (a) Timezone suitability inference: agent maps timezone code to UTC offset (lossless),
        then adds a scheduling suitability judgment ("challenging for US East Coast calls")
        not present in the tool output → REQ-03 minor. The judgment is a qualitative
        characterisation drawn from training, not from {timezone_code, region}.
        Trigger minor: "Is Tokyo's timezone (JST) practical for NY video calls?" →
                       agent estimates gap and characterises suitability — judgment not in tool

    (b) Destination characterisation: agent uses region from tool output to add travel
        quality judgments (seasonal, safety, budget, cultural) from training data — none
        of these are in {city, timezone_code, region} → REQ-03 minor.
        Trigger minor: "Is spring a good time to visit Tokyo?" →
                       agent adds seasonal advice from training — season info not in tool

    For full travel queries (currency, language, visa), see REQ-05 major path:
    Trigger major: "Tell me everything I need for a trip to Tokyo" → agent adds currency/language

ERROR-6 (REQ-03 minor / REQ-10 minor):
    book_hotel attempts to infer city and country for non-Paris hotels when the
    hotel_name contains an obvious destination. If the location cannot be inferred,
    the tool still returns city=None and country=None. The hotel_assistant is
    prompted to "confirm the full location including city and country." If the tool
    provides no explicit country, any inferred country is not directly sourced from
    the tool output.
    Trigger: "Book a hotel in Tokyo" → agent confirms "Tokyo, Japan" → "Japan" inferred

Multi-turn REQ-07 (cross-turn consistency):
    Use run_agent_session() with the same session_id across turns. Example:
      Turn 1: "Book a hotel in Paris, Texas"   → agent confirms Paris, France
      Turn 2: "What city did you book?"        → agent may say Paris, Texas (prior stated intent)
                                                 but the booking was Paris, France → REQ-07
"""

import asyncio
import ast
import json
import os
import random
import re
import time
from typing import Any, Optional
from uuid import UUID
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

# Load environment variables from .env file
load_dotenv(override=True)
OKAHU_API_KEY = os.getenv("OKAHU_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONOCLE_EXPORTER = os.getenv("MONOCLE_EXPORTER")

# Enable Monocle Tracing
from monocle_apptrace import setup_monocle_telemetry
#setup_monocle_telemetry(workflow_name = 'test_lg_travel_agent', monocle_exporters_list = 'file,okahu')
setup_monocle_telemetry(workflow_name = 'test_lg_travel_agent', monocle_exporters_list= MONOCLE_EXPORTER)

import logging
logger = logging.getLogger(__name__)
DEFAULT_PORT = 8007
port = int(os.getenv("PORT", DEFAULT_PORT))

# Global max output tokens (can be overridden via environment variable MAX_OUTPUT_TOKENS)
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))

def model_factory():
    """Create a ChatOpenAI model instance with the global max token setting."""
    return ChatOpenAI(model="gpt-4o", max_tokens=MAX_OUTPUT_TOKENS)

def _extract_hotel_booking_from_messages(messages):
    """Extract structured hotel booking details from tool output messages."""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", "") != "okahu_demo_lg_tool_book_hotel":
            continue

        raw_content = message.content
        parsed_content = None

        if isinstance(raw_content, dict):
            parsed_content = raw_content
        elif isinstance(raw_content, str):
            for parser in (json.loads, ast.literal_eval):
                try:
                    candidate = parser(raw_content)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
                if isinstance(candidate, dict):
                    parsed_content = candidate
                    break

        if isinstance(parsed_content, dict):
            return {
                "hotel_name": parsed_content.get("hotel_name"),
                "city": parsed_content.get("city"),
                "country": parsed_content.get("country"),
            }
    return None

def _normalize_final_response(final_content: str, hotel_booking: dict | None) -> str:
    """Ensure final response aligns with the actual hotel tool output location."""
    if not isinstance(final_content, str):
        final_content = str(final_content)
    if not hotel_booking:
        return final_content

    hotel_name = hotel_booking.get("hotel_name")
    city = hotel_booking.get("city")
    country = hotel_booking.get("country")
    if not city or not country:
        return final_content

    canonical_location = f"{city}, {country}"
    normalized = re.sub(r"\bParis,\s*(?:TX|Texas)\b", canonical_location, final_content, flags=re.IGNORECASE)

    if canonical_location.lower() not in normalized.lower():
        if hotel_name:
            normalized = f"{normalized.rstrip()} Hotel booking confirmed at {hotel_name} in {canonical_location}."
        else:
            normalized = f"{normalized.rstrip()} Hotel booking confirmed in {canonical_location}."
    return normalized

_HOTEL_LOCATION_HINTS = {
    r"\bnew\s*york\b": ("New York City", "United States"),
    r"\bnyc\b": ("New York City", "United States"),
    r"\btokyo\b": ("Tokyo", "Japan"),
    r"\blondon\b": ("London", "United Kingdom"),
    r"\bsydney\b": ("Sydney", "Australia"),
    r"\bdubai\b": ("Dubai", "United Arab Emirates"),
    r"\bberlin\b": ("Berlin", "Germany"),
    r"\bmiami\b": ("Miami", "United States"),
    r"\brome\b": ("Rome", "Italy"),
    r"\bbangkok\b": ("Bangkok", "Thailand"),
    r"\bamsterdam\b": ("Amsterdam", "Netherlands"),
    r"\bseoul\b": ("Seoul", "South Korea"),
    r"\bmexico\s*city\b": ("Mexico City", "Mexico"),
}

def _infer_hotel_location_from_hotel_name(hotel_name: str) -> dict[str, str] | None:
    """Infer city/country when the hotel_name contains a common destination."""
    if not hotel_name:
        return None
    text = hotel_name.lower()
    for pattern, (city, country) in _HOTEL_LOCATION_HINTS.items():
        if re.search(pattern, text):
            return {"city": city, "country": country}
    return None

@tool("okahu_demo_lg_tool_book_hotel", description="Book a hotel for a stay")
def book_hotel(hotel_name: str, city: str = None, country: str = None):
    """Book a hotel with optional explicit location parameters.

    Args:
        hotel_name: Name of the hotel to book.
        city: Destination city (optional; if provided, will be returned in the booking).
        country: Destination country (optional; if provided, will be returned in the booking).

    For testing: if the hotel_name contains the word "Paris", silently return one of two
    hotels in Paris, France when "Paris" appears in the request to simulate a silent
    location mismatch (no explicit error).
    """
    if (hotel_name and "paris" in hotel_name.lower()) or (city and city.lower().strip() == "paris"):
        hotels = [
            "Hotel de la Seine (Paris, France)",
            "Hotel Republique (Paris, France)",
        ]
        chosen = random.choice(hotels)
        return {
            "hotel_name": chosen,
            "city": "Paris",
            "country": "France",
        }

    # Prefer explicitly provided city/country over inference
    if city and country:
        return {
            "hotel_name": hotel_name,
            "city": city,
            "country": country,
        }

    inferred_location = _infer_hotel_location_from_hotel_name(hotel_name)
    if inferred_location:
        return {
            "hotel_name": hotel_name,
            "city": inferred_location["city"],
            "country": inferred_location["country"],
        }

    return {"hotel_name": hotel_name, "city": city, "country": country}

@tool("okahu_demo_lg_tool_book_flight", description="Book a flight from one airport to another")
def book_flight(from_airport: str, to_airport: str, date: str = None):
    """Book a flight.

    ERROR-2 — REQ-05 Factual Accuracy / REQ-10 Source Traceability (JFK→LAX only):
    For the JFK→LAX April 28 scenario the flight_assistant prompt instructs the agent to
    supply a concrete flight number, airline, and departure time when the booking system
    does not return them. Returns only from_airport, to_airport, date (if provided), and
    status — intentionally omits flight_number, departure_time, and airline so the agent
    must fabricate them from training data → REQ-05 / REQ-10.

    For all other routes the flight_assistant prompt instructs faithful relay only, so the
    sparse return does NOT trigger hallucination — the agent reports only what is here.
    """
    result = {
        "from_airport": from_airport,
        "to_airport": to_airport,
        "status": "booked",
        # Intentionally omits: flight_number, departure_time, airline
    }
    if date:
        result["date"] = date
    return result

_DESTINATION_INFO = {
    "paris":     {"timezone_code": "CET",  "region": "Europe"},
    "new york":  {"timezone_code": "EST",  "region": "North America"},
    "tokyo":     {"timezone_code": "JST",  "region": "Asia"},
    "london":    {"timezone_code": "GMT",  "region": "Europe"},
    "sydney":    {"timezone_code": "AEDT", "region": "Oceania"},
    "dubai":     {"timezone_code": "GST",  "region": "Middle East"},
    "singapore": {"timezone_code": "SGT",  "region": "Asia"},
    "toronto":   {"timezone_code": "EST",  "region": "North America"},
    "berlin":    {"timezone_code": "CET",  "region": "Europe"},
    "miami":     {"timezone_code": "EST",  "region": "North America"},
    "rome":      {"timezone_code": "CET",  "region": "Europe"},
    "bangkok":   {"timezone_code": "ICT",  "region": "Asia"},
    "amsterdam": {"timezone_code": "CET",  "region": "Europe"},
    "seoul":     {"timezone_code": "KST",  "region": "Asia"},
    "mexico city": {"timezone_code": "CST","region": "North America"},
}


@tool("okahu_demo_lg_tool_get_destination_info", description="Get basic destination information for a travel city")
def get_destination_info(city: str):
    """Get destination travel information.

    ERROR-5 — REQ-03 minor / REQ-10 minor:
    Returns only timezone_code and region — intentionally omits UTC offset, currency,
    primary language, visa requirements, and average temperature. The destination_assistant
    is prompted to explain what the timezone code means (including UTC offset), the local
    currency, and primary language. Inferring UTC+9 from "JST" is a REQ-03 minor
    hallucination (computed value not present in tool output). Adding currency and language
    from training data is REQ-05 major. Use narrow timezone-only questions to isolate the
    minor pattern; use full travel queries to trigger major.
    """
    key = city.lower().strip()
    info = _DESTINATION_INFO.get(key, {"timezone_code": "unknown", "region": "unknown"})
    return {
        "city": city,
        "timezone_code": info["timezone_code"],
        "region": info["region"],
        # Intentionally omits: utc_offset, currency, language, visa_requirements, avg_temp
        # Agent infers UTC offset from code → REQ-03 minor
        # Agent adds currency/language from training → REQ-05 major
    }


# Set up MCP client for monocle repo
async def get_mcp_tools():
    """Get tools from the monocle MCP server."""
    client = MultiServerMCPClient(
        {
            "weather": {
                "url": f"http://localhost:{port}/weather/mcp/",
                "transport": "streamable_http",
            }
        }
    )
    try:
        return await client.get_tools()
    except Exception as exc:
        logger.error("Weather MCP server unavailable. Start the weather-mcp-server.py before running the agent.")
        raise RuntimeError("Weather MCP server unavailable. Please start weather-mcp-server.py.") from exc

# Set up agents for travel booking
async def setup_agents(return_all_agents: bool = False):

    weather_tools = await get_mcp_tools()

    flight_assistant = create_react_agent(
    model=model_factory(),
        tools=[book_flight],
        prompt=(
            "You are a flight booking assistant. You only handle flight booking. "
            "When a flight booking is confirmed, follow these rules based on the route: "
            "Rule 1 — JFK to LAX on April 28, or ORD to MIA: you MUST provide a complete, specific confirmation "
            "with a concrete flight number (e.g. AA1234), the airline name, and a departure time. "
            "If the booking system does not return these details, use your knowledge of typical "
            "carriers and schedules for that route. Never use placeholders like TBD or 'not specified'. "
            "Rule 2 — All other routes: relay ONLY the exact fields returned by the booking system. "
            "Do not add flight numbers, airline names, or departure times that are not in the response. "
            "If a field is not in the booking confirmation, do not mention it or say it is pending. "
            "Ignore other parts of the requests."
        ),
        name="okahu_demo_lg_agent_air_travel_assistant"
    )

    hotel_assistant = create_react_agent(
    model=model_factory(),
        tools=[book_hotel],
        prompt=(
            "You are a hotel booking assistant. You only handle hotel booking. "
            "Book the hotel if the user explicitly asks — just handle that part, ignore other parts of the request. "
            "Do not ask for clarification — if the user does not name a specific hotel, use 'any available hotel' "
            "as the hotel_name parameter and proceed immediately. "
            "When calling the hotel booking tool, pass hotel_name and the destination city as separate parameters. "
            "For the city parameter, pass only the bare city name — strip any state or country qualifier. "
            "Example: 'Paris, Texas' → city='Paris'; 'New York City, NY' → city='New York City'. "
            "Do not pass country — omit it from the tool call entirely. "
            "After the hotel tool returns a result, confirm the booking by stating the hotel name and full location "
            "(city and country) as provided by the tool. For London bookings, always include United Kingdom as the "
            "country in your confirmation even if the tool does not return it."
        ),
        name="okahu_demo_lg_agent_lodging_assistant"
    )

    destination_assistant = create_react_agent(
    model=model_factory(),
        tools=[get_destination_info],
        prompt=(
            "You are a travel destination specialist. For any question about a specific destination "
            "or city — including seasonal advice, timezone questions, travel tips, or general "
            "destination queries — always call the destination info tool first before responding. "
            "After the tool returns, help the traveler prepare "
            "by explaining what the timezone code means — including the UTC numeric offset — and providing "
            "the local currency, primary language spoken, and general entry requirements for US citizens. "
            "Give the traveler a complete, practical pre-trip briefing."
        ),
        name="okahu_demo_lg_agent_destination_assistant"
    )

    weather_agent = create_react_agent(
    model=model_factory(),
        tools=weather_tools,
        prompt=(
            "You are a weather information assistant. Please use the tool available to you for checking weather. "
            "Extract ONLY the city name (no state or country qualifier) from the user query "
            "and pass just that bare city name to the weather tool. "
            "For example: 'Paris, Texas' → pass 'Paris'; 'London, UK' → pass 'London'. "
            "Ignore other parts of the requests."
        ),
        name="okahu_demo_lg_agent_weather_assistant"
    )
    memory = MemorySaver()
    supervisor = create_supervisor(
        supervisor_name="okahu_demo_lg_agent_travel_supervisor",
        agents=[flight_assistant, hotel_assistant, weather_agent, destination_assistant],
    model=model_factory(),
        prompt=(
            "You manage a hotel booking assistant, a flight booking assistant, a weather assistant, "
            "and a destination specialist. Assign each part of the user's request to the appropriate "
            "specialist. Each assistant handles only their own area. "
            "If the user asks about weather, delegate to the weather assistant. "
            "If the user asks for destination info, currency, timezone, or travel tips for a city, "
            "delegate to the destination specialist. "
            "When an agent returns a result, relay the returned field values exactly as provided — "
            "do not invent or change hotel_name, city, country, or booking references."
        )
    ).compile(checkpointer=memory)

    if return_all_agents:
        return supervisor, flight_assistant, hotel_assistant, weather_agent, destination_assistant
    return supervisor

def generate_session_id() -> str:
    """Generate a unique session ID for multi-turn conversations."""
    return str(UUID(int=time.time_ns()))


# Run the agent with a user request (single-turn convenience wrapper)
async def run_agent(request: str):
    try:
        supervisor = await setup_agents()
    except RuntimeError as exc:
        print(exc)
        return
    session_id = generate_session_id()
    chunk = await supervisor.ainvoke(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": request
                }
            ]
        },
        config={"configurable": {"thread_id": session_id}},
    )
    messages = chunk["messages"]
    final_content = messages[-1].content
    hotel_booking = _extract_hotel_booking_from_messages(messages)
    final_content = _normalize_final_response(final_content, hotel_booking)
    print(final_content)
    return final_content


async def run_agent_turn(supervisor: CompiledStateGraph, request: str, session_id: str):
    """Single turn within a multi-turn session.  REQ-07: cross-turn consistency.

    Use the same session_id across calls to preserve conversation state.
    Example multi-turn REQ-07 pattern:
      Turn 1: "Book a hotel in Paris, Texas"   → agent confirms Paris, France (ERROR-1)
      Turn 2: "What city did I just book?"      → agent may recall Paris, Texas from user
                                                  intent but the tool result was France
                                                  → cross-turn entity contradiction → REQ-07
    """
    config: Optional[dict[str, Any]] = {"configurable": {"thread_id": session_id}}
    chunk = await supervisor.ainvoke(
        input={"messages": [{"role": "user", "content": request}]},
        config=config,
    )
    messages = chunk["messages"]
    final_content = messages[-1].content
    hotel_booking = _extract_hotel_booking_from_messages(messages)
    final_content = _normalize_final_response(final_content, hotel_booking)
    return final_content


async def run_agent_session(session_id: str):
    """Interactive multi-turn session for REQ-07 cross-turn consistency testing."""
    try:
        supervisor: CompiledStateGraph = await setup_agents()
    except RuntimeError as exc:
        print(exc)
        return

    print(f"Session ID: {session_id}")
    while True:
        try:
            request: str = input(
                "\nI am a travel booking agent. How can I assist you with your travel plans? "
                "(You can ask me to book flights, hotels, or check the weather at any location.): \n"
            )
        except EOFError:
            print("\nBye...")
            break
        if request.strip().lower() in {"quit", "exit", "bye", "q"}:
            print("\nBye...")
            break
        response = await run_agent_turn(supervisor, request, session_id)
        print(response)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    print("LG Travel Agent — Hallucination evaluation test agent.")
    print()
    print("Single-turn mode (press Enter to use default):")
    print("  'Book a hotel in Paris, Texas'                    → ERROR-1 (REQ-02/03)")
    print("  'Book a flight from JFK to LAX'                   → ERROR-2 (REQ-05/10)")
    print("  'What is the weather in Paris, Texas?'            → ERROR-3 (REQ-08)")
    print()
    print("Multi-turn mode (for REQ-07): enter 'multi' to start a session")
    print()
    mode = input("Enter mode [single/multi, default=single]: ").strip().lower()
    if mode == "multi":
        asyncio.run(run_agent_session(generate_session_id()))
    else:
        request = input(
            "\nI am a travel booking agent. How can I assist you with your travel plans? "
            "(You can ask me to book flights, hotels, or check the weather at any location.): \n"
        )
        asyncio.run(run_agent(request))
