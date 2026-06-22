from asyncio import sleep
from typing import Optional
import pytest
import pytest_asyncio
from monocle_test_tools import TraceAssertion
from lg_travel_agent import setup_agents


# ---------------------------------------------------------------------------
# Framework-aware asserter
# ---------------------------------------------------------------------------

class AgentTypeTraceAssertion(TraceAssertion):
    """TraceAssertion subclass that stores a default agent framework type.

    Call with_agent_type() once (via the autouse fixture below) so that
    run_agent_async() does not need the framework string repeated on every call.
    The stored type is injected automatically at invocation time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_agent_type: Optional[str] = None

    def with_agent_type(self, agent_type: str) -> "AgentTypeTraceAssertion":
        """Set the default agent framework type for this asserter instance."""
        self._default_agent_type = agent_type
        return self

    async def run_agent_async(self, agent, *args, session_id: str = None, **kwargs):
        """Run the agent, injecting the stored agent type when not supplied."""
        if self._default_agent_type:
            return await self.validator.run_agent_async(
                agent, self._default_agent_type, *args,
                session_id=session_id, **kwargs
            )
        # Fall through to parent — requires agent_type as the first positional arg
        return await super().run_agent_async(agent, *args, session_id=session_id, **kwargs)


# ---------------------------------------------------------------------------
# Session-scoped agent setup
# ---------------------------------------------------------------------------

supervisor = None
flight_assistant = None
hotel_assistant = None
weather_agent = None

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_supervisor():
    """Build the travel booking supervisor and sub-agents once per session."""
    global supervisor, flight_assistant, hotel_assistant, weather_agent
    supervisor, flight_assistant, hotel_assistant, weather_agent = await setup_agents(return_all_agents=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _is_test_failed(request: pytest.FixtureRequest) -> bool:
    return request.node.rep_call.passed == False if hasattr(request.node, "rep_call") else False


@pytest.fixture()
def monocle_trace_asserter(request: pytest.FixtureRequest):
    """Override the plugin fixture to yield an AgentTypeTraceAssertion.

    Replicates all setup/cleanup logic from the plugin's built-in fixture so
    that span collection, trace export, and assertion reporting work identically.
    """
    asserter = AgentTypeTraceAssertion()
    asserter.cleanup()  # mirrors TraceAssertion.get_trace_asserter()
    token = asserter.validator.pre_test_run_setup(request.node.name)
    exception_message = None
    try:
        yield asserter
    except Exception as e:
        exception_message = str(e)
        raise
    finally:
        is_test_failed = _is_test_failed(request)
        assertion_messages = exception_message or asserter.get_assertion_messages() if is_test_failed else None
        asserter.validator.post_test_cleanup(token, request.node.name, is_test_failed, assertion_messages)
        asserter.cleanup()


@pytest.fixture(autouse=True)
def set_agent_framework(monocle_trace_asserter: AgentTypeTraceAssertion):
    """Set the agent framework type once per test.

    Every test in this file targets the same LangGraph supervisor, so the
    framework string is declared here rather than repeated in each call.
    """
    monocle_trace_asserter.with_agent_type("langgraph")


# ---------------------------------------------------------------------------
# Tests — identical to test_lg_travel_agent_fluent.py but without the
# repeated "langgraph" argument in every run_agent_async() call.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_okahu_tone_evaluation(monocle_trace_asserter: AgentTypeTraceAssertion):
    """v0: Basic sentiment, bias evaluation on trace - only specify eval name and expected value."""
    travel_request = "Book a flight from Rochester to New York City for July 5th 2026"
    await monocle_trace_asserter.run_agent_async(supervisor, travel_request)
    monocle_trace_asserter.with_evaluation("okahu").check_eval("sentiment", "positive")\
        .check_eval("bias", "unbiased")


@pytest.mark.asyncio
async def test_okahu_quality_evaluation(monocle_trace_asserter: AgentTypeTraceAssertion):
    """v0: Multiple evaluations on trace - frustration, hallucination, contextual_precision."""
    travel_request = "Please Book a flight from New York to Delhi for 1st Dec 2025. Book a flight from Delhi to Mumabi on January 1st. Then book a hotel room in Mumbai for 5th Jan 2026."
    await monocle_trace_asserter.run_agent_async(supervisor, travel_request)
    monocle_trace_asserter.with_evaluation("okahu").check_eval("frustration", "ok")\
        .check_eval("hallucination", "no_hallucination")
    monocle_trace_asserter.check_eval("contextual_precision", "high_precision")


@pytest.mark.asyncio
async def test_token_limit(monocle_trace_asserter: AgentTypeTraceAssertion):
    await monocle_trace_asserter.run_agent_async(supervisor, "Book a flight from Chennai to Mumbai on April 30th 2026. Book hotel Marriott in Central Mumbai. Also how is the weather going to be in Mumbai?")
    monocle_trace_asserter.under_token_limit(5000)


@pytest.mark.asyncio
async def test_agent_and_tool_invocation(monocle_trace_asserter: AgentTypeTraceAssertion):
    await monocle_trace_asserter.run_agent_async(supervisor,
                    "Book a flight from San Francisco to Mumbai on April 30th 2026. Book hotel Marriott in Central Mumbai. Also how is the weather going to be in Mumbai?")

    monocle_trace_asserter.called_tool("okahu_demo_lg_tool_book_flight", "okahu_demo_lg_agent_air_travel_assistant") \
        .contains_input("Mumbai").contains_input("San Francisco") \
        .contains_output("flight").contains_output("from San Francisco to Mumbai").contains_output("booked")

    monocle_trace_asserter.called_tool("okahu_demo_lg_tool_book_hotel", "okahu_demo_lg_agent_lodging_assistant") \
        .contains_input("Marriott").contains_input("Mumbai") \
        .contains_output("Marriott") \
        .contains_output("Central Mumbai") \
        .contains_output("booked")

    monocle_trace_asserter.called_tool("demo_get_weather", "okahu_demo_lg_agent_weather_assistant") \
        .contains_input("city").contains_input("Mumbai") \
        .contains_output("temperature")

    monocle_trace_asserter.called_agent("okahu_demo_lg_agent_weather_assistant") \
        .contains_input("Book a flight from San Francisco to Mumbai on April 30th 2026. Book hotel Marriott in Central Mumbai. Also how is the weather going to be in Mumbai?") \
        .contains_output("weather") \
        .contains_output("Mumbai")

    monocle_trace_asserter.called_agent("okahu_demo_lg_agent_lodging_assistant") \
        .contains_input("Book a flight from San Francisco to Mumbai on April 30th 2026. Book hotel Marriott in Central Mumbai. Also how is the weather going to be in Mumbai?") \
        .contains_output("Marriott") \
        .contains_output("Central Mumbai") \
        .contains_output("successfully") \
        .contains_output("booked")

    monocle_trace_asserter.called_agent("okahu_demo_lg_agent_air_travel_assistant") \
        .contains_input("Book a flight from San Francisco to Mumbai on April 30th 2026. Book hotel Marriott in Central Mumbai. Also how is the weather going to be in Mumbai?") \
        .contains_output("San Francisco to Mumbai") \
        .contains_output("successfully") \
        .contains_output("booked")


@pytest.mark.asyncio
async def test_tool_invocation(monocle_trace_asserter: AgentTypeTraceAssertion):
    await monocle_trace_asserter.run_agent_async(supervisor,
                    "Book a flight from Chennai to Mumbai on April 30th 2026. Book hotel Marriott in Central Mumbai. Also how is the weather going to be in Mumbai?")

    monocle_trace_asserter.called_tool("okahu_demo_lg_tool_book_flight", "okahu_demo_lg_agent_air_travel_assistant") \
        .contains_input("Taiwan").contains_input("Chennai") \
        .contains_output("Successfully").contains_output("booked") \
        .contains_output("flight").contains_output("Chennai to Mumbai")

    monocle_trace_asserter.called_tool("okahu_demo_lg_tool_book_hotel", "okahu_demo_lg_agent_lodging_assistant") \
        .contains_input("Marriott").contains_input("Mumbai") \
        .contains_output("Marriott") \
        .contains_output("Central Mumbai") \
        .contains_output("booked")

    monocle_trace_asserter.called_tool("demo_get_weather", "okahu_demo_lg_agent_weather_assistant") \
        .contains_input("city").contains_input("Mumbai") \
        .contains_output("temperature")


@pytest.mark.asyncio
async def test_agent_invocation(monocle_trace_asserter: AgentTypeTraceAssertion):
    await monocle_trace_asserter.run_agent_async(supervisor,
                        "Book a flight from Chennai to Bengaluru for 28th April 2026. Book a two delux rooms at ITC Hotel at Bengaluru for 29th April 2026 for 5 nights")

    monocle_trace_asserter.called_agent("okahu_demo_lg_agent_air_travel_assistant") \
        .contains_input("Book a flight from Chennai to Bengaluru for 28th April 2026. Book a two delux rooms at ITC Hotel at Bengaluru for 29th April 2026 for 5 nights") \
        .contains_output("flight") \
        .contains_output("Chennai to Bengaluru") \
        .contains_output("successfully") \
        .contains_output("booked")

    monocle_trace_asserter.called_agent("okahu_demo_lg_agent_lodging_assistant") \
        .contains_input("Book a flight from Chennai to Bengaluru for 28th April 2026. Book a two delux rooms at ITC Hotel at Bengaluru for 29th April 2026 for 5 nights") \
        .contains_output("ITC Hotel") \
        .contains_output("Bengaluru") \
        .contains_output("successfully") \
        .contains_output("booked")


@pytest.mark.asyncio
async def test_individual_flight_agent(monocle_trace_asserter: AgentTypeTraceAssertion):
    request = "Book a flight from Seattle to Tokyo"
    await monocle_trace_asserter.run_agent_async(flight_assistant, request)

    monocle_trace_asserter.called_tool("okahu_demo_lg_tool_book_flight", "okahu_demo_lg_agent_air_travel_assistant") \
        .does_not_contain_output("booked")\
        .under_token_limit(5000)\
        .under_duration(300)

    monocle_trace_asserter.with_evaluation("okahu").check_eval("sentiment", "positive")\
        .check_eval("bias", "unbiased")


@pytest.mark.asyncio
async def test_individual_hotel_agent(monocle_trace_asserter: AgentTypeTraceAssertion):
    request = "Book hotel Taj Palace in New Delhi"
    await monocle_trace_asserter.run_agent_async(hotel_assistant, request)

    monocle_trace_asserter.called_tool("okahu_demo_lg_tool_book_hotel", "okahu_demo_lg_agent_lodging_assistant") \
        .does_not_contain_output("booked")\
        .under_token_limit(5000)\
        .under_duration(300)

    monocle_trace_asserter.with_evaluation("okahu").check_eval("frustration", "ok")\
        .check_eval("hallucination", "no_hallucination")


@pytest.mark.asyncio
async def test_individual_weather_agent(monocle_trace_asserter: AgentTypeTraceAssertion):
    request = "What is the weather like tomorrow ?"
    await monocle_trace_asserter.run_agent_async(weather_agent, request)

    monocle_trace_asserter.called_tool("demo_get_weather", "okahu_demo_lg_agent_weather_assistant") \
        .does_not_contain_output("temperature")\
        .under_token_limit(5000)\
        .under_duration(300)

    monocle_trace_asserter.with_evaluation("okahu").check_eval("contextual_precision", "high_precision")


if __name__ == "__main__":
    pytest.main([__file__])
