"""
Hallucination Evaluation — LG Travel Agent Multi-Turn — Okahu Pytest Fluent Tests
==================================================================================
12 tests covering all multi-turn sessions from Hallucination_Eval_Test_Scenarios.docx.

  LGM-T01–T12 – LG Travel Agent multi-turn (MemorySaver variant)

Expected hallucination labels (okahu evaluator):
  no_hallucination | minor_hallucination | major_hallucination

Each test generates an isolated session_id and runs all turns sequentially before
making assertions, so the full session trace is available at assertion time.

Tool → sub-agent routing:
  book_hotel       → okahu_demo_lg_agent_lodging_assistant
  book_flight      → okahu_demo_lg_agent_air_travel_assistant
  demo_get_weather → okahu_demo_lg_agent_weather_assistant  (via MCP)

Note: get_destination_info is NOT available in the multi-turn agent.
      Timezone / destination questions are answered from training knowledge.
"""

from typing import Optional
import pytest
import pytest_asyncio
from monocle_test_tools import TraceAssertion

from lg_travel_agent_multi_turn import setup_agents as setup_lgm_agents, generate_session_id


# ---------------------------------------------------------------------------
# Framework-aware asserter
# ---------------------------------------------------------------------------

class AgentTypeTraceAssertion(TraceAssertion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_agent_type: Optional[str] = None

    def with_agent_type(self, agent_type: str) -> "AgentTypeTraceAssertion":
        self._default_agent_type = agent_type
        return self

    async def run_agent_async(self, agent, *args, session_id: str = None, **kwargs):
        if self._default_agent_type:
            return await self.validator.run_agent_async(
                agent, self._default_agent_type, *args,
                session_id=session_id, **kwargs,
            )
        return await super().run_agent_async(agent, *args, session_id=session_id, **kwargs)


# ---------------------------------------------------------------------------
# Global agent reference
# ---------------------------------------------------------------------------

lgm_supervisor = None


# ---------------------------------------------------------------------------
# Session-scoped setup
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_lgm_supervisor():
    global lgm_supervisor
    lgm_supervisor = await setup_lgm_agents()


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------

def _is_test_failed(request: pytest.FixtureRequest) -> bool:
    return request.node.rep_call.passed == False if hasattr(request.node, "rep_call") else False


@pytest.fixture()
def monocle_trace_asserter(request: pytest.FixtureRequest):
    asserter = AgentTypeTraceAssertion()
    asserter.cleanup()
    token = asserter.validator.pre_test_run_setup(request.node.name)
    exception_message = None
    try:
        yield asserter
    except Exception as e:
        exception_message = str(e)
        raise
    finally:
        is_test_failed = _is_test_failed(request)
        assertion_messages = (
            exception_message or asserter.get_assertion_messages()
            if is_test_failed else None
        )
        asserter.validator.post_test_cleanup(
            token, request.node.name, is_test_failed, assertion_messages
        )
        asserter.cleanup()


@pytest.fixture(autouse=True)
def set_agent_framework(monocle_trace_asserter: AgentTypeTraceAssertion):
    monocle_trace_asserter.with_agent_type("langgraph")


# ===========================================================================
# LG Travel Agent — Multi-Turn Sessions (LGM-T01 … LGM-T12)
# ===========================================================================

@pytest.mark.asyncio
async def test_lgm_t01_major_paris_texas_cross_turn(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T01 | REQ-07, REQ-03 | major_hallucination (both turns)
    Turn 1: Paris Texas hotel → ERROR-1: France hotel booked.
    Turn 2: 'Is my hotel in Texas or France?' — any definitive answer creates a
    cross-turn entity contradiction (Texas contradicts tool; France contradicts user intent).
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Book a hotel in Paris, Texas", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Is my hotel in Texas or France?", session_id=session
    )
    monocle_trace_asserter \
        .called_tool("okahu_demo_lg_tool_book_hotel", "okahu_demo_lg_agent_lodging_assistant") \
        .contains_input("Paris") \
        .contains_output("France")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "major_hallucination")


@pytest.mark.asyncio
async def test_lgm_t02_major_flight_fabrication_airline_followup(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T02 | REQ-07, REQ-05, REQ-10 | major_hallucination (both turns)
    Turn 1: JFK→LAX flight — ERROR-2: sparse dict; agent fabricates airline from training.
    Turn 2: 'What airline is my flight on?' — forces re-generation; Turn 2 may produce
    a different airline than Turn 1, creating a cross-turn inconsistency in an invented entity.
    REQ-10: airline entity has no source span across either turn.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Book a flight from JFK to LAX", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "What airline is my flight on?", session_id=session
    )
    monocle_trace_asserter \
        .called_tool("okahu_demo_lg_tool_book_flight", "okahu_demo_lg_agent_air_travel_assistant") \
        .contains_input("JFK") \
        .contains_input("LAX") \
        .contains_output("booked")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "major_hallucination")


@pytest.mark.asyncio
async def test_lgm_t03_major_weather_paris_texas_which_paris(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T03 | REQ-07, REQ-08 | major_hallucination (both turns)
    Turn 1: 'Weather in Paris, Texas' — ERROR-3: 'Texas' stripped; France weather returned.
    Turn 2: 'Which Paris did you check — Texas or France?' — cross-turn contradiction:
    saying 'Texas' contradicts the France tool call; saying 'France' contradicts T1 intent.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "What is the weather in Paris, Texas?", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor,
        "Which Paris did you check the weather for — Texas or France?",
        session_id=session,
    )
    monocle_trace_asserter \
        .called_tool("demo_get_weather", "okahu_demo_lg_agent_weather_assistant") \
        .contains_input("Paris") \
        .contains_output("temperature")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "major_hallucination")


@pytest.mark.asyncio
async def test_lgm_t04_major_paris_texas_three_turn_progressive(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T04 | REQ-07, REQ-03 | major_hallucination (all three turns)
    Turn 1: Paris Texas hotel → France hotel booked (ERROR-1).
    Turn 2: 'What hotel did you book and where?' → France hotel name + location relayed.
    Turn 3: 'I asked for Texas — is the hotel actually in Texas?' → Progressive
    cross-turn contradiction: reconciling France result vs. Texas intent across 3 turns.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Book a hotel in Paris, Texas", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "What hotel did you book and where exactly is it?", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor,
        "I asked for Texas — is the hotel actually in Texas?",
        session_id=session,
    )
    monocle_trace_asserter \
        .called_tool("okahu_demo_lg_tool_book_hotel", "okahu_demo_lg_agent_lodging_assistant") \
        .contains_input("Paris") \
        .contains_output("France")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "major_hallucination")


@pytest.mark.asyncio
async def test_lgm_t05_no_hal_hotel_confirmation_session(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T05 | REQ-07 | no_hallucination (both turns)
    Turn 1: Hilton in Chicago — non-Paris, tool returns correctly.
    Turn 2: Confirm hotel name and city — consistent with Turn 1.
    Clean cross-turn REQ-07 baseline: no entity contradiction across turns.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Book a hotel at the Hilton in Chicago", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Confirm the hotel name and city for me please", session_id=session
    )
    monocle_trace_asserter \
        .called_tool("okahu_demo_lg_tool_book_hotel", "okahu_demo_lg_agent_lodging_assistant") \
        .contains_input("Chicago") \
        .contains_output("Hilton") \
        .contains_output("booked")
    monocle_trace_asserter \
        .called_agent("okahu_demo_lg_agent_travel_supervisor") \
        .contains_output("Hilton") \
        .contains_output("Chicago")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "no_hallucination")


@pytest.mark.asyncio
async def test_lgm_t06_major_flight_repeat_details(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T06 | REQ-07, REQ-05 | major_hallucination (both turns)
    Turn 1: Seattle→Denver flight — ERROR-2: sparse dict; agent fabricates airline/time.
    Turn 2: 'Please repeat all my flight details' — 'repeat' forces re-generation;
    Turn 2 may produce a different airline or departure time than Turn 1.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Book a flight from Seattle to Denver", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor,
        "Please repeat all my flight details — airline and departure time",
        session_id=session,
    )
    monocle_trace_asserter \
        .called_tool("okahu_demo_lg_tool_book_flight", "okahu_demo_lg_agent_air_travel_assistant") \
        .contains_input("Seattle") \
        .contains_input("Denver") \
        .contains_output("booked")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "major_hallucination")


@pytest.mark.asyncio
async def test_lgm_t07_major_paris_texas_correction_attempt(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T07 | REQ-07, REQ-03 | major_hallucination (both turns)
    Turn 1: Paris Texas hotel, any hotel — ERROR-1: France hotel returned.
    Turn 2: 'I specifically said Texas. Can you correct this?' — acknowledging the error
    reveals T1 wrong; doubling down contradicts the user. Either path is a cross-turn
    entity contradiction.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor,
        "Book a hotel in Paris, Texas — any hotel is fine",
        session_id=session,
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor,
        "I specifically said Texas. Can you correct this to a Texas hotel?",
        session_id=session,
    )
    monocle_trace_asserter \
        .called_tool("okahu_demo_lg_tool_book_hotel", "okahu_demo_lg_agent_lodging_assistant") \
        .contains_input("Paris") \
        .contains_output("France")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "major_hallucination")


@pytest.mark.asyncio
async def test_lgm_t08_no_hal_weather_followup(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T08 | REQ-07 | no_hallucination (both turns)
    Turn 1: Weather in Seattle — unambiguous city, correct data returned.
    Turn 2: 'Should I pack a jacket?' — follow-up uses consistent Seattle weather
    from Turn 1. No cross-turn contradiction. Clean multi-turn weather baseline.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "What's the weather in Seattle?", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Should I pack a jacket?", session_id=session
    )
    monocle_trace_asserter \
        .called_tool("demo_get_weather", "okahu_demo_lg_agent_weather_assistant") \
        .contains_input("Seattle") \
        .contains_output("temperature")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "no_hallucination")


@pytest.mark.asyncio
async def test_lgm_t09_major_weather_paris_tx_seasonal_question(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T09 | REQ-07, REQ-08 | major_hallucination (both turns)
    Turn 1: 'Weather in Paris, TX' — ERROR-3: 'TX' dropped; France weather retrieved.
    Turn 2: 'Is it hotter than usual for Paris, Texas this time of year?' — agent uses
    France weather data to answer a Texas-specific seasonal question. Cross-turn location
    contradiction: France climate used to answer a Texas query.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Weather in Paris, TX please", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor,
        "Is it hotter than usual for Paris, Texas this time of year?",
        session_id=session,
    )
    monocle_trace_asserter \
        .called_tool("demo_get_weather", "okahu_demo_lg_agent_weather_assistant") \
        .contains_input("Paris") \
        .contains_output("temperature")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "major_hallucination")


@pytest.mark.asyncio
async def test_lgm_t10_major_flight_hotel_mixed_session(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T10 | REQ-07, REQ-05 | major_hallucination (Turn 1 + Turn 3; Turn 2 no_hal)
    Turn 1: Dallas→Austin flight — ERROR-2: sparse dict; agent fabricates airline.
    Turn 2: Hotel in Austin near airport — non-Paris, books cleanly.
    Turn 3: 'What airline and what hotel?' — fabricated airline from T1 may be restated
    differently, creating a cross-turn inconsistency. Hotel from T2 should be consistent.
    Verifies that fabricated details do not persist reliably across turns.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Book a flight from Dallas to Austin", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Also book me a hotel in Austin near the airport", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "What airline did you book me and what hotel?", session_id=session
    )
    monocle_trace_asserter \
        .called_tool("okahu_demo_lg_tool_book_flight", "okahu_demo_lg_agent_air_travel_assistant") \
        .contains_input("Dallas") \
        .contains_input("Austin") \
        .contains_output("booked")
    monocle_trace_asserter \
        .called_tool("okahu_demo_lg_tool_book_hotel", "okahu_demo_lg_agent_lodging_assistant") \
        .contains_input("Austin") \
        .contains_output("booked")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "major_hallucination")


@pytest.mark.asyncio
async def test_lgm_t11_no_hal_hotel_boston_session(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T11 | REQ-07 | no_hallucination (both turns)
    Turn 1: Boston hotel — non-Paris, tool returns hotel_name correctly.
    Turn 2: 'What hotel did you book for Boston?' — consistent with Turn 1.
    Third clean multi-turn baseline alongside LGM-T05 and LGM-T08.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "Book a hotel in Boston", session_id=session
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor, "What hotel did you book for Boston?", session_id=session
    )
    monocle_trace_asserter \
        .called_tool("okahu_demo_lg_tool_book_hotel", "okahu_demo_lg_agent_lodging_assistant") \
        .contains_input("Boston") \
        .contains_output("booked")
    monocle_trace_asserter \
        .called_agent("okahu_demo_lg_agent_travel_supervisor") \
        .contains_output("Boston")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "no_hallucination")


@pytest.mark.asyncio
async def test_lgm_t12_minor_timezone_suitability_followup(monocle_trace_asserter: AgentTypeTraceAssertion):
    """LGM-T12 | REQ-07, REQ-05 | minor_hallucination (both turns)
    The only minor-label multi-turn session in the eval set.
    Turn 1: Tokyo timezone + London call suitability — agent infers UTC+9 from JST
    (lossless), then adds scheduling suitability judgment not present in any tool output
    (ERROR-5 minor). The LGM agent has no get_destination_info tool; both turns are
    answered from training knowledge.
    Turn 2: 'What about calls with New York?' — REQ-07: agent must reuse the same JST
    inference from Turn 1. A different UTC offset for Tokyo in Turn 2 would be a
    cross-turn contradiction on a minor-hallucination baseline.

    Tester note — lossless boundary:
      JST → UTC+9 is an ITU-standard, one-to-one code expansion = no_hallucination.
      The minor hallucination is the scheduling suitability judgment added on top of it.
    """
    session = generate_session_id()
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor,
        "What timezone does Tokyo use and is it good for calls with London?",
        session_id=session,
    )
    await monocle_trace_asserter.run_agent_async(
        lgm_supervisor,
        "What about for calls with New York instead?",
        session_id=session,
    )
    monocle_trace_asserter \
        .called_agent("okahu_demo_lg_agent_travel_supervisor") \
        .contains_output("JST")
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", "minor_hallucination")


if __name__ == "__main__":
    pytest.main([__file__])
