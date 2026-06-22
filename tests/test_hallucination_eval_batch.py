"""
Hallucination Evaluation — Batch Run via Okahu — Parameterised Pytest
======================================================================
Drives all 58 single-turn hallucination scenarios from
okahu_eval_test_scenarios.json through the Okahu evaluator in a single
pytest session, with each trace labelled by scenario_id for post-run
comparison.

Key differences from test_hallucination_eval_single_turn_fluent.py
-------------------------------------------------------------------
1. Scenarios are loaded dynamically from ``okahu_eval_test_scenarios.json``
   — no hard-coded per-test functions, so adding/removing JSON scenarios
   automatically updates the test suite.

2. Every agent invocation uses a deterministic session_id:
       session_id = "hal_batch_<scenario_id>"   # e.g. "hal_batch_CC-T01"
   Monocle tags the emitted trace with this value, making every trace
   addressable by scenario after the run via Okahu's get_traces API.

3. ``compare_batch_results()`` is a standalone utility that, after the
   pytest run, retrieves each trace's eval label from Okahu (keyed by
   session_id) and compares it against the expected_outcome in the JSON,
   printing a per-scenario pass/fail report.

Traceability design
-------------------
  pytest runs → agent emits Monocle trace
             → trace stored in Okahu tagged session_id="hal_batch_CC-T01"
  Post-run:   → get_traces(workflow=<name>, session_id="hal_batch_CC-T01")
              → compare trace eval label vs expected_outcome from JSON
              → report pass/fail per scenario

Running the batch
-----------------
  # Full batch (all 58 scenarios):
  pytest tests/test_hallucination_eval_batch.py -v

  # Single scenario:
  pytest tests/test_hallucination_eval_batch.py -k "CC-T01"

  # Customer Care scenarios only:
  pytest tests/test_hallucination_eval_batch.py -k "CC-T"

Automated comparison (standalone after pytest run)
--------------------------------------------------
  python - <<'EOF'
  from tests.test_hallucination_eval_batch import compare_batch_results
  compare_batch_results()
  EOF

  Or as part of a CI step — see compare_batch_results() docstring for the
  recommended MCP tool sequence.

Coverage
--------
  CC-T01–T20  – Customer Care Agent          (20 scenarios)
  FS-T01–T20  – Financial Services Agent     (20 scenarios)
  LGS-T01–T18 – LG Travel Agent single-turn  (18 scenarios)
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional

import pytest
import pytest_asyncio
from monocle_test_tools import TraceAssertion

from hallucination_customer_care_agent import setup_agents as setup_cc_agents
from hallucination_financial_services_agent import setup_agents as setup_fs_agents
from hallucination_lg_travel_agent import setup_agents as setup_lgs_agents


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SCENARIOS_JSON = _REPO_ROOT / "okahu_eval_test_scenarios.json"

# ---------------------------------------------------------------------------
# Load hallucination scenarios from JSON
# ---------------------------------------------------------------------------

with open(_SCENARIOS_JSON) as _f:
    _ALL_DATA: Dict[str, Any] = json.load(_f)

HALLUCINATION_SCENARIOS: List[Dict[str, Any]] = [
    s for s in _ALL_DATA["scenarios"] if s["eval_type"] == "hallucination"
]

# Build expected-outcome lookup: scenario_id → expected_outcome label
EXPECTED_OUTCOMES: Dict[str, str] = {
    s["scenario_id"]: s["expected_outcome"] for s in HALLUCINATION_SCENARIOS
}

# ---------------------------------------------------------------------------
# Session-id convention
# ---------------------------------------------------------------------------

BATCH_PREFIX = "hal_batch"


def batch_session_id(scenario_id: str) -> str:
    """Return the deterministic session_id for a given scenario.

    Embeds the scenario_id so the Monocle trace can be retrieved from
    Okahu after the run using:
        get_traces(workflow=<workflow>, session_id="hal_batch_CC-T01")
    """
    return f"{BATCH_PREFIX}_{scenario_id}"


# ---------------------------------------------------------------------------
# Framework-aware asserter  (mirrors single-turn file)
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
# Global agent references  (populated by session fixture)
# ---------------------------------------------------------------------------

cc_supervisor        = None
cc_order_agent       = None
cc_eligibility_agent = None
cc_refund_agent      = None

fs_supervisor        = None
fs_account_agent     = None
fs_trade_agent       = None
fs_transfer_agent    = None
fs_suitability_agent = None

lgs_supervisor        = None
lgs_flight_agent      = None
lgs_hotel_agent       = None
lgs_weather_agent     = None
lgs_destination_agent = None


def _get_supervisor(agent_key: str):
    """Return the supervisor agent object for the given JSON agent key."""
    if agent_key == "customer_care_agent":
        return cc_supervisor
    if agent_key == "financial_services_agent":
        return fs_supervisor
    if agent_key == "lg_travel_agent":
        return lgs_supervisor
    raise ValueError(f"Unknown agent key: {agent_key!r}")


# ---------------------------------------------------------------------------
# Session-scoped setup  (mirrors single-turn file)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_all_agents():
    global cc_supervisor, cc_order_agent, cc_eligibility_agent, cc_refund_agent
    global fs_supervisor, fs_account_agent, fs_trade_agent, fs_transfer_agent, fs_suitability_agent
    global lgs_supervisor, lgs_flight_agent, lgs_hotel_agent, lgs_weather_agent, lgs_destination_agent

    (cc_supervisor,
     cc_order_agent,
     cc_eligibility_agent,
     cc_refund_agent) = setup_cc_agents(return_all_agents=True)

    (fs_supervisor,
     fs_account_agent,
     fs_trade_agent,
     fs_transfer_agent,
     fs_suitability_agent) = setup_fs_agents(return_all_agents=True)

    (lgs_supervisor,
     lgs_flight_agent,
     lgs_hotel_agent,
     lgs_weather_agent,
     lgs_destination_agent) = await setup_lgs_agents(return_all_agents=True)


# ---------------------------------------------------------------------------
# Per-test fixtures  (mirrors single-turn file)
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


# ---------------------------------------------------------------------------
# Parametrised batch test  (one pytest node per scenario)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    HALLUCINATION_SCENARIOS,
    ids=[s["scenario_id"] for s in HALLUCINATION_SCENARIOS],
)
async def test_hallucination_batch(
    scenario: Dict[str, Any],
    monocle_trace_asserter: AgentTypeTraceAssertion,
) -> None:
    """Run a single hallucination scenario through the Okahu evaluator.

    The test id is the scenario_id (e.g. ``test_hallucination_batch[CC-T01]``),
    making per-scenario pass/fail directly visible in the pytest report.

    Session-id convention:
        session_id = "hal_batch_<scenario_id>"
    Okahu stores the trace under this session_id, enabling post-run retrieval
    via get_traces(session_id="hal_batch_CC-T01").
    """
    scenario_id: str = scenario["scenario_id"]
    agent_key: str = scenario["agent"]
    message: str = scenario["user_input"][0]["message"]
    expected_label: str = scenario["expected_outcome"]
    sid: str = batch_session_id(scenario_id)

    supervisor = _get_supervisor(agent_key)
    await monocle_trace_asserter.run_agent_async(supervisor, message, session_id=sid)

    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", expected_label)


# ---------------------------------------------------------------------------
# Post-run comparison utility
# ---------------------------------------------------------------------------

def compare_batch_results(
    *,
    workflow_filter: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare Okahu batch eval results against JSON expected outcomes.

    This function is designed to be called AFTER the pytest batch run has
    completed and all traces have been submitted to Okahu.  It performs the
    following steps:

    1. Build the set of expected outcomes from ``okahu_eval_test_scenarios.json``
       (already loaded at module level as ``EXPECTED_OUTCOMES``).

    2. For each scenario, retrieve the stored trace from Okahu using the
       deterministic session_id (``"hal_batch_<scenario_id>"``).

       Recommended MCP call (from an MCP-enabled client such as Claude Code):
           mcp__okahu-prod-df__get_traces(
               workflow=<workflow_name>,
               session_id="hal_batch_CC-T01",
           )
       Repeat for each scenario_id, or filter by the ``hal_batch_`` prefix
       if the API supports prefix matching.

    3. Extract the ``eval_result.label`` from each returned trace and compare
       it against the expected_outcome.

    4. Return a structured comparison report:
       {
         "total":    58,
         "passed":   <n>,
         "failed":   <n>,
         "pass_rate": 0.xx,
         "details": [
           {
             "scenario_id":     "CC-T01",
             "session_id":      "hal_batch_CC-T01",
             "expected":        "major_hallucination",
             "actual":          "major_hallucination",
             "match":           True,
           },
           ...
         ]
       }

    Parameters
    ----------
    workflow_filter:
        Optional workflow name to pass to get_traces.  If None the function
        returns a skeleton report with ``actual=None`` for every scenario
        (useful for testing the comparison logic without a live Okahu instance).
    output_path:
        If provided, save the report as JSON to this path.

    Note
    ----
    This function does NOT make live API calls itself — it is structured so
    that you slot in the Okahu trace-retrieval logic (MCP or REST) at the
    ``# TODO`` marker below.  This keeps the utility runnable in both
    connected and disconnected contexts.
    """
    details: List[Dict[str, Any]] = []

    for scenario in HALLUCINATION_SCENARIOS:
        scenario_id = scenario["scenario_id"]
        sid = batch_session_id(scenario_id)
        expected = EXPECTED_OUTCOMES[scenario_id]

        # ------------------------------------------------------------------
        # TODO: replace this block with a live Okahu trace lookup.
        #
        # Option A — Okahu MCP tool (recommended from Claude Code / CI with
        #             MCP server configured):
        #
        #   result = mcp__okahu-prod-df__get_traces(
        #       workflow=workflow_filter,
        #       session_id=sid,
        #   )
        #   actual_label = result["traces"][0]["eval_result"]["label"]
        #
        # Option B — Okahu REST API (for standalone scripts / GitHub Actions):
        #
        #   import requests
        #   resp = requests.get(
        #       f"{OKAHU_BASE_URL}/traces",
        #       params={"workflow": workflow_filter, "session_id": sid},
        #       headers={"Authorization": f"Bearer {os.environ['OKAHU_API_KEY']}"},
        #   )
        #   actual_label = resp.json()["traces"][0]["eval_result"]["label"]
        #
        # Option C — Re-use the inline Monocle result collected during the
        #            pytest run by storing it in a module-level dict:
        #
        #   actual_label = _RUNTIME_RESULTS.get(scenario_id)
        #
        # ------------------------------------------------------------------
        actual_label: Optional[str] = None   # replace with lookup above
        # ------------------------------------------------------------------

        details.append({
            "scenario_id":  scenario_id,
            "session_id":   sid,
            "agent":        scenario["agent"],
            "expected":     expected,
            "actual":       actual_label,
            "match":        (actual_label == expected) if actual_label is not None else None,
        })

    passed  = sum(1 for d in details if d["match"] is True)
    failed  = sum(1 for d in details if d["match"] is False)
    pending = sum(1 for d in details if d["match"] is None)
    total   = len(details)

    report: Dict[str, Any] = {
        "total":     total,
        "passed":    passed,
        "failed":    failed,
        "pending":   pending,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "details":   details,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Hallucination Batch Eval — Comparison Report")
    print(f"{'='*60}")
    print(f"  Total scenarios : {total}")
    print(f"  Passed          : {passed}")
    print(f"  Failed          : {failed}")
    print(f"  Pending (no data): {pending}")
    if total:
        print(f"  Pass rate       : {report['pass_rate']:.1%}")
    print(f"{'='*60}")

    if failed:
        print("\nFailed scenarios:")
        for d in details:
            if d["match"] is False:
                print(f"  {d['scenario_id']:10s}  expected={d['expected']!r:25s}  actual={d['actual']!r}")

    if output_path:
        with open(output_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nReport saved to: {output_path}")

    return report


# ---------------------------------------------------------------------------
# Optional: collect inline eval results during the pytest run
# ---------------------------------------------------------------------------

# Uncomment and wire up if you want to collect actual labels during the run
# for use with compare_batch_results(Option C above):
#
# _RUNTIME_RESULTS: Dict[str, str] = {}
#
# In test_hallucination_batch, after check_eval:
#   _RUNTIME_RESULTS[scenario_id] = monocle_trace_asserter.last_eval_label
# (Requires TraceAssertion to expose last_eval_label — check monocle_test_tools API.)
