"""
Text-to-SQL Analyst — Okahu Pytest Fluent Tests
================================================
Fluent rewrite of test_analyst.py using monocle_test_tools TraceAssertion.

The agent MUST use Okahu MCP traces to debug failures.
NO local logs are available — traces only exist in Okahu Cloud.

Tests:
  test_generate_sql_show_all_users      — basic SQL generation (similarity check)
  test_generate_sql_high_value_users    — complex JOIN query (similarity check)
  test_execute_query_users_table        — direct DB execution (no monocle)
  test_execute_query_orders_table       — direct DB execution (no monocle)
  test_text_to_sql_e2e                  — full pipeline (fluent + exact result check)
"""

import logging
import pytest
from typing import Optional, Union
from monocle_test_tools import TraceAssertion
from monocle_test_tools.evals.okahu_eval import OkahuEval

from analyst import generate_sql, execute_query, text_to_sql

# Suppress ALL local output — force MCP-only debugging
logging.disable(logging.CRITICAL)


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

    def with_evaluation(self, eval: Union[str, OkahuEval], eval_options=None) -> "AgentTypeTraceAssertion":
        return super().with_evaluation(eval, eval_options)

    async def run_agent_async(self, agent, *args, **kwargs):
        if self._default_agent_type:
            return await self.validator.run_agent_async(
                agent, self._default_agent_type, *args, **kwargs
            )
        return await super().run_agent_async(agent, *args, **kwargs)


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
    monocle_trace_asserter.with_agent_type("openai")


# ===========================================================================
# SQL Generation Tests (monocle-traced)
# ===========================================================================

@pytest.mark.asyncio
async def test_generate_sql_show_all_users(monocle_trace_asserter: AgentTypeTraceAssertion):
    """Basic SQL generation — similarity check against SELECT * FROM users.

    Fails on Bug #1 (invalid model) and Bug #2 (.text attribute).
    Monocle validates: inference span exists, SQL output matches expected pattern.
    """
    await monocle_trace_asserter.run_agent_async(
        generate_sql, "Show all users"
    )
    monocle_trace_asserter.contains_output("SELECT") \
        .contains_output("users")
    monocle_trace_asserter.with_evaluation("similarity").check_eval(
        "response", "SELECT * FROM users"
    )


@pytest.mark.asyncio
async def test_generate_sql_high_value_users(monocle_trace_asserter: AgentTypeTraceAssertion):
    """Complex JOIN query — similarity check against expected SQL.

    Fails on Bug #1 (invalid model), Bug #2 (.text), and Bug #3 (wrong schema/tables).
    Monocle validates: inference span exists, JOIN + WHERE clause present.
    """
    await monocle_trace_asserter.run_agent_async(
        generate_sql,
        "Find all users who have made orders with amount greater than 100",
    )
    monocle_trace_asserter.contains_output("SELECT") \
        .contains_output("JOIN") \
        .contains_output("orders") \
        .contains_output("100")
    monocle_trace_asserter.with_evaluation("similarity").check_eval(
        "response",
        "SELECT users.* FROM users JOIN orders ON users.user_id = orders.user_id WHERE orders.amount > 100",
    )


# ===========================================================================
# Direct Database Tests (no monocle — these test the DB layer itself)
# ===========================================================================

def test_execute_query_users_table():
    """Direct SQL execution on users table — verifies table exists and has data."""
    results = execute_query("SELECT * FROM users LIMIT 3")
    assert len(results) == 3, "Should return 3 users"
    assert len(results[0]) == 3, "Each user row should have 3 columns (user_id, username, email)"


def test_execute_query_orders_table():
    """Direct SQL execution on orders table — verifies table exists and has data."""
    results = execute_query("SELECT * FROM orders WHERE amount > 100 LIMIT 5")
    assert len(results) > 0, "Should find orders over $100"
    assert len(results[0]) == 4, "Each order row should have 4 columns (order_id, user_id, amount, order_date)"


# ===========================================================================
# End-to-End Test (fluent trace assertions + exact result check)
# ===========================================================================

EXPECTED_HIGH_VALUE_USER_RESULTS = [
    (1, "Alice", "alice@example.com"),
    (2, "Bob", "bob@example.com"),
    (3, "Charlie", "charlie@example.com"),
    (3, "Charlie", "charlie@example.com"),
    (4, "David", "david@example.com"),
    (1, "Alice", "alice@example.com"),
    (4, "David", "david@example.com"),
]


@pytest.mark.asyncio
async def test_text_to_sql_e2e(monocle_trace_asserter: AgentTypeTraceAssertion):
    """End-to-end: natural language → SQL → executed results.

    Validates the full pipeline:
    - Inference span generated (OpenAI called correctly)
    - SQL contains expected clauses (fluent span assertions)
    - Executed results match exactly (direct equality check)
    """
    result = await monocle_trace_asserter.run_agent_async(
        text_to_sql,
        "Find all users who have made orders with amount greater than 100",
    )
    monocle_trace_asserter.contains_output("SELECT") \
        .contains_output("users") \
        .contains_output("orders")
    assert result == EXPECTED_HIGH_VALUE_USER_RESULTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
