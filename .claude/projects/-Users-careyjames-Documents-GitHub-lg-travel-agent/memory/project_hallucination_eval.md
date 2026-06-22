---
name: Hallucination Eval Project Architecture
description: Three intentional-error demo agents for testing Okahu hallucination evaluation; Monocle tracing gap is the primary test failure cause
type: project
---

Three demo agents (hallucination_customer_care_agent.py, hallucination_financial_services_agent.py, hallucination_lg_travel_agent.py) deliberately produce hallucinations via ERROR-1 through ERROR-6 patterns for testing the Okahu eval platform.

**Why:** Validate that Okahu's evaluator correctly classifies agent outputs as major_hallucination, minor_hallucination, or no_hallucination.

**Primary test failure root cause (as of 2026-04-23):** Monocle tool invocation spans record empty `data.output` events. All `called_tool(...).contains_output(value)` assertion chains fail because the span is empty even when the tool returned valid data. Fix is in test code: move `contains_output(value)` to standalone `monocle_trace_asserter.contains_output(value)`.

**DO NOT change these intentional hallucination triggers:**
- `check_balance` suffix stripping (ACC-X-R → ACC-X) — underpins FS-T05/T06/T17 scope_drift tests
- `transfer_funds` returning {} or partial for >$5k — underpins FS-T01/T10/T16 fabrication_from_empty_tool tests
- `process_refund` returning partial for ORD-NS/amount>$200 — underpins CC-T01/T08/T15
- BRK.A→BRK.B substitution in execute_trade — underpins FS-T03/T19
- Paris, France hotel substitution in book_hotel — underpins LGS-T01/T11/T13
- Sparse outputs on get_stock_info, get_shipping_status, get_product_warranty — all intentional

**Okahu MCP:** Prod env confirmed working with key in .env (OKAHU_API_KEY). SRE agent does not support filtering by scope.git.run.id directly; use time ranges. Apps: customer_care_agent_bw0jll, financial_services_agent_wdkmcx, test_lg_travel_agent_ysbfrk.

**Analysis docs:** test_run_analysis_2026-04-23T14-13-22.md has authoritative per-test analysis for the 14:13 run. docs/2026-04-23-test-run-analysis.md covers the 15:31 run. OPERATION_FIXES_PROPOSAL.md has been revised with correct 3-track fix strategy.

**How to apply:** Fix order: P0 test assertions (~30 tests, no agent code changes) → P1 routing fixes (~9 tests) → P2 prompt calibration (~13 tests).
