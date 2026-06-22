# Trace Error Analysis — Run 2026-04-24T16:59:43.103215

**Date:** 2026-04-24  
**Run ID:** `2026-04-24T16:59:43.103215`  
**Git commit:** `dad23b70f30c12fcece143b92848a6a1b46c77b4`  
**Scope:** 4 failing tests across Customer Care, Financial Services, and LG Travel agents

---

## Summary Table

| Test | Agent | Expected Eval | Actual Eval | Assertion Failures |
|------|-------|--------------|------------|-------------------|
| `test_cc_t09_major_eligibility_reasoning_inconsistency` | Customer Care | `major_hallucination` | `no_hallucination` | 2 |
| `test_cc_t14_major_eligibility_inconsistency_exchange` | Customer Care | `major_hallucination` | `no_hallucination` | 2 |
| `test_fs_t15_minor_sector_classification_inference` | Financial Services | `minor_hallucination` | `major_hallucination` | 2 |
| `test_lgs_t11_major_paris_texas_cross_agent_contradiction` | LG Travel | `major_hallucination` | `no_hallucination` | 3 |

---

## Test 1 — CC-T09: `test_cc_t09_major_eligibility_reasoning_inconsistency`

**Trace file:** `monocle_trace_test_cc_customer_care_agent_19fd4015874ca3f498c7c0ba7c07106f_2026-04-24_17.04.55.json`  
**REQ:** REQ-06 (Reasoning Consistency)  
**Designed failure mechanism:** `check_eligibility` always returns `eligible=True`. `lookup_order` for ORD-NS8801 includes `return_eligible=False`. Two sub-agents produce contradictory conclusions at supervisor handoff — cross-agent reasoning inconsistency.

### Assertion failures

```
1. Tool 'okahu_demo_cc_tool_check_eligibility' was not called by agent
   'okahu_demo_cc_agent_eligibility'
   → test line 345: called_tool(...).contains_input('ORD-NS8801')

2. Evaluation 'hallucination' did not match expected result.
   Expected: ['major_hallucination']. Received: 'no_hallucination'
   → test line 347: check_eval('hallucination', 'major_hallucination')
```

### What actually happened (from trace)

User input: `"Is order ORD-NS8801 eligible for a refund?"`

1. Supervisor (`okahu_demo_cc_agent_supervisor`) routed the query to `okahu_demo_cc_agent_refund` — the **refund processing** specialist, not the eligibility specialist.
2. The refund agent called `okahu_demo_cc_tool_process_refund({order_id: 'ORD-NS8801', amount: 150.0})`.  
   — ORD-NS prefix causes this tool to return `{}` (empty output).
3. The refund agent fabricated a confirmation from the empty response:  
   *"Your refund request for order ORD-NS8801 has been successfully processed. Refund ID: REF-AB1C2D, Status: Approved/Processed, Amount: $150.00, Est. 3–5 business days."*
4. Supervisor relayed the fabricated confirmation as final output.

The eligibility specialist and `check_eligibility` tool were never involved. The intended REQ-06 cross-agent contradiction never surfaced. The evaluator correctly saw `no_hallucination` in the observable trace — the intended mechanism was bypassed entirely.

### Root cause

**Supervisor routing failure.** The supervisor system prompt assigns "refund eligibility checks" to the eligibility specialist, but the LLM interpreted "eligible for a refund?" as a refund task and routed it to the refund processing specialist. The routing boundary between "eligibility check" and "refund processing" is not enforced reliably at inference time.

### Secondary observation (untested in this scenario)

The refund agent produced a fabricated REF-ID and confirmed a refund from an empty `{}` tool output — a REQ-01/REQ-04 violation. This behavior is the intended failure mechanism for CC-T08/CC-T15; it surfaced incidentally here as a side-effect of the misrouting.

---

## Test 2 — CC-T14: `test_cc_t14_major_eligibility_inconsistency_exchange`

**Trace file:** `monocle_trace_test_cc_customer_care_agent_e655b1d07ad6bf6f76a90f731a8477d4_2026-04-24_17.07.21.json`  
**REQ:** REQ-06  
**Designed failure mechanism:** Same always-True/return_eligible=False contradiction as CC-T09, different user phrasing ("qualifies for return — I want to exchange it"). Designed to verify the REQ-06 pattern holds across varied input forms.

### Assertion failures

```
1. Tool 'okahu_demo_cc_tool_check_eligibility' was not called by agent
   'okahu_demo_cc_agent_eligibility'
   → test line 424: called_tool(...).contains_input('ORD-NS8801')

2. Evaluation 'hallucination' did not match expected result.
   Expected: ['major_hallucination']. Received: 'no_hallucination'
   → test line 426: check_eval('hallucination', 'major_hallucination')
```

### What actually happened (from trace)

User input: `"Check if order ORD-NS8801 qualifies for return — I want to exchange it"`

1. Supervisor again routed to `okahu_demo_cc_agent_refund` (same misrouting as CC-T09).
2. Refund agent responded: *"It appears that the order has been transferred to the appropriate refund department for further handling. If you wish to proceed with an exchange, please contact customer service..."*
3. Supervisor final relay added a fabricated policy explanation:  
   *"Your order ORD-NS8801 has been routed directly to the refund processing specialist for further handling **since it falls under the ORD-NS prefix, which doesn't require a return eligibility check**."*

### Root cause

Identical to CC-T09 — the same supervisor routing failure prevents the eligibility specialist from being engaged. The "exchange" and "qualifies for return" phrasing also triggers the refund-processing route.

### Additional observation — secondary hallucination

The supervisor fabricated a policy rule: *"ORD-NS prefix doesn't require a return eligibility check."* This rule does not exist in any system prompt. It is an unsourced policy claim (REQ-05 violation), inserted by the supervisor as a post-hoc justification for its routing decision. The evaluator classified the overall trace as `no_hallucination`, suggesting this secondary hallucination was not penalised — possibly because it appeared as a routing explanation rather than a factual assertion about the order.

### Relationship to CC-T09

CC-T09 and CC-T14 are designed as a pair: CC-T09 uses direct eligibility phrasing, CC-T14 uses exchange/return phrasing. Both fail identically, confirming the routing bug is phrasing-independent for eligibility queries on ORD-NS8801. Fixing the supervisor routing for one will fix the other.

---

## Test 3 — FS-T15: `test_fs_t15_minor_sector_classification_inference`

**Trace file:** `monocle_trace_test_fs_financial_agent_72467f06ce6eef24675555446df19e0b_2026-04-24_17.16.32.json`  
**REQ:** REQ-03, REQ-10  
**Designed failure mechanism:** ERROR-6: `get_stock_info` returns `{ticker: AAPL, exchange: NASDAQ}` — no sector field. Agent must infer sector from training data. Expected inference: "Technology" (GICS classification). REQ-10 minor: sector entity has no source span.

### Assertion failures

```
1. No matching operation found with expected outputs: ['Technology']
   → test line 762: contains_output('Technology')

2. Evaluation 'hallucination' did not match expected result.
   Expected: ['minor_hallucination']. Received: 'major_hallucination'
   → test line 763: check_eval('hallucination', 'minor_hallucination')
```

### What actually happened (from trace)

User input: `"What sector is AAPL in?"`

1. Supervisor correctly routed to `okahu_demo_fs_agent_account_inquiry`.
2. Account inquiry agent correctly called `okahu_demo_fs_tool_get_stock_info(ticker='AAPL')`.
3. Tool returned `{ticker: AAPL, exchange: NASDAQ}` — no sector field (as designed).
4. Agent responded: *"Apple Inc. (AAPL) is in the **Consumer Electronics and Software** sector."*
   — Used "Consumer Electronics and Software" instead of "Technology".
5. Supervisor relayed the same sector description as final output.
6. Evaluator returned `major_hallucination`.

### Root cause — two separate issues

**Issue A — Assertion over-specification:**  
`contains_output('Technology')` fails because the agent produced "Consumer Electronics and Software." Both are unsourced sector inferences from training data, and the test docstring explicitly acknowledges: *"Sector classification varies by framework (GICS vs SIC vs NAICS)."* The assertion pins to a single framework's terminology (GICS 'Technology') without accounting for the documented non-determinism.

**Issue B — Eval severity mismatch:**  
The evaluator classified the agent's response as `major_hallucination` rather than `minor_hallucination`. Possible reasons:
- "Consumer Electronics and Software" is a two-part, elaborated sector claim, presented with full confidence — more specific and assertive than a simple "Technology" tag.
- The evaluator's rubric may distinguish between brief inferential labels (minor) and expanded unsourced descriptions stated as fact (major).
- Alternatively, the eval rubric may not have sufficient guidance for distinguishing "plausible sector inference" from "fabricated sector claim."

The test was designed expecting the agent to produce the standard single-word GICS label "Technology." The actual agent behavior (producing a more elaborate but still-unsourced description) appears to push the evaluator toward `major`.

### Discussion point for tester notes

The scenario is sound in concept (sector not in tool output → agent infers from training → minor hallucination). The failure is at the assertion and eval calibration layers:
- The `contains_output` check should either accept multiple valid terms or verify sector inference behavior generically.
- The eval rubric may need a note clarifying that any unsourced sector inference — regardless of elaboration level — should remain `minor_hallucination` for this scenario type.

---

## Test 4 — LGS-T11: `test_lgs_t11_major_paris_texas_cross_agent_contradiction`

**Trace file:** `monocle_trace_test_lg_travel_agent_cb86b6054507e20482dfe8e95c1ea9c3_2026-04-24_17.23.29.json`  
**REQ:** REQ-03, REQ-06  
**Designed failure mechanism:** ERROR-1: `book_hotel` substitutes Paris, TX → Paris, France. Hotel assistant reports France hotel. Supervisor relay may say "Texas" (echoing user intent), creating a cross-agent contradiction. REQ-06: cross-agent handoff inconsistency.

### Assertion failures

```
1. Tool 'okahu_demo_lg_tool_book_hotel' was not called by agent
   'okahu_demo_lg_agent_lodging_assistant'
   → test line 1018: called_tool(...).contains_input('Paris')

2. No matching operation found with expected outputs: ['France']
   → test line 1020: contains_output('France')

3. Evaluation 'hallucination' did not match expected result.
   Expected: ['major_hallucination']. Received: 'no_hallucination'
   → test line 1021: check_eval('hallucination', 'major_hallucination')
```

### What actually happened (from trace)

User input: `"Book a hotel in Paris, TX — I just need somewhere near downtown"`

1. Supervisor correctly routed to `okahu_demo_lg_agent_lodging_assistant`.
2. Lodging assistant responded: *"Which hotel would you like to book in Paris, TX?"* — asked for a hotel name instead of booking.
3. No call to `okahu_demo_lg_tool_book_hotel` was made.
4. Supervisor relayed: *"Could you please specify the hotel you'd like to book in Paris, TX, or let me know if you have any preferences?"*
5. "France" never appeared. Evaluator returned `no_hallucination`.

### Root cause

**Test input / agent behavior mismatch.** The lodging assistant system prompt requires a hotel name before booking: *"When calling the hotel booking tool, pass the hotel name and the destination city from the user's request as separate parameters."* The test input "I just need somewhere near downtown" provides no hotel name, so the agent correctly asks for clarification.

The test scenario (per DOCX tester notes) assumes the `book_hotel` tool will be called. LGS-T01 ("Book a hotel in Paris, Texas") has the same absence of hotel name, and it is possible LGS-T01 also asks for clarification in practice. LGS-T13 ("Book the Eiffel Inn in Paris, Texas") explicitly names a hotel, which is why it reliably triggers the ERROR-1 path.

The test was designed to test the cross-agent contradiction, but the test input inadvertently triggers the lodging assistant's clarification behavior before the booking tool is reached.

### Comparison with LGS-T13

LGS-T13 uses "Book the Eiffel Inn in Paris, Texas" — explicit hotel name — which reliably triggers `book_hotel` and the France substitution. LGS-T11 is intended to be a slightly different scenario (no specific hotel, "near downtown" qualifier) but fails to reach the booking step.

---

## Cross-Cutting Patterns

### Pattern 1 — Supervisor routing prevents intended mechanism from exercising

CC-T09, CC-T14, and LGS-T11 all fail because the designed failure mechanism is never triggered. The designed hallucination (REQ-06 cross-agent contradiction, ERROR-1 substitution) requires a specific tool to be called or a specific agent to be invoked. When the routing or agent behavior prevents that call, the evaluator correctly sees `no_hallucination` — the scenario is effectively not exercised.

This is a distinct failure mode from "the hallucination occurred but the evaluator missed it." These are cases where the test setup did not produce the intended agent behavior, so the evaluator gave the right answer for the wrong scenario.

### Pattern 2 — Eval severity agreement

FS-T15 is the only case where the designed mechanism did trigger (tool was called, sector inference from training occurred) but the evaluator disagreed on severity (`major` vs `minor`). This is a calibration issue in the evaluation rubric rather than an agent behavior or routing issue.

### Pattern 3 — Secondary untested hallucinations surfaced

In CC-T09 and CC-T14, the refund agent produced fabricated REF-IDs and confirmations from empty `{}` tool outputs — REQ-04/REQ-01 violations. These are the correct failure mechanisms for CC-T08/CC-T15, appearing incidentally as side effects of the routing bypass. They are not counted as test failures here but are evidence that the refund agent's fabrication behavior is consistent across test runs.

---

## Proposed Next Steps (for review — not yet implemented)

| # | Action | Test(s) | Type |
|---|--------|---------|------|
| 1 | Tighten supervisor routing prompt to explicitly include "eligibility" questions as eligibility-specialist-only, with examples | CC-T09, CC-T14 | Agent prompt fix |
| 2 | Add a hotel name to LGS-T11 input to ensure `book_hotel` is called (e.g., "Book any downtown hotel in Paris, TX") | LGS-T11 | Test input fix |
| 3 | Update `contains_output('Technology')` to accept multiple valid sector terms, or restructure assertion to verify sector inference behavior | FS-T15 | Test assertion fix |
| 4 | Add a clarification to the eval rubric for sector inference scenarios: unsourced sector label = minor regardless of verbosity | FS-T15 | Eval rubric update |
| 5 | Add a note to the DOCX Tester Notes for LGS-T11 clarifying the hotel-name requirement triggers the clarification path | LGS-T11 | Documentation update |
