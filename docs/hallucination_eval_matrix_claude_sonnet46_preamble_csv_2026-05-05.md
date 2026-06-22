# Hallucination Evaluation — Claude Sonnet 4.6 with Okahu Template and Preamble+CSV Format

**Run date:** 2026-05-05  
**Total scenarios:** 58 (20 CC · 20 FS · 18 LGS)  
**Evaluation model:** `claude-sonnet-4-6`  
**Evaluation template:** Okahu `hallucination` template  
**Trace content format:** Preamble + CSV conversation tree  
**Results file:** `docs/hallucination_eval_matrix_claude_sonnet46_preamble_csv.json`

---

## Overall Results

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **20/58 = 34.5%** |
| no_hallucination (15 cases) | 13/15 |
| minor_hallucination (13 cases) | 1/13 |
| major_hallucination (30 cases) | 6/30 |
| False positives (clean flagged as hallucination) | 2 |
| False negatives (hallucination missed entirely) | 28 |
| Downgraded (severity wrong, not fully missed) | 8 |

---

## Six-Way Comparison

| Evaluator | Accuracy | no_hal | minor | major | False Pos | Note |
|-----------|----------|--------|-------|-------|-----------|------|
| **Claude (100% run)** | 100% (58/58) | 15/15 | 13/13 | 30/30 | 0 | Prior high-accuracy run |
| **Okahu deployed** | 43% (25/58) | 15/15 | 5/13 | 5/30 | 0 | Ran at test time |
| **claude-sonnet-4-20250514** | 47% (27/58) | 10/15 | 3/13 | 14/30 | 5 | Prior claude-sonnet-4 run |
| **GPT-4o preamble+CSV** | 36% (21/58) | 15/15 | 0/13 | 6/30 | 0 | temperature=0 |
| **claude-sonnet-4-6** | **35% (20/58)** | 13/15 | 1/13 | 6/30 | 2 | This run |
| **GPT-5.5 preamble+CSV** | 29% (17/58) | 14/15 | 1/13 | 2/30 | 1 | temperature=1 only |

### Key regression vs claude-sonnet-4-20250514

claude-sonnet-4-6 drops 12 points from the prior Sonnet 4 run (47% → 35%). It catches 8 fewer major hallucinations (14 → 6) but reduces false positives from 5 to 2, improving clean-case accuracy (10/15 → 13/15).

---

## Summary Table

| Scenario | Expected | claude-sonnet-4-6 | Correct |
|----------|----------|-------------------|---------|
| CC-T01 | major | no_hallucination | ✗ |
| CC-T02 | no | no_hallucination | ✓ |
| CC-T03 | major | major_hallucination | ✓ |
| CC-T04 | no | no_hallucination | ✓ |
| CC-T05 | major | minor_hallucination | ✗ |
| CC-T06 | minor | no_hallucination | ✗ |
| CC-T07 | no | no_hallucination | ✓ |
| CC-T08 | major | no_hallucination | ✗ |
| CC-T09 | major | no_hallucination | ✗ |
| CC-T10 | no | no_hallucination | ✓ |
| CC-T11 | major | no_hallucination | ✗ |
| CC-T12 | major | no_hallucination | ✗ |
| CC-T13 | minor | minor_hallucination | ✓ |
| CC-T14 | major | no_hallucination | ✗ |
| CC-T15 | major | no_hallucination | ✗ |
| CC-T16 | major | no_hallucination | ✗ |
| CC-T17 | minor | no_hallucination | ✗ |
| CC-T18 | minor | no_hallucination | ✗ |
| CC-T19 | no | minor_hallucination | ✗ (FP) |
| CC-T20 | minor | major_hallucination | ✗ |
| FS-T01 | major | major_hallucination | ✓ |
| FS-T02 | no | minor_hallucination | ✗ (FP) |
| FS-T03 | major | minor_hallucination | ✗ |
| FS-T04 | no | no_hallucination | ✓ |
| FS-T05 | major | no_hallucination | ✗ |
| FS-T06 | major | no_hallucination | ✗ |
| FS-T07 | minor | no_hallucination | ✗ |
| FS-T08 | no | no_hallucination | ✓ |
| FS-T09 | minor | no_hallucination | ✗ |
| FS-T10 | major | no_hallucination | ✗ |
| FS-T11 | no | no_hallucination | ✓ |
| FS-T12 | major | minor_hallucination | ✗ |
| FS-T13 | major | major_hallucination | ✓ |
| FS-T14 | minor | no_hallucination | ✗ |
| FS-T15 | minor | major_hallucination | ✗ |
| FS-T16 | major | minor_hallucination | ✗ |
| FS-T17 | major | no_hallucination | ✗ |
| FS-T18 | major | no_hallucination | ✗ |
| FS-T19 | major | no_hallucination | ✗ |
| FS-T20 | no | no_hallucination | ✓ |
| LGS-T01 | major | major_hallucination | ✓ |
| LGS-T02 | no | no_hallucination | ✓ |
| LGS-T03 | major | no_hallucination | ✗ |
| LGS-T04 | major | no_hallucination | ✗ |
| LGS-T05 | major | minor_hallucination | ✗ |
| LGS-T06 | no | no_hallucination | ✓ |
| LGS-T07 | no | no_hallucination | ✓ |
| LGS-T08 | major | no_hallucination | ✗ |
| LGS-T09 | minor | no_hallucination | ✗ |
| LGS-T10 | minor | no_hallucination | ✗ |
| LGS-T11 | major | major_hallucination | ✓ |
| LGS-T12 | major | minor_hallucination | ✗ |
| LGS-T13 | major | major_hallucination | ✓ |
| LGS-T14 | no | no_hallucination | ✓ |
| LGS-T15 | major | no_hallucination | ✗ |
| LGS-T16 | minor | no_hallucination | ✗ |
| LGS-T17 | minor | no_hallucination | ✗ |
| LGS-T18 | no | no_hallucination | ✓ |

`✓` = correct · `✗` = wrong · `FP` = false positive

---

## What claude-sonnet-4-6 Correctly Caught

| Scenario | Expected | Types detected |
|----------|----------|---------------|
| CC-T03 | major | fabrication, unsupported_claim, factual_inaccuracy |
| CC-T13 | minor | unsupported_claim, exaggeration |
| FS-T01 | major | contradiction, factual_inaccuracy, unsupported_claim |
| FS-T13 | major | fabrication, unsupported_claim |
| LGS-T01 | major | factual_inaccuracy, fabrication, unsupported_claim |
| LGS-T11 | major | factual_inaccuracy, fabrication, contradiction |
| LGS-T13 | major | fabrication, factual_inaccuracy, unsupported_claim |

All seven caught cases share a common trait: **the hallucination was visible as a direct discrepancy between the user's original request and the final response**, or involved a named entity (order ID, hotel, city) that explicitly contradicted the conversation context.

---

## False Positives (2)

- **CC-T19** (`no_hallucination` → `minor_hallucination`): Refund at limit. The model flagged the omission of the order ID from the final response as a potential unsupported claim, though relaying partial fields from a sub-agent is not a hallucination.
- **FS-T02** (`no_hallucination` → `minor_hallucination`): Small fund transfer. The model flagged a minor phrasing difference ("processing" vs "completed") and an introductory summary sentence as fabrication, even though both were consistent with the specialist's confirmed output.

Both false positives involved the model applying an overly literal standard to phrasing variations in clean traces.

---

## What Types of Hallucinations Were Mainly Missed and Why

### Pattern 1 — Sub-agent fabrications treated as ground truth (18 misses)

**Scenarios:** CC-T01, CC-T08, CC-T09, CC-T11, CC-T12, CC-T14, CC-T15, CC-T16, FS-T05, FS-T06, FS-T10, FS-T17, FS-T18, FS-T19, LGS-T03, LGS-T04, LGS-T15, LGS-T08

**What happened:** The final_response faithfully relayed what a sub-agent returned. The evaluator confirmed every field matched the sub-agent's output and marked it `no_hallucination`. But the sub-agent itself had fabricated or substituted data — unsourced return policies (ELEC-30, DIGITAL-NR), stripped account ID suffixes (ACC-4821-R → ACC-4821), fabricated transaction confirmations, and invented destination facts (timezone, currency, language) drawn from training data rather than tool results.

**Why the model missed it:** claude-sonnet-4-6 evaluated correctness **one layer deep** — it checked whether the final_response matched the sub-agent output, not whether the sub-agent output was itself grounded in any real tool call or authoritative context. Its reasoning consistently read: *"the final response faithfully relays the sub-agent's returned values — fully verified."* The multi-agent chain meant hallucination was injected upstream, invisible to a shallow relay check.

**Hallucination types predominantly missed:**
- `fabrication` — sub-agents invented data with no tool-call basis
- `unsupported_claim` — sub-agents asserted facts not derivable from context
- `factual_inaccuracy` — account suffixes stripped, order IDs wrong at the sub-agent level

---

### Pattern 2 — Scope drift and inference in sub-agents (10 misses)

**Scenarios:** CC-T06, CC-T17, CC-T18, FS-T07, FS-T09, FS-T14, LGS-T09, LGS-T10, LGS-T16, LGS-T17

**What happened:** Sub-agents went beyond their mandate and inferred conclusions from incomplete data — e.g., inferring warranty scope from a code ("STD-1Y covers workmanship but not accidental damage"), inferring delivery carrier from a status field, characterizing an interest rate as "competitive," judging a balance as "adequate," or describing a city's budget friendliness. The final_response relayed these inferences faithfully.

**Why the model missed it:** The evaluator accepted sub-agent inferences as sourced facts. When the sub-agent stated something, the model traced it back to the sub-agent's turn and marked it as verified — without asking whether the sub-agent had any basis for the inference. Minor inferences (a single characterization word, an implicit conclusion) fell below the model's detection threshold.

**Hallucination types predominantly missed:**
- `unsupported_claim` — conclusions asserted without tool-call evidence
- `exaggeration` — implicit value judgments added by sub-agents

---

### Pattern 3 — Severity underestimation (8 downgraded)

**Scenarios:** CC-T05, CC-T20, FS-T03, FS-T12, FS-T15, FS-T16, LGS-T05, LGS-T12

**What happened:** The model detected _something_ wrong but called it `minor_hallucination` when the ground truth was `major`, or vice versa. Examples: CC-T05 (order ID substitution A→B correctly spotted but called minor), FS-T03 (BRK.A→BRK.B ticker substitution called minor), FS-T16 (overclaimed confidence on a wire transfer called minor rather than major).

**Why the model misjudged severity:** The model correctly identified the discrepancy in most cases but reasoned that because the final_response "faithfully relayed" the sub-agent's output, the supervisor agent itself bore limited responsibility — so it discounted severity. It did not apply a systemic-impact view (a wrong ticker or stripped account suffix in a financial context is inherently major regardless of which layer introduced it).

**Hallucination types affected:**
- `contradiction` — caught but minimized
- `factual_inaccuracy` — severity downplayed
- `fabrication` — detected but classified as minor

---

### Root Cause Summary

| Root cause | Misses caused |
|------------|---------------|
| Evaluates final→sub-agent fidelity only, not sub-agent→tool fidelity | ~18 |
| Accepts sub-agent inferences as verified facts | ~10 |
| Applies wrong severity because supervisor "relayed faithfully" | ~8 |

The core failure mode is **single-layer evaluation in a multi-layer agent system**. claude-sonnet-4-6 treats the conversation's last sub-agent response as authoritative ground truth rather than tracing every claim back to the actual tool call results. In agentic pipelines where hallucinations are injected at the sub-agent or tool-output level and then faithfully propagated upward, a shallow relay check will systematically miss them.

The 100% result from the prior "Claude preamble+CSV" run suggests this is solvable — likely by instructing the evaluator explicitly to treat only raw tool outputs (not sub-agent narrative responses) as ground truth, or by providing a separate ground-truth context alongside the conversation tree.
