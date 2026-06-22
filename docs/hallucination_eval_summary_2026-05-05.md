# Hallucination Evaluation Summary — All Runs After May 4, 2026

**Date:** 2026-05-05  
**Scope:** All `hallucination_eval_matrix` markdown files in `docs/` created after May 4, 2026  
**Test corpus:** 58 scenarios across three agent types — Customer Care (CC, 20), Financial Services (FS, 20), LGS Travel (LGS, 18)  
**Ground truth:** 15 `no_hallucination` · 13 `minor_hallucination` · 30 `major_hallucination`

---

## Master Comparison Table

| # | Judge Model | Conversation Format | Template | Accuracy | no_hal (15) | minor (13) | major (30) | False Pos | Source File |
|---|-------------|---------------------|----------|----------|-------------|------------|------------|-----------|-------------|
| 1 | Okahu deployed (prod) | Preamble+CSV (flat)| `hallucination` | 43% (25/58) | 15/15 | 5/13 | 5/30 | 0 | `hallucination_eval_matrix_full_2026-04-27T20-22-37.md` |
| 2 | claude-sonnet-4-6 (independent criteria) | Extracted trace outputs (JSON) | Independent (6 criteria) | 100% (58/58) | 15/15 | 13/13 | 30/30 | 0 | `hallucination_eval_matrix_full_2026-04-27T20-22-37.md` |
| 3 | claude-sonnet-4-6 (Okahu template) | Extracted trace outputs (JSON) | `hallucination` | 100% (58/58) | 15/15 | 13/13 | 30/30 | 0 | `hallucination_eval_matrix_full_2026-04-27T20-22-37.md` |
| 4 | claude-sonnet-4-20250514 | Preamble+CSV (flat) | `hallucination` | 47% (27/58) | 10/15 | 3/13 | 14/30 | 5 | `hallucination_eval_matrix_claude_sonnet4_preamble_csv.md` |
| 5 | GPT-4o (temp=0) | Preamble+CSV (flat) | `hallucination` | 36% (21/58) | 15/15 | 0/13 | 6/30 | 0 | `hallucination_eval_matrix_gpt4o_preamble_csv_2026-04-27T20-22-37.md` |
| 6 | claude-sonnet-4-6 | Preamble+CSV (flat) | `hallucination` | 35% (20/58) | 13/15 | 1/13 | 6/30 | 2 | `hallucination_eval_matrix_claude_sonnet46_preamble_csv_2026-05-05.md` |
| 7 | GPT-5.5 (temp=1) | Preamble+CSV (flat) | `hallucination` | 29% (17/58) | 14/15 | 1/13 | 2/30 | 1 | `hallucination_eval_matrix_gpt55_preamble_csv_2026-04-27T20-22-37.md` |

### Key signal:

All runs using **extracted tool outputs** (rows 2–3) scored 100%. All runs using **preamble+CSV flat format** (rows 1, 4–7) scored 29–47%. 

---

## Per-File Summaries

### File 1: `hallucination_eval_matrix_full_2026-04-27T20-22-37.md`

**Two evaluators compared in this file.**

#### Evaluator A — Okahu deployed (prod)
- **Judge:** Okahu prod service  
- **Conversation format:** Preamble+CSV (flat conversation tree, `id, parent_id, content`)  
- **Template:** `hallucination` v1  
- **Result:** 43% (25/58) — no_hal 15/15, minor 5/13, major 5/30, 0 false positives

#### Evaluator B — claude-sonnet-4-6 (Independent criteria)
- **Judge:** claude-sonnet-4-6  
- **Conversation format:** Extracted tool outputs (JSON)  
- **Template:** 6 independent evaluation criteria (not Okahu template)  
- **Result:** 100% (58/58)

#### Evaluator B — claude-sonnet-4-6 (Okahu template)
- **Judge:** claude-sonnet-4-6  
- **Conversation format:** Extracted tool outputs (JSON)  
- **Template:** `hallucination` v1 (same template as Okahu prod in File 1)  
- **Result:** 100% (58/58)

**Why Evaluator A underperformed:** The v1 template's severity tiers were not sufficiently differentiated. Minor vs major decisions collapsed toward `minor_hallucination` because the template did not define clear impact criteria (financial harm, identity substitution, fabrication vs inference).

**Notable finding:** The same `hallucination` v1 template that produced 43% accuracy when run by Okahu prod produced 100% when run by claude-sonnet-4-6 on the extracted-output conversation content. This indicates the template itself was not the primary bottleneck in the Okahu prod run — it is the conversation content passed by the Okahu prod pipeline differed from the extracted-output format used by claude.

**Why both achieved 100%:** The extracted tool outputs format made hallucination visible as a direct discrepancy between the `tool JSON` block and the `agent response` block. The judge could compare a `{"order_id": "ORD-B1042"}` tool return against an agent claim of `ORD-A1042` by inspection. No causal inference required.

---

### File 2: `hallucination_eval_matrix_claude_sonnet4_preamble_csv.md`

- **Judge:** claude-sonnet-4-20250514 (prior Sonnet 4 version)  
- **Conversation format:** Preamble+CSV (flat)  
- **Template:** `hallucination` v1  
- **Result:** 47% (27/58) — no_hal 10/15, minor 3/13, major 14/30, **5 false positives**

**Why accuracy was limited:**
- 5 false positives (clean cases flagged as hallucinations) — the model over-applied the hallucination standard, flagging phrasing variations and partial field relaying as fabrication
- 16 false negatives on major hallucinations — multi-layer sub-agent fabrications not traced to tool outputs
- 3/13 minor hallucinations caught — scope drift and inference missed at near-same rate as other preamble+CSV models

**Comparison vs claude-sonnet-4-6 on same format (row 8):** claude-sonnet-4-20250514 catches more major hallucinations (14 vs 6) but produces more false positives (5 vs 2). The older model applies a stricter but less precise standard; the newer model is more conservative but misses more fabrications.

---

### File 3: `hallucination_eval_matrix_gpt4o_preamble_csv_2026-04-27T20-22-37.md`

- **Judge:** GPT-4o (temperature=0)  
- **Conversation format:** Preamble+CSV (flat)  
- **Template:** `hallucination` v1  
- **Result:** 36% (21/58) — no_hal 15/15, minor 0/13, major 6/30, 0 false positives

**Why accuracy was limited:**
- **0/13 minor hallucinations** — GPT-4o at temperature=0 applied a high-confidence threshold; minor hallucinations (scope drift, single-word inferences, implicit characterizations) fell below its detection threshold entirely
- 6/30 major hallucinations — same single-layer evaluation failure as other preamble+CSV runs
- Perfect no_hallucination accuracy (15/15, 0 false positives) — highly conservative, essentially biased toward `no_hallucination` on ambiguous cases

**Behavioral profile:** GPT-4o temp=0 behaves as a high-precision, low-recall detector when given a flat conversation log. It correctly identifies clean cases and catches only the most egregious fabrications, but systematically misses all minor hallucinations and sub-agent-layer fabrications.

---

### File 4: `hallucination_eval_matrix_claude_sonnet46_preamble_csv_2026-05-05.md`

- **Judge:** claude-sonnet-4-6  
- **Conversation format:** Preamble+CSV (flat)  
- **Template:** `hallucination` v1  
- **Result:** 35% (20/58) — no_hal 13/15, minor 1/13, major 6/30, **2 false positives**

**Why accuracy was 35%:** Three failure patterns, documented in detail:

1. **Sub-agent fabrications treated as ground truth (18 misses)** — the judge evaluated whether the final response matched the sub-agent output (relay fidelity) but did not check whether the sub-agent output was itself grounded in a tool call. CC-T01, CC-T08, CC-T09, CC-T11, CC-T12, CC-T14–16, FS-T05, FS-T06, FS-T10, FS-T17–19, LGS-T03, LGS-T04, LGS-T08, LGS-T15 all fell here. The flat preamble+CSV log does not structurally distinguish tool outputs from agent narrations — they appear as consecutive rows in an undifferentiated tree.

2. **Scope drift and inference accepted as verified facts (10 misses)** — CC-T06, CC-T17, CC-T18, FS-T07, FS-T09, FS-T14, LGS-T09, LGS-T10, LGS-T16, LGS-T17. Sub-agents characterized interest rates as "competitive," described cities as "budget-friendly," inferred warranty scope from a code, inferred delivery carrier from a status field. The judge traced each inference back to the sub-agent's turn and marked it verified, without checking whether the sub-agent had a tool-call basis for the claim.

3. **Severity underestimation (8 downgraded)** — CC-T05, CC-T20, FS-T03, FS-T12, FS-T15, FS-T16, LGS-T05, LGS-T12. The judge detected a discrepancy but classified it as minor because the supervisor "faithfully relayed" the sub-agent. BRK.A→BRK.B ticker substitution, account suffix stripping, and fabricated TXN IDs were all downgraded on this basis.

---

### File 5: `hallucination_eval_matrix_gpt55_preamble_csv_2026-04-27T20-22-37.md`

- **Judge:** GPT-5.5 (temperature=1, forced by Responses API limitation)  
- **Conversation format:** Preamble+CSV (flat)  
- **Template:** `hallucination` v1  
- **Result:** 29% (17/58) — no_hal 14/15, minor 1/13, major 2/30, 1 false positive

**Why accuracy was worst among all preamble+CSV runs:**
- Temperature=1 is non-deterministic; GPT-5.5 was the only model tested without a temperature=0 option due to Responses API constraints at the time of the run
- 2/30 major hallucinations — the lowest major-hallucination recall of any run, including Okahu prod
- 1/13 minor hallucinations — near-zero recall on minor cases
- The model appears to exhibit the same single-layer relay-fidelity evaluation pattern but with higher output variance, leading to even more false negatives

**Note on temperature:** Runs 6, 7, 8 used temperature=0 (where available). GPT-5.5 at temperature=1 cannot be directly compared on accuracy grounds — the non-determinism makes any single-run result less reliable. A temperature=0 GPT-5.5 run was not available at time of writing.

---

## Cross-Cutting Analysis: Why Evals Did Not Match Expectations

### Finding 1 — Conversation format is the dominant variable

The single largest accuracy predictor is whether the conversation content passed to the judge explicitly presents the **reality boundary**: the raw tool output as a separate, labeled, authoritative datum.

| Conversation format | Accuracy range | Runs |
|---------------------|----------------|------|
| Extracted tool outputs (JSON) | 43–100% | Rows 1–5 |
| Preamble+CSV (flat) | 29–47% | Rows 6–9 |

The preamble+CSV format records conversation history as a tree of content rows. Tool outputs, agent narratives, and system prompts appear in the same format with no structural distinction. A judge reading a flat log cannot determine which rows represent external system returns versus agent elaborations without reasoning about row sequence — and all evaluated models failed to perform that reasoning reliably at scale.

The extracted tool outputs format makes the tool return a discrete, labeled block. This changes the evaluation task from "reconstruct the causality from a flat history" to "compare two labeled things." The latter is the task these models perform well.

### Finding 2 — Multi-layer agent systems defeat single-layer relay checks

The specific failure mode in preamble+CSV runs: every model evaluated the final response against the sub-agent output and confirmed they matched. In 18 cases, the sub-agent itself had fabricated or substituted data that no tool call ever returned — and the final response faithfully propagated the fabrication. The judge's relay-fidelity check returned `no_hallucination` in all 18 cases because it confirmed what it was designed to confirm: that the final response matched the preceding agent output.

This is not a model failure. It is a task formulation failure. The flat log does not surface the tool call/tool return boundary as a labeled evaluation target. Without that boundary, any model instructed to "check grounding" will default to checking the most proximate prior context — the sub-agent output — rather than tracing all the way back to tool results.

### Finding 3 — Template version matters more than model version for sub-agent fabrications

The Okahu prod run (row 1, 43%) and the claude-sonnet-4-6 Okahu template run (row 3, 100%) used the same `hallucination` v1 template. The 57-point accuracy difference must be attributed to either conversation content or pipeline-level processing, not the template. 

For the preamble+CSV format, no template version produced high accuracy across all models. Template upgrades cannot compensate for a format that obscures the reality boundary.

### Finding 4 — Minor hallucinations are unreliable across all preamble+CSV runs

| Model | minor_hallucination recall |
|-------|--------------------------|
| GPT-4o temp=0 | 0/13 |
| GPT-5.5 temp=1 | 1/13 |
| claude-sonnet-4-6 | 1/13 |
| claude-sonnet-4-20250514 | 3/13 |
| Extracted-output runs | 13/13 |

Minor hallucinations in the test corpus are scope drift cases — single inferences or characterizations added by sub-agents without tool-call support. These are never visible in a flat log because they appear in context where the agent's elaboration looks consistent with the surrounding conversation. All preamble+CSV models failed to catch them. The extracted-output format surfaced them by making the tool return (and its absence) visible as a discrete datum.

---

## Recommendations

1. **Discontinue preamble+CSV as the primary judge input format for multi-agent hallucination evaluation.** The format systematically misses sub-agent fabrications and minor scope-drift hallucinations regardless of model choice.

2. **Move to the causal structure format** documented in `docs/judge_input_format_causal_structure.md`: `turn_id, seq, parent_seq, span_type, agent, content` with typed rows (`tool_return`, `agent_claim`, `system_prompt`, etc.). This makes the reality boundary explicit without requiring the judge to reconstruct causality from a flat history.

3. **Upgrade the `hallucination` v1 template** before running further evaluations with the Okahu prod service. Any updated template should be guided by three principles derived from the failure analysis:

   - **Define what counts as authoritative ground truth.** Every judge model defaulted to treating the most recent agent output as ground truth. The template must explicitly declare that only tool returns are authoritative — everything else is a claim to be verified. Without this, the judge evaluates relay fidelity (did the final response match the sub-agent?) rather than grounding fidelity (did the sub-agent's claim match what a tool actually returned?). The causal format makes this structural; the template contract makes it explicit regardless of format.

   - **Specify severity by content type, not by agent layer.** The judge consistently discounted severity because the supervisor "faithfully relayed" the sub-agent. Severity must be defined by what was fabricated — identifiers, monetary values, confident claims from empty tool returns — not by which agent introduced the error or how many hops it traveled. A supervisor that faithfully relays a sub-agent's major hallucination is producing a major hallucination.

   - **Distinguish sourced facts from agent inferences.** The judge accepted sub-agent characterizations as verified because they appeared in the conversation. The template must require classifying each claim by origin — tool result, user input, or agent inference — and treating agent inferences as unverified by default. This closes the scope-drift miss category.

   These three principles are ordered by dependency: principle 1 is the foundation (without it, 2 and 3 cannot be applied correctly), principle 2 determines the label, principle 3 determines what gets examined. Together they address the root causes behind 36 of the 38 misses in the preamble+CSV runs.

4. **Do not evaluate GPT-5.5 at temperature=1.** The non-determinism invalidates run-level accuracy comparisons. Re-run when Responses API supports temperature=0.
