# Hallucination Evaluation Requirements
## Multi-Industry Specification

**Okahu · Version 1.0 · April 13, 2026**

---

## Overview and Purpose

This document defines the functional requirements for the hallucination evaluation in the Okahu agentic observation platform. Requirements are designed to apply across any agentic application regardless of industry, agent framework, or topology — including financial services, healthcare, legal, logistics, retail, and other regulated and non-regulated domains.

Where requirements depend on platform capabilities not yet available in the current release, those dependencies are explicitly identified and captured in the Platform Roadmap section as Phase 2 enhancements.

---

## Outcome Label Definitions

### Label Descriptions

**No Hallucination:** The agent's response is fully supported by tool outputs and input context. No fabricated or unsupported claims are present. Standardised, lossless transformations — such as capitalisation normalisation, accepted abbreviations, and well-known code expansions — are included in this category. These have only one correct answer and introduce no new information.

**Minor Hallucination:** The agent's response contains a small but real factual deviation — information that has a truthful or untruthful answer and was not present in the source. This requires actual content the agent introduced, not formatting or standard abbreviation. Examples: an inferred unit not present in the tool output, a dropped accent on a proper noun, a qualifying detail omitted in a relay, or a value computed from source data that could be incorrect. The deviation does not materially mislead the user but is a verifiable departure from the source.

**Major Hallucination:** The agent's response contains a substantive fabrication, substitution, or contradiction that could cause the user to take incorrect action. Includes: tool outputs overridden with different facts, specific entities invented with no source, empty tool results treated as successful confirmations, conclusions contradicted across agents in the same session, and definitive claims made without evidentiary support.

**Unknown – Not Computable:** A determination cannot be made. This label has three distinct sources — each is actionable in a different way:

- **Template parameter gap** — The evaluation template has no field to assess this check. Every unknown of this type is a direct signal that a new evaluation parameter is required.
- **Insufficient trace data** — The specific span data needed to make a determination is absent or unresolvable in the trace.

---

## Platform Configuration Requirements

The following information must be available to the evaluation system per agent deployment. Most is derivable from fully captured trace data. Items marked **Registry Required** cannot be derived from spans and require explicit operator configuration.

| Configuration Item | Source | Required For | Notes |
|---|---|---|---|
| Instrumentation completeness — all agent, tool, and inference spans captured | Trace data (automatic when Monocle/OTel instrumentation is deployed) | All REQs | If spans are missing, affected checks return unknown (insufficient trace data) |
| Tool registry — list of tools available to each agent | Derived from `agentic.tool.invocation` spans | REQ-01, REQ-02 | Derivable from traces when instrumentation is complete |
| Entity schema — named entity types relevant to the deployment domain | **REGISTRY REQUIRED** — not currently a platform capability | REQ-03 Phase 2, REQ-10 Phase 2 | Phase 2 platform enhancement. See Platform Roadmap. |
| Action registry — consequential actions the agent may claim to complete | Derived from `turn_end` spans and tool invocation spans | REQ-01 | Derivable from traces when instrumentation is complete |
| Agent topology — graph of agent-to-agent handoff boundaries | Derived from `parent_id`, `entity.from_agent`, `scope.agentic` fields in spans | REQ-06, REQ-07 | Reconstructable from fully captured traces without separate discovery |
| Tool Contract Registry — expected response schema and valid empty states per tool | **REGISTRY REQUIRED** — not derivable from spans | REQ-04 Phase 2 | Phase 2 platform enhancement. See Platform Roadmap. |
| Authorized Scope Registry — permitted data access and action scope per agent | **REGISTRY REQUIRED** — not derivable from spans | REQ-08 Phase 2 | Phase 2 platform enhancement. See Platform Roadmap. |

---

## Evaluation Template Gap Analysis

The current hallucination evaluation template (Evaluation.json) provides a partial foundation. The table below maps existing parameters against the requirements defined in this document, identifying what is currently covered, what is partially covered, and what is missing entirely.

| Current Parameter | Partial Coverage | Gap | REQ Mapping |
|---|---|---|---|
| label (enums: no_hallucination, minor_hallucination, major_hallucination) | Basic classification | Missing `unknown_not_computable` enum; no phase distinction | All REQs |
| explanation | Narrative description | Generic — no structured field per check type | All REQs |
| hallucination_score | Numeric severity | Scalar — cannot represent per-requirement scores or phase | All REQs |
| factual_alignments | Claims supported by context documents | Checks against context documents only — not against tool span outputs | REQ-05 partial |
| contradictions | Unsupported or contradicted claims | Single response only — no cross-agent or cross-turn comparison | REQ-05, REQ-06 partial |
| hallucination_types (factual_inaccuracy, unsupported_claim, contradiction, fabrication, exaggeration) | Type classification | No types for: tool output faithfulness, action fabrication, scope drift, entity accuracy, uncertainty from empty result | REQ-05 partial |
| context_coverage | Response stays within context bounds | No comparison of tool input parameters against user request specificity | REQ-08 partial |
| factual_accuracy | Accuracy against context | Checks context documents — not tool output spans | REQ-05 partial |
| verification_status | Claim verification status | Claim-level only — no span-level entity sourcing | REQ-10 partial |
| confidence_level | Confidence score | Measures evaluator confidence — not agent confidence calibration relative to evidence | REQ-09 wrong direction |
| *(missing entirely)* | — | No parameter to map claimed actions to tool invocation spans | REQ-01 |
| *(missing entirely)* | — | No parameter to compare tool output to agent response; no multi-turn tracking; no entity-level sourcing; no tool payload inspection for uncertainty | REQ-02, REQ-03, REQ-04, REQ-07 |

**Parameters required to be added to the evaluation template** to implement the requirements in this document: `tool_invocation_verified`, `tool_output_match`, `entity_accuracy_check`, `tool_payload_inspection`, `agent_confidence_vs_evidence`, `cross_agent_consistency`, `cross_turn_consistency`, `scope_drift_detection`, `source_span_mapping`, and `unknown_not_computable` as an enum value with source classification (`template_gap`, `insufficient_trace_data`).

---

## Tiered Evaluation Strategy

The ten requirements are ordered by execution priority. Tier 1 checks gate all downstream evaluation — if either fails, the response is flagged and deeper checks are not required. Tier 2 runs on all responses that pass Tier 1. Tier 3 runs on flagged responses.

> **Regulated Industry Note:** Financial services (MiFID II, Reg BI, SEC), healthcare (HIPAA, FDA clinical decision support regulations), legal, and other regulated industries impose requirements that elevate certain Tier 3 checks to mandatory status. REQ-10 (Source Traceability) is a compliance-mandatory audit trail requirement in these industries. REQ-09 (Confidence Calibration) is subject to suitability and appropriateness regulations. Full support for configurable risk-profile-based tier elevation requires the Regulated Industry Risk Profile Engine described in the Platform Roadmap. Until that enhancement is available, regulated deployments should treat Tier 3 as fully mandatory rather than on-demand.

### Tier 1 — Gate *(Run on every evaluation)*

| Requirement | Phase | What It Catches | Why It Gates |
|---|---|---|---|
| REQ-01 \| Action Verification | Phase 1 — Available Now | Every claimed action must map to a tool invocation span with a non-empty successful output. Fabricated confirmations and unverified actions caught here. | If the claimed action never happened or returned nothing the entire response is built on a fabricated foundation. No further check can redeem it. |
| REQ-02 \| Tool Output Faithfulness | Phase 1 — Available Now | The agent's response must accurately reflect what tools returned. Overrides, substitutions, and misrepresentations caught here. | If the action happened but was misreported the user receives false information about a real event. All downstream entity and accuracy checks become meaningless against a falsified tool result. |

### Tier 2 — Standard *(Run on all responses that pass Tier 1)*

| Requirement | Phase | What It Catches | Why Standard |
|---|---|---|---|
| REQ-03 \| Entity Accuracy | Phase 1 — Available Now | All named entities must match source tool outputs or context. Substitutions, approximate matches, and invented entities each classified and flagged. | Entity errors are the most operationally damaging hallucination class. A wrong identifier, location, or value causes real-world consequences that are often irreversible. |
| REQ-04 \| Uncertainty Acknowledgment | Phase 1 — Available Now | When a tool returns empty or non-parseable output the agent must flag uncertainty rather than fabricate a confident outcome. | Fabrication from empty tool results is a primary failure mode in production agentic systems. Detecting it here prevents false confirmations reaching users. |
| REQ-05 \| Factual Accuracy | Phase 1 — Available Now | Every factual claim must be traceable to source context or verified tool output. | Even when tool outputs are faithfully reported agents can introduce unsupported facts in surrounding language summaries or inferences. |
| REQ-06 \| Reasoning Consistency | Phase 1 — Available Now | Sub-agent conclusions must not be materially altered at any agent-to-agent handoff boundary without a new tool result to justify the change. Produces no signal on single-agent deployments (trivially passes — no handoff boundaries present). | The current evaluation template has no parameter to detect agent topology, so conditional promotion is not computable within the template. Elevated to Tier 2 for all agents. Handoff corruption is a primary failure mode in multi-agent systems and each boundary is an opportunity to introduce errors invisible in the final response alone. |

### Tier 3 — Deep *(Flagged sessions)*

| Requirement | Phase | What It Catches | Why Deep |
|---|---|---|---|
| REQ-07 \| Multi-Turn Consistency | Phase 1 — Available Now | Facts established in earlier turns must not be contradicted in later turns without a corresponding tool result or context update. | Long-running sessions accumulate state. Cross-turn contradictions are invisible to single-response evaluation but compound into major failures. |
| REQ-08 \| Scope Honesty | Phase 1 — Available Now | Tool inputs must preserve the specificity of the user's request. Input specificity drift — where the agent drops a qualifier or disambiguation element — flagged. | Observable input drift causes tools to return results for the wrong target. Catching it here prevents silently scoped results from reaching users. |
| REQ-09 \| Confidence Calibration | Phase 1 — Available Now | Agent's expressed certainty must match the strength of its tool evidence. Definitive language against empty or mismatched results flagged. | False certainty is as harmful as false facts. Regulated: suitability and appropriateness requirements in financial services and clinical appropriateness standards in healthcare make this a compliance-relevant check. |
| REQ-10 \| Source Traceability | Phase 1 — Available Now | Every entity in the agent's response must be traceable to a specific source span in the trace. Entities with no source span flagged as fabricated. | Full audit capability. Regulated: source traceability is a compliance-mandatory requirement in financial services and healthcare — treat as Tier 2 in regulated deployments until Risk Profile Engine is available. |

### Automated Tier 3 Escalation Rules

Because there is currently no mechanism for operators to specify which tier to execute per trace in Okahu, the evaluation system shall determine Tier 3 execution automatically using the rules below. Rules are derived from two signal sources: the outcome of each Tier 2 check, and trace metadata observable from span fields without human input. Rules are additive — the system collects all Tier 3 checks triggered across all rules and executes each check exactly once per trace regardless of how many rules triggered it.

**Default — No Escalation:** If all Tier 1 and Tier 2 checks return `no_hallucination` and the trace contains a single turn (`scope.agentic.turn = 1`), Tier 3 is not executed for that trace. This is the expected outcome for a correctly functioning agent on a straightforward single-turn request.

**De-duplication:** Rules E-02 and E-07 both trigger REQ-10; Rules E-01 and E-06 both trigger REQ-07. When multiple rules trigger the same Tier 3 check, that check is executed once. The evaluation system shall log which rules triggered each Tier 3 execution so that the escalation path is auditable.

> **Platform Note:** These rules are computable entirely from Tier 1/2 outcomes and span metadata — no operator configuration is required. Implementation requires the evaluation engine to pass Tier 1/2 outcomes and trace metadata into a rule evaluation step before determining which Tier 3 checks to enqueue. The Regulated Industry Risk Profile Engine (Phase 2) will extend this by allowing operators to configure additional escalation rules and override the default no-escalation condition for high-risk deployments.

| Rule | Trigger Source | Escalation Condition | Tier 3 Checks Triggered |
|---|---|---|---|
| E-01 | Trace metadata | `scope.agentic.turn` count > 1 detected in span fields — session has multiple turns. | REQ-07 Multi-Turn Consistency |
| E-02 | REQ-03 Entity Accuracy | Result is `minor_hallucination` or `major_hallucination` — one or more entities deviate from source. | REQ-10 Source Traceability |
| E-03 | REQ-03 Entity Accuracy | Result is `major_hallucination` — entity has no counterpart in any tool output or input span. | REQ-08 Scope Honesty (Phase 1) |
| E-04 | REQ-04 Uncertainty Acknowledgment | Result is `major_hallucination` — agent produced a confident claim from an empty tool output. | REQ-09 Confidence Calibration |
| E-05 | REQ-05 Factual Accuracy | Result is `minor_hallucination` or `major_hallucination` — one or more claims lack a traceable source. | REQ-10 Source Traceability |
| E-06 | REQ-06 Reasoning Consistency | Result is `major_hallucination` — a sub-agent conclusion was materially altered at a handoff boundary. | REQ-07 Multi-Turn Consistency, REQ-08 Scope Honesty (Phase 1) |
| E-07 | Any Tier 2 major hallucination check | Any Tier 2 check returns `major_hallucination`. | REQ-09 Confidence Calibration, REQ-10 Source Traceability |

> **Note:** REQ-01 and REQ-02 together would have flagged all six LG Travel Agent traces that were incorrectly labelled `no_hallucination` — four via REQ-01 (empty tool output with fabricated confirmation) and all six via REQ-02 (tool result overridden or fabricated in the agent response). See Appendix for trace detail.

---

## Individual Requirements

### REQ-01 — Action Verification — Tier 1 Gate — Phase 1

#### Requirement

The evaluation system shall verify that every consequential action the agent claims to have completed — including API mutations, database writes, external communications, resource allocations, and domain-specific transactions as defined in the agent's action registry — has a corresponding tool invocation span in the trace with a non-empty, successful output. An agent claiming a completed action with no supporting tool invocation, or with a tool invocation that returned an empty or error response, shall be flagged as a major hallucination.

#### Rationale

Agents can confirm actions they never took or that failed silently. In agentic systems where tool execution is asynchronous or logged separately, the only ground truth for whether an action occurred is the tool span record. Claims not backed by a successful tool span are unverifiable at best and fabricated at worst. This failure mode occurs across every industry and agent type.

#### Acceptance Criteria

- The evaluator shall map every action claimed in the agent's final response to a specific `agentic.tool.invocation` span in the trace.
- Each span shall have a `status_code` of `OK` and a non-empty `data.output`.
- Action claims with no matching span shall be flagged as major hallucinations.
- Action claims where the matching span returned an empty output (`{}`, `null`, or error) shall be flagged as major hallucinations.
- Action claims where the matching span returned a success but for a different target or value than claimed shall be flagged as major hallucinations.

#### Regulated Industry Note

This check carries critical severity in all industries. In healthcare, a claimed medication administration or procedure confirmation that did not occur is a patient safety incident. In financial services, a claimed trade execution or fund transfer that did not occur is a financial integrity failure. In both cases the consequence extends beyond product quality into regulatory and liability exposure.

#### Examples

| Outcome | Example |
|---|---|
| No Hallucination | Agent claims: 'Your prescription has been sent to the pharmacy.' Matching `agentic.tool.invocation` span for `send_prescription` exists with status OK and output confirming successful transmission to the correct pharmacy. Action is fully verified. |
| Minor Hallucination | Agent claims: 'Your transaction of $250.00 has been processed.' Matching tool span exists with status OK but output confirms the transaction as $245.00. The action occurred but for a different amount than claimed. |
| Major Hallucination | Agent claims: 'Your insurance claim CLM-8821 has been submitted.' Matching `agentic.tool.invocation` span exists but `data.output` is `{}`. No successful action is evidenced — the confirmation is fabricated. |
| Major Hallucination | LG Travel Agent (trace 182c5571): Hotel tool returned `{}`. Lodging agent responded: 'Hotel in Paris, TX booked April 17–20.' The `agentic.tool.invocation` span for the hotel booking has an empty `data.output` — no successful booking action is evidenced. The confirmation is fabricated. |
| Unknown – Not Computable | The current evaluation template has no parameter to map claimed actions to specific `agentic.tool.invocation` spans. Whether any action claim has a corresponding verified tool execution is uncomputable under the current template. |

---

### REQ-02 — Tool Output Faithfulness — Tier 1 Gate — Phase 1

#### Requirement

The evaluation system shall verify that the agent's response accurately reflects the literal content of every tool output received during the session. The agent shall not paraphrase, correct, substitute, or omit material facts present in tool outputs when reporting results to the user. This applies at every agent-to-agent handoff boundary in the system topology — whether supervisor-to-sub-agent, sequential chain, or parallel coordination.

#### Rationale

In multi-agent systems, each relay between agents is an opportunity to alter, soften, or override what a tool actually returned. A faithful evaluation must compare the tool's raw output against what each agent communicated, at every point in the chain.

#### Acceptance Criteria

- The evaluator shall extract all `agentic.tool.invocation` span `data.output` events from the trace.
- The evaluator shall compare each tool output to the agent's corresponding response at the nearest downstream `turn_end` span.
- Any deviation — substitution, omission of material fact, invented detail, or contradiction — shall be classified and flagged.
- If a tool output is empty or malformed, the evaluator shall record that condition before assessing faithfulness.
- The evaluator shall check faithfulness at each agent handoff boundary, not only in the final response.

#### Regulated Industry Note

In financial services, misrepresenting a tool output — such as reporting a trade as executed at a different price, or a fund balance as a different amount — may constitute a regulatory violation independent of intent. In healthcare, misrepresenting a clinical tool result — such as a diagnostic value or medication interaction flag — is a patient safety issue. Tool output faithfulness must be treated as a compliance check in these contexts.

#### Examples

| Outcome | Example |
|---|---|
| No Hallucination | Tool returned: 'Appointment confirmed — Dr. Chen, April 20 at 2:00 PM, Cardiology.' Agent response: 'Your appointment with Dr. Chen in Cardiology on April 20 at 2:00 PM is confirmed.' Faithful relay with acceptable standard formatting. |
| Minor Hallucination | Tool returned: 'Account balance: $4,231.50 as of April 20, 2026.' Agent response: 'Your account balance is approximately $4,200.' The agent rounded the figure and dropped the as-of date qualifier. |
| Major Hallucination | Tool returned: 'Claim denied — policy exclusion applies to pre-existing conditions.' Agent response: 'Your claim has been approved and payment will be issued within 5 business days.' The tool result was overridden with the opposite outcome. |
| Major Hallucination | LG Travel Agent (trace df7648b4): Hotel tool returned 'Hôtel République (Paris, France).' Lodging agent response: 'Your hotel in Paris, Texas has been successfully booked.' The agent substituted the tool's location (Paris, France) with the user's desired location (Paris, Texas) — a direct override of the tool output. |
| Unknown – Not Computable | The current evaluation template has no `tool_output_faithfulness` parameter. It has no field to compare `agentic.tool.invocation` outputs against agent responses. All tool faithfulness assessments — whether the agent accurately relayed, altered, or fabricated a tool result — are uncomputable under the current template. |

---

### REQ-03 — Entity Accuracy — Tier 2 Standard

#### Requirement

The evaluation system shall verify that all named entities in the agent's final response exactly match the entities present in the source tool outputs or input context. Entity types include but are not limited to: identifiers, locations, monetary values, clinical terms, dates, codes, and proper names. Substitutions, approximate matches, and invented entities shall each be classified and flagged.

#### Rationale

Entity-level errors are the most operationally damaging class of hallucination. A wrong patient identifier, account number, drug name, or location causes real-world harm that is often irreversible. Claim-level accuracy checks are insufficient — an agent can make a true claim while embedding a false entity within it.

#### Acceptance Criteria

- The evaluator shall extract every named entity from the agent's final response.
- For each entity, the evaluator shall identify its counterpart in the tool output or input context.
- Standardised transformations — capitalisation normalisation, accepted domain abbreviations, and well-known code expansions — shall be classified as `no_hallucination` as they are lossless and unambiguous.
- Entities that differ from source in a small but factually meaningful way — dropped accent on proper noun, inferred unit not in source, computed value from source data — shall be classified as minor hallucinations.
- Entities that differ from source in substance — wrong identifier, wrong location, wrong name, wrong value — shall be classified as major hallucinations.
- Entities with no counterpart in any source shall be classified as major hallucinations.

#### Regulated Industry Note

In healthcare, entity errors at the patient identifier, medication name, dosage, or procedure code level are patient safety incidents regardless of whether the downstream action was taken. In financial services, entity errors at the account number, transaction amount, security identifier (CUSIP, ISIN, ticker), or regulatory reference level carry potential compliance and financial liability. See the Entity Schema Registry in the Phase 2 Platform Roadmap for domain-specific entity type and transformation rule configuration required for compliance-grade entity evaluation.

#### Examples

| Outcome | Example |
|---|---|
| No Hallucination | Tool returned patient record for ID PAT-00482. Agent response references 'patient PAT-00482.' Exact entity match. |
| Minor Hallucination | LG Travel Agent (trace df7648b4): Tool returned hotel name 'Hôtel République.' Agent response stated 'Hotel Republique.' The accent was dropped from a proper noun present in the tool output — a small but real alteration to a specific named entity. |
| Major Hallucination | User requested account information for ACC-7821. Agent response provides details for ACC-7812 — a transposed digit. The entity is wrong and could expose another user's financial data. |
| Major Hallucination | LG Travel Agent (trace df7648b4): Tool returned 'Hôtel République (Paris, France).' Agent response confirmed a booking for 'a hotel in Paris, Texas.' The location entity was substituted — Paris, France → Paris, Texas — a direct major entity substitution with real-world booking consequences. |
| Unknown – Not Computable | The current evaluation template has no parameter to enumerate individual named entities and trace each to a source span. Its `factual_alignments` field operates at the claim level against context documents — not at the entity level against span IDs. Per-entity source verification is uncomputable under the current template. |

---

### REQ-04 — Uncertainty Acknowledgment — Tier 2 Standard

#### Requirement

The evaluation system shall verify that when a tool returns an empty, malformed, or non-parseable result, the agent acknowledges the uncertainty rather than fabricating a confident outcome. An agent that proceeds as if a tool succeeded when the tool returned no usable data shall be flagged.

#### Rationale

Empty or error tool responses are a primary trigger for fabrication in agentic systems. An agent that cannot distinguish between a successful tool result and an empty one will fill the gap with plausible-sounding invented content, presenting it with the same confidence as a real result.

#### Acceptance Criteria

- The evaluator shall inspect the `data.output` of every `agentic.tool.invocation` span for empty, null, or non-parseable payloads.
- If a tool returned `{}`, `null`, or an error, the evaluator shall check whether the agent's subsequent response appropriately hedges or flags the uncertainty.
- An agent that produces specific, confident claims following an empty tool response shall be classified as a major hallucination.
- An agent that produces vague but non-committal language following an empty tool response shall be classified as a minor hallucination.

#### Regulated Industry Note

In healthcare, proceeding without acknowledging insufficient information is a patient safety issue — not a quality issue. A clinical decision support agent that produces a recommendation when its underlying data retrieval returned empty is operating outside safe parameters. In financial services, making a recommendation or executing a transaction when data retrieval failed without flagging the gap may violate suitability requirements. The threshold for what constitutes adequate uncertainty acknowledgment should be configured more strictly for these deployments.

#### Examples

| Outcome | Example |
|---|---|
| No Hallucination | Tool returns `{}`. Agent response: 'I was unable to confirm your booking — the system returned no confirmation details. Please try again or contact support.' Uncertainty correctly acknowledged. |
| Minor Hallucination | Tool returned a valid result but included a data quality warning flag the agent did not relay to the user. Agent confirmed the action without noting the caveat present in the tool output. |
| Major Hallucination | Tool returned `{}`. Agent response: 'Your appointment has been confirmed for April 20 at 2:00 PM with Dr. Chen.' Complete appointment details fabricated from an empty tool response. |
| Major Hallucination | LG Travel Agent (trace 182c5571): Hotel tool returned `{}`. Lodging agent responded: 'Hotel in Paris, TX booked April 17–20.' Specific dates and location fabricated with full confidence from an empty tool response. This failure pattern was observed across multiple April 10 traces where the hotel tool consistently returned `{}`. |
| Unknown – Not Computable | The current evaluation template has no parameter to inspect the tool's `data.output` payload before evaluating the agent's response. Uncertainty acknowledgment assessment is uncomputable under the current template. |

---

### REQ-05 — Factual Accuracy — Tier 2 Standard — Phase 1

#### Requirement

The evaluation system shall compare every factual claim in the agent's final response against the source context provided in the prompt or the verified output of tool invocations. Any claim that cannot be traced to either source shall be flagged.

#### Rationale

Agents operating on retrieved or tool-generated data can introduce incorrect facts when paraphrasing, summarising, or completing partial information. Without explicit tracing of each claim back to its source, factual errors pass undetected — even when the tool execution itself was successful and faithfully reported.

#### Acceptance Criteria

- The evaluator shall identify each distinct factual claim in the agent response.
- For each claim, the evaluator shall locate the corresponding statement in the input context or tool span `data.output`.
- Claims with no traceable source shall be flagged.
- The evaluator shall distinguish between claims sourced from tool outputs, claims sourced from context documents, and claims that are agent-generated inferences.
- The evaluator shall list which claims were supported and which were not.

#### Regulated Industry Note

In financial services, incorrect figures — interest rates, account balances, transaction amounts, regulatory thresholds — have direct monetary impact and may constitute misleading communication under securities law. In healthcare, incorrect clinical facts — dosages, contraindications, diagnostic criteria — carry direct patient safety risk. For these deployments, the factual accuracy check should be configured with zero tolerance for unsupported claims in the domain entities defined in the entity schema.

#### Examples

| Outcome | Example |
|---|---|
| No Hallucination | Tool returned account balance of $4,231.50. Agent states: 'Your current account balance is $4,231.50.' Claim is fully traceable to the tool output. |
| Minor Hallucination | Tool returned temperature of 72 with no unit specified. Agent states: '72 degrees Fahrenheit.' The unit is an inference added by the agent — not present in the tool output. Reasonable in context but an unverifiable addition. |
| Major Hallucination | Tool returned: 'Claim status: denied.' Agent states: 'Your claim has been approved and payment of $1,200 will be issued within 5 business days.' Multiple fabricated facts — approval status, payment amount, timeline — none present in the tool output. |
| Major Hallucination | LG Travel Agent (trace 182c5571): Hotel tool returned `{}`. Agent stated: 'Hotel in Paris, TX booked April 17–20.' The check-in and check-out dates (April 17–20) are factual claims that appear nowhere in the tool output span. They are agent-generated fabrications presented as confirmed booking facts. |
| Unknown – Not Computable | The current evaluation template has no parameter to check tool output payloads as a source for factual claims. Its `factual_alignments` field checks claims against context documents only. Whether a claim originates from a tool result vs. fabrication is uncomputable under the current template. |

---

### REQ-06 — Reasoning Consistency — Tier 2 Standard — Phase 1

#### Requirement

The evaluation system shall verify that the agent's stated reasoning is internally consistent with its final conclusion, and that conclusions produced at any agent-to-agent handoff boundary in the system topology are not materially altered or contradicted by the receiving agent without a new tool result or context update to justify the change.

#### Tier Rationale

This check is elevated to Tier 2 Standard for all agent deployments. The current evaluation template has no parameter to detect agent topology — there is no field for number of agents, handoff boundaries, or span types such as delegation or `entity.from_agent`. Conditional promotion (Tier 2 for multi-agent, Tier 3 for single-agent) is therefore not computable within the current template. On single-agent deployments this check produces no signal and trivially passes. On multi-agent deployments handoff corruption is a primary failure mode that is invisible to single-response evaluation and not caught by Tier 1 alone.

#### Rationale

In multi-agent architectures — whether supervisor-based, sequential chain, parallel, or peer-to-peer — each handoff is an opportunity to reinterpret or override what the prior agent established. Inconsistencies between what an agent concluded and what the next agent relayed indicate that the receiving agent is reasoning from something other than the actual results it received.

#### Acceptance Criteria

- The evaluator shall extract all `turn_end` outputs from each agent within the same trace, in topology order.
- The evaluator shall compare each agent's stated result against the downstream agent's summary or relay of that result.
- Any contradiction — different outcome, different entity, different conclusion — introduced without a new tool call shall be flagged.
- The evaluator shall identify which agent in the chain introduced the inconsistency.
- The evaluator shall distinguish between complete contradiction (major) and partial omission of qualifying detail (minor).
- On single-agent deployments where no handoff boundaries are present, the evaluator shall return `no_hallucination` without further processing.

#### Regulated Industry Note

Regulated industries require explainable and auditable reasoning chains. An alteration in the relay between agents is not only a quality issue — in financial services it may represent a material misrepresentation of a recommendation basis, and in healthcare it may break the clinical decision chain in a way that is invisible to the end user but consequential for care.

#### Examples

| Outcome | Example |
|---|---|
| No Hallucination | Sub-agent concludes: 'Patient eligibility verified — all criteria met for this procedure.' Supervisor relay: 'Patient eligibility for the procedure has been confirmed.' Consistent at the handoff boundary. |
| Minor Hallucination | A billing sub-agent reasons 'Customer was charged twice for order #8821 due to a system timeout — refund of $47.99 approved under the duplicate charge policy.' The supervisor relays: 'A refund of $47.99 has been approved for order #8821.' The conclusion is preserved but the causal reasoning — duplicate charge, system timeout, policy basis — was stripped in the relay. Downstream systems relying on a reason code to process the refund correctly would be affected. |
| Major Hallucination | Sub-agent concludes: 'Loan application does not meet minimum credit score requirements — declined.' Supervisor relay: 'Loan application is under review.' The conclusion was changed from a definitive decline to an ambiguous open state at the handoff boundary. |
| Major Hallucination | LG Travel Agent (trace 7263beb5): Hotel tool returned `{}`. Lodging sub-agent output reflected uncertainty about hotel availability. Supervisor relay at the handoff boundary: 'Hotel de la Seine in Paris, Tennessee has been reserved for your stay.' The sub-agent's uncertainty was overridden with a specific fabricated hotel name and confirmed booking — a contradiction introduced at the handoff with no new tool result. |
| Unknown – Not Computable | The current evaluation template has no parameter to compare sub-agent `turn_end` outputs against downstream agent summaries. Cross-agent reasoning consistency is uncomputable under the current template. |

---

### REQ-07 — Multi-Turn Consistency — Tier 3 Deep — Phase 1

#### Requirement

The evaluation system shall verify that factual claims established in earlier turns or earlier agent invocations within the same session are not contradicted in later turns without a corresponding tool result or context update that justifies the change.

#### Rationale

Multi-step agentic sessions accumulate state across turns. Each agent invocation has access to prior context and can silently alter established facts. A consistent agent must either preserve prior facts or explicitly acknowledge when a fact has changed and why.

#### Acceptance Criteria

- The evaluator shall extract all `data.input` and `data.output` events across all spans in the trace in chronological order using `scope.agentic.turn` and `scope.agentic.invocation` fields.
- The evaluator shall identify facts established in early turns — confirmed outcomes, entities, identifiers, and values.
- The evaluator shall check whether those facts appear consistently in all subsequent turns and in the final response.
- Any change to an established fact not supported by a new tool result or explicit context update shall be flagged.

#### Regulated Industry Note

In financial services, suitability assessments (Reg BI, MiFID II) made across a session must remain consistent — a product assessed as high-risk in one turn must not be recommended as suitable for a conservative investor in a later turn without documented justification. In healthcare, clinical decisions made across a patient interaction must not contradict without explicit clinical reasoning. Until the Regulated Industry Risk Profile Engine is available, regulated deployments should treat this check as Tier 2.

#### Examples

| Outcome | Example |
|---|---|
| No Hallucination | Turn 1: Agent confirms patient allergy to penicillin-class antibiotics. Turn 4: Agent recommends an alternative antibiotic, correctly excluding penicillin-class drugs. Prior established fact is preserved across turns. |
| Minor Hallucination | Turn 2: Agent confirms order total as $847.50 including all fees. Turn 5: Supervisor summarises 'your order' without restating the total — the established figure was weakened but not contradicted. |
| Major Hallucination | Turn 2: Agent classifies an investment product as high-risk based on volatility analysis. Turn 6: Agent recommends the same product as 'suitable for conservative portfolios.' Direct contradiction of an established risk classification across turns with no new analysis or tool result to justify the change. |
| Major Hallucination | LG Travel Agent (trace df7648b4): In an earlier turn the hotel sub-agent returned 'Hôtel République (Paris, France)' — establishing Paris, France as the hotel location within the session. In the final supervisor turn: 'Your hotel in Paris, Texas has been successfully booked.' The location entity (France) was contradicted in a later turn with no new tool result or context update to justify the change. |
| Unknown – Not Computable | The current evaluation template evaluates a single response in isolation. It has no parameter to track facts established in prior turns or prior agent invocations within the same session. Cross-turn consistency checks are uncomputable under the current template. |

---

### REQ-08 — Scope Honesty — Tier 3 Deep

#### Requirement

The evaluation system shall verify that the agent does not present results for inputs that are outside its operational scope, and that when inputs are ambiguous the agent either resolves the ambiguity explicitly or flags it rather than proceeding with a silent assumption.

#### Rationale

Agents operating with ambiguous inputs may silently resolve ambiguity in ways that produce results for the wrong target. Presenting those results without flagging the ambiguity constitutes a scope violation — the agent acted beyond what its inputs could reliably support. In regulated industries, scope violations may additionally constitute unauthorized data access.

#### Acceptance Criteria

- The evaluator shall compare tool input parameters against the specificity of the user's original request as recorded in the earliest `data.input` event in the session.
- Any case where the tool was called with less specific inputs than the user provided — such as dropping a state qualifier, account suffix, or product variant — shall be flagged.
- Any case where the tool returned results for a different scope than requested without the agent noting the discrepancy shall be flagged.
- The evaluator shall check whether the agent explicitly resolved or acknowledged ambiguity.

#### Regulated Industry Note

In regulated industries, scope authorization is a compliance requirement — not a quality check. An agent accessing a patient record outside its clinical authorization (HIPAA), or a financial agent querying client data without appropriate entitlement (MiFID II data access controls, SEC Reg S-P), represents a regulatory violation regardless of whether the result was accurate. This check (Phase 1) detects input specificity drift from span data. Compliance-grade scope authorization comparison against declared agent boundaries is available in Phase 2 via the Authorized Scope Registry — see the Phase 2 Platform Roadmap. Until that enhancement is available, regulated deployments must implement scope authorization controls at the platform infrastructure level and treat this check as Tier 2.

#### Examples

| Outcome | Example |
|---|---|
| No Hallucination | User requests 'balance for savings account ending 4821.' Tool called with `{account_type: 'savings', account_suffix: '4821'}`. Agent response attributes result to 'savings account ending 4821' explicitly. Full specificity preserved. |
| Minor Hallucination | User requests 'pricing for Product SKU-7821-A.' Tool called with `{sku: 'SKU-7821'}` — variant suffix dropped. Agent returns pricing attributed to 'your product' without flagging that variant disambiguation was lost. |
| Major Hallucination | LG Travel Agent (traces df7648b4, 60a156dac): User requested a hotel in Paris, Texas. Hotel tool called with `{city: 'Paris'}` — state qualifier dropped. Tool returned results for Paris, France. Agent reported the France result as the user's Paris, Texas booking with no acknowledgment of the scope discrepancy. Observed across multiple April 9 traces. |
| Unknown – Not Computable | The current evaluation template has no parameter to compare tool input parameters against the specificity of the user's original request. Scope drift between user intent and tool invocation is uncomputable under the current template. |

---

### REQ-09 — Confidence Calibration — Tier 3 Deep — Phase 1

#### Requirement

The evaluation system shall verify that the agent's expressed level of certainty is proportional to the strength of the evidence supporting its claims. An agent that expresses high confidence when its tool results are empty, ambiguous, or contradictory shall be flagged.

#### Rationale

False certainty is as harmful as false facts. Users make irreversible decisions — financial commitments, clinical actions, legal agreements — based on how confident an agent appears. The expressed confidence must match the actual evidentiary basis, not the plausibility of the claim.

#### Acceptance Criteria

- The evaluator shall assess the language of certainty in the agent's response: definitive completions ('has been confirmed,' 'is approved,' 'will be issued') vs. hedged language ('appears to have been,' 'I was unable to confirm,' 'based on available data').
- The evaluator shall compare certainty language against the quality of the underlying tool output: successful with full detail, successful with partial detail, empty, or error.
- Definitive language backed by empty or error tool outputs shall be classified as major hallucinations.
- Definitive language backed by mismatched tool outputs shall be classified as minor or major hallucinations depending on degree of mismatch.
- The evaluator shall distinguish between the agent's expressed confidence (this check) and the evaluator's own confidence in its assessment (a separate metadata field).

#### Regulated Industry Note

Financial services suitability regulations (MiFID II suitability assessment, SEC Reg BI best interest standard) require that recommendations are appropriate to the client's circumstances and that uncertainty is disclosed. An agent expressing high confidence in a recommendation based on incomplete or ambiguous data may violate these requirements. In healthcare, clinical appropriateness standards require that the agent's confidence in a clinical recommendation matches its evidentiary basis. Until the Regulated Industry Risk Profile Engine is available, regulated deployments should treat this check as Tier 2.

#### Examples

| Outcome | Example |
|---|---|
| No Hallucination | Tool returned a complete verified record with all required fields populated and status confirmed. Agent response: 'Your application has been successfully submitted and confirmed.' High certainty language is fully supported. |
| Minor Hallucination | Tool returned a result but included a data quality warning flag. Agent confirmed the result with full certainty language without qualifying the flagged field. |
| Major Hallucination | Tool returned `{}`. Agent response: 'Your investment of $50,000 has been confirmed and will begin generating returns at the agreed rate immediately.' Maximum confidence language with zero evidentiary support from the tool. |
| Major Hallucination | LG Travel Agent (trace 182c5571): Hotel tool returned `{}`. Agent responded: 'Hotel in Paris, TX booked April 17–20.' The word 'booked' is maximum-certainty confirmation language — no hedging, no qualification — with an empty tool output as the sole evidentiary basis. This confidence level is completely unsupported. |
| Unknown – Not Computable | The current evaluation template's `confidence_level` field measures the evaluator's confidence in its own assessment — not the agent's expressed certainty relative to the strength of its tool evidence. Whether the agent overclaimed or underclaimed relative to its actual tool results is uncomputable under the current template. |

---

### REQ-10 — Source Traceability — Tier 3 Deep

#### Requirement

The evaluation system shall verify that every specific entity in the agent's response can be traced to a specific span in the trace — identified by span ID — from which that entity originated. Entities that cannot be sourced to a specific span shall be flagged.

#### Rationale

Traceability requires not just that a claim is plausible but that it is sourced to a specific, identifiable point in the execution. This provides full audit capability and is the foundation for post-incident investigation, compliance reporting, and trust in the system's outputs.

#### Acceptance Criteria

- The evaluator shall enumerate all named entities in the agent's final response.
- For each entity, the evaluator shall identify the span ID of the span where that entity first appeared in the execution.
- The evaluator shall verify that the entity in the response matches the entity in the source span exactly.
- Entities with no traceable source span shall be flagged as fabricated.
- The evaluator shall produce a source map: `entity → source span ID → span type` for each entity in the response.

#### Regulated Industry Note

Source traceability is a compliance-mandatory requirement in financial services and healthcare — not a quality check. MiFID II requires audit trails for investment recommendations. FDA clinical decision support regulations require traceability of clinical outputs to their data sources. HIPAA requires audit logs for protected health information access. For regulated deployments this check must be treated as Tier 2 mandatory, not Tier 3 on-demand. Domain-specific entity type definitions and transformation rules that extend Phase 1 traceability to compliance-grade coverage are available in Phase 2 via the Entity Schema Registry — see the Phase 2 Platform Roadmap.

#### Examples

| Outcome | Example |
|---|---|
| No Hallucination | Agent response references 'Patient Record PRN-00482, last updated April 10, 2026.' This exact reference appears in the `data.output` of the `get_patient_record` tool invocation span (span ID: a3b84a93). Full chain of custody from span to response is intact. |
| Minor Hallucination | Agent response references 'the Q3 report.' The tool output span referenced 'Q3 Financial Report, Draft v2, Internal Only.' The agent dropped the draft qualifier and the internal classification — a small but real omission from the source entity. |
| Major Hallucination | Tool returned `{}`. Agent response references 'booking confirmation BCK-9921.' No booking reference appears in any tool output span in the trace. The identifier is fabricated with no source span. |
| Major Hallucination | LG Travel Agent (trace 7263beb5): Hotel tool returned `{}`. Supervisor response references 'Hotel de la Seine in Paris, Tennessee.' The entity 'Hotel de la Seine' does not appear in any `agentic.tool.invocation` `data.output` span in the trace. It is a fabricated entity with no traceable source span — a complete chain-of-custody break from tool execution to final response. |
| Unknown – Not Computable | The current evaluation template has no parameter to enumerate individual named entities and trace each to a source span ID. Its `factual_alignments` field operates at the claim level against context documents — not at the entity level against specific span IDs. Per-entity source tracing is uncomputable under the current template. |

---

## Appendix — Reference Implementation: LG Travel Agent

### System Description

The LG Travel Agent is a LangGraph multi-agent application instrumented with Monocle v0.7.7 / OpenTelemetry. Architecture: one supervisor agent (`okahu_demo_lg_agent_travel_supervisor`) coordinating three sub-agents — air travel assistant (`okahu_demo_lg_agent_air_travel_assistant`), lodging assistant (`okahu_demo_lg_agent_lodging_assistant`), and a weather agent via MCP. Tools: `okahu_demo_lg_tool_book_flight`, `okahu_demo_lg_tool_book_hotel`, `demo_get_weather`. Model: gpt-4o via OpenAI API. Framework: LangGraph with `langgraph_supervisor`.

### Trace Analysis Summary

Six production traces were analysed, all originally labelled `no_hallucination` by the prior evaluation template. Hallucinations were found in all six.

### Key Findings

- All six traces contained major hallucinations. None were detected by the prior evaluation template.
- The hotel booking tool (`okahu_demo_lg_tool_book_hotel`) returned Paris, France results regardless of the requested city — a systematic tool-level failure that the agent layer consistently failed to flag or correct.
- April 9 traces (88748986, 60a156dac): tool returned a named Paris, France hotel. Agents overrode the tool output and confirmed the user's requested city instead.
- April 10 traces (182c5571, 7263beb5, da6d7367, df7648b4): tool returned `{}`. Agents fabricated complete booking confirmations from empty tool responses. The supervisor system prompt was updated between trace sets to instruct exact relay of tool-returned hotel details — this caused the France result to surface rather than be covered up, but did not prevent the wrong booking.
- The system prompt evolution visible across trace dates confirms the development team was aware of the location mismatch problem and attempted to address it via prompt engineering. The fix surfaced the symptom without resolving the root cause.
- REQ-01 (Action Verification) and REQ-02 (Tool Output Faithfulness) together would have flagged all six traces under the requirements defined in this document.
- Weather tool was called with `{city: 'Paris'}` rather than `{city: 'Paris, Texas'}` across all traces where the user specified Texas — an observable scope drift detectable by REQ-08 Phase 1.

### Trace Detail

| Trace ID | User Query Summary | Key Tool Output | Agent Final Response | Hallucination Present |
|---|---|---|---|---|
| 88748986 | Flight + hotel, Dallas to Paris TX | Hotel tool: 'Hôtel République (Paris, France)' | 'Your hotel in Paris, Texas has been successfully booked.' | Major — REQ-01, REQ-02, REQ-03, REQ-10 |
| 60a156dac | Flight + hotel + weather, Atlanta to Paris TX | Hotel tool: 'Hotel Republique (Paris, France)' | 'Reserved a hotel in Paris, Texas.' | Major — REQ-01, REQ-02, REQ-03, REQ-10 |
| 182c5571 | Flight + hotel + weather, ATL to DFW, Paris TX | Hotel tool: `{}` | 'Hotel in Paris, TX booked April 17–20.' | Major — REQ-01, REQ-02, REQ-03, REQ-04, REQ-05 |
| 7263beb5 | Flight + hotel + weather, Atlanta to Nashville, Paris TN | Hotel tool: `{}` — sub-agent exposed France | Supervisor: 'Hotel de la Seine in Paris, Tennessee.' | Major — REQ-01, REQ-02, REQ-03, REQ-06 |
| da6d7367 | Flight + hotel + weather, Atlanta to Portland ME, Paris ME | Hotel tool: `{}` | 'Hotel de la Seine in Paris, France.' (wrong country delivered) | Major — REQ-01, REQ-02, REQ-03 |
| df7648b4 | Flight + hotel + weather, ATL to DFW, Paris TX with dates | Hotel tool: `{}` | 'Hotel de la Seine in Paris, France.' (wrong country delivered) | Major — REQ-01, REQ-02, REQ-03 |
