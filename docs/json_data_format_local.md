# Json Data format for evaluations

## Span types referenced by eval templates

Source: `okahu-prod.okahu_templates.eval_templates`, `input` column (queried 2026-05-07).
12 distinct span types appear across the 18 templates currently registered.

| Span type | # templates | Templates referencing it |
|---|---:|---|
| `inference` | 18 | answer_relevancy, argument_correctness, bias, contextual_precision, contextual_recall, contextual_relevancy, conversation_completeness, correctness, frustration, hallucination, knowledge_retention, misuse, offtopic, pii_leakage, role_adherence, sentiment, summarization, toxicity |
| `inference.framework` | 18 | (same set as `inference`) |
| `agentic.turn` | 15 | answer_relevancy, argument_correctness, bias, contextual_relevancy, correctness, hallucination, knowledge_retention, mcp_task_completion, misuse, offtopic, pii_leakage, role_adherence, sentiment, summarization, toxicity |
| `agentic.invocation` | 9 | answer_relevancy, bias, contextual_precision, contextual_recall, contextual_relevancy, correctness, hallucination, knowledge_retention, summarization |
| `agentic.tool.invocation` | 9 | (same 9 as `agentic.invocation`) |
| `embedding` | 9 | answer_relevancy, contextual_precision, contextual_recall, contextual_relevancy, correctness, hallucination, knowledge_retention, pii_leakage, toxicity |
| `embedding.modelapi` | 9 | (same 9 as `embedding`) |
| `retrieval` | 9 | (same 9 as `embedding`) |
| `retrieval.embedding` | 9 | (same 9 as `embedding`) |
| `http.process` | 5 | answer_relevancy, misuse, pii_leakage, role_adherence, toxicity |
| `http.send` | 5 | (same 5 as `http.process`) |
| `agentic.mcp.invocation` | 1 | mcp_task_completion |

Patterns of note:
- `inference` and `inference.framework` are universal — every template references both.
- `embedding` / `embedding.modelapi` / `retrieval` / `retrieval.embedding` always co-occur in the same 9 RAG-flavored evaluators.
- `http.process` / `http.send` always co-occur in the same 5 templates.
- `agentic.mcp.invocation` is unique to `mcp_task_completion`.
- `conversation_completeness` and `frustration` are the only templates that do not reference `agentic.turn` — they key only off inference spans.

## Hallucination template — input column breakdown

Source: `okahu-prod.okahu_templates.eval_templates` where `name = 'hallucination'` (queried 2026-05-07).
Three rows exist for `hallucination`, one per `group_by` value. They share an identical event schema and differ only in the set of span types each one targets.

| id (`group_by`) | version | created_at | input event (name / field / roles) | output event (name / field / roles) | span types |
|---|---|---|---|---|---|
| `evaluation__agent_requests__hallucination` (`agent_requests`) | 34 | 2026-04-07 | `data.input` / `input` / [system, user, assistant] | `data.output` / `response` / [system, user, assistant] | `agentic.turn` |
| `evaluation__traces__hallucination` (`traces`) | 1758174694 | 2025-09-18 | `data.input` / `input` / [system, user, assistant] | `data.output` / `response` / [system, user, assistant] | `inference`, `inference.framework`, `agentic.tool.invocation`, `agentic.invocation`, `embedding`, `embedding.modelapi`, `retrieval`, `retrieval.embedding` |
| `evaluation__agent_sessions__hallucination` (`agent_sessions`) | 1758174694 | 2026-01-23 | `data.input` / `input` / [system, user, assistant] | `data.output` / `response` / [system, user, assistant] | `inference`, `inference.framework`, `agentic.tool.invocation`, `agentic.invocation`, `embedding`, `embedding.modelapi`, `retrieval`, `retrieval.embedding`, `agentic.turn` |

### Analysis

- **Event schema is identical across all three rows.** Input prompt is built from the `input` attribute of `data.input` events, output from the `response` attribute of `data.output` events. Roles `system`, `user`, `assistant` are accepted on both.
- **Differences are span-scoping only.** Each row targets a different aggregation level via `group_by`:
  - `agent_requests` evaluates a single `agentic.turn` span — a turn-level hallucination check.
  - `traces` evaluates the leaf-level execution spans (inference, tool, RAG retrieval, embedding) — explicitly excludes `agentic.turn`. This is the row hit when test code calls `check_eval('hallucination', ...)` with the default `fact_name='traces'`.
  - `agent_sessions` is a superset of `traces` plus `agentic.turn` — used when grouping by an entire session.
- **`traces` is the row this repo's tests use.** `monocle_test_tools/evals/okahu_eval.py` defaults `fact_name` to `traces`, which resolves to `trace_id` in the fact map and selects the `evaluation__traces__hallucination` template. Because that template does **not** include `agentic.turn`, only the leaf inference/tool/embedding/retrieval spans contribute to the prompt.
- **`agent_requests` and `agent_sessions` are not currently used by the LG / CC / FS test suites** in this repo — they would be reachable via `fact_name='agent_requests'` or `fact_name='agent_sessions'` if a test explicitly passed those.
- **Version asymmetry.** `traces` and `agent_sessions` share `version=1758174694` (a unix-timestamp-style stamp from late 2025); `agent_requests` is on `version=34`, suggesting a different release/edit cadence.

## monocle span type data
Type of data we are including
agentic invocation, inference, inference framework, agentic tool invocation, agentic turn

In general, what data we are keeping from in each span
 - general
    - parent_id
 - context
    - span_id
 - attributes
    - (if present) inference.decision.span.id (this appears limited by engine --> present in langgraph, but not in adk)
    - (if present) entity.1.name (aka agent_name)
    - span.type
    - span.subtype
 - events
    - name (both data.input and data.output)
    - attributes.input (input content)
    - attributes.response (output content)

implementation aspect
1) modify query to collect all trace data first, then prepare data
2) modify the json to look like one of the data example we have. It will still include all spans at this step
3) apply span and role the filtering specified by the templates

## single turn

### agentic turn
 - context
    - span_id
 - attributes
    - span.type
    - span.subtype
 - events
    - name (both data.input and data.output)
    - attributes.input (input content)
    - attributes.response (output content)


### trace
#### inferences + inference.framework

In general, what data we are keeping from in each span
 - general
    - parent_id
 - context
    - span_id
 - attributes
    - (if present) inference.decision.span.id (this appears limited by engine --> present in langgraph, but not in adk)
    - (if present) entity.3.name (aka agent_name)
    - span.type
    - span.subtype
 - events
    - name (both data.input and data.output)
    - attributes.input (input content)
    - attributes.response (output content)

#### agentic invocation + inferences

#### agentic invocation + tool + inferences
for agentic.invocation
 - general
    - parent_id
 - context
    - span_id
 - attributes
    - (if present) inference.decision.span.id (this appears limited by engine --> present in langgraph, but not in adk)
    - (if present) entity.1.name (aka agent_name)
    - (if present) entity.1.from_agent (aka parent_agent)
    - (if present) entity.1.from_agent_span_id (aka id of parent agent)
    - span.type
    - span.subtype
 - events
    - name (both data.input and data.output)
    - attributes.input (input content)
    - attributes.response (output content)



for agentic.tool.invocation
 - general
    - parent_id
 - context
    - span_id
 - attributes
    - (if present) inference.decision.span.id (this appears limited by engine --> present in langgraph, but not in adk)
    - (if present) entity.1.name (aka tool_name)
    - (if present) entity.2.name (aka agent_name)
    - span.type
    - span.subtype
 - events
    - name (both data.input and data.output)
    - attributes.input (input content)
    - attributes.response (output content)

### agentic invocation
#### inferences + inference.framework

### inferences 
#### single inference at a time
what data are we collecting only from an inference span?


## Multi-turn

### agentic session


### agentic turn




## Representative trace samples — `scope.git.run.id: 2026-05-06T19:15:07.352350`

Source: Okahu Stage tenant via `mcp__okahu-stage__get_traces` (apps: `eval_customer_care_agent`, `eval_financial_services_agent`, `eval_lg_travel_agent`), filtered to spans where `scope.git.run.id == "2026-05-06T19:15:07.352350"`.

Retrieval: `get_available_apps_and_workflows` to enumerate the 3 apps → `get_traces` per app for the run's time window (~2026-05-06 19:15–20:05 UTC) → `get_trace_spans` on each trace to confirm `scope.git.run.id` matches.

The run produced 57 traces across 3 agents (cc=20, fs=19, lg=18) clustering into ~12 unique structural signatures (most are a 14-span baseline). The 5 traces below were selected to maximize variety across agent type, span count, span-type vocabulary, sub-agent count, and tool-invocation patterns.

| # | Trace | Agent | Spans | Why it's distinct |
|---|------|-------|------:|-------------------|
| 1 | `50c58519c8437562e6d4ec4878020199` | cc | 32 | **Largest trace.** 4 sub-agents (supervisor → order_lookup → eligibility → refund), 3 distinct tools — deepest multi-agent orchestration in the run. Query: `Process a refund of $350 for order ORD-STD-0350` |
| 2 | `31c3d461f169d6d9eb00cf023c9ab886` | lg | 15 | **Only family with `agentic.mcp.invocation` span type** — MCP-based weather tool call. Unique span vocabulary. Query: `What is the weather in Paris, Texas?` |
| 3 | `ceb2b7186cc7ea8e5f548cc124563c0d` | fs | 17 | **Repeated-tool pattern**: same tool (`get_stock_info`) invoked twice in one trace — different from all others which use distinct tools. Query: `What sector is INTC in?` |
| 4 | `82572f0dbf6be118fe2541b4d3dddfe9` | cc | 16 | **High tool-to-agent ratio**: only 2 sub-agents but 3 distinct tools — a single sub-agent calls multiple tools sequentially (vs. multi-agent fan-out). Query: `I need details on order ORD-A5509` |
| 5 | `2841a214c3fd8586f3a829134963fee4` | fs | 20 | **Reasoning-heavy with single tool**: 3 sub-agents (supervisor → suitability → trade_execution) with 6 inferences but only 1 tool call — most reasoning per action. Query: `Buy 5 shares of BRK.A` |

Together these cover all 3 agents, sizes 15 → 32 spans, the MCP-vs-direct-tool split, repeated-tool reuse, single-agent-multi-tool, and multi-agent-single-tool patterns.

