# Trace visual → field mapping

This documents which fields in the raw trace JSON map to each element of the rendered Okahu trace visualization. Every element on the diagram comes from one of three places: the span's top-level fields, `attributes.*`, or the `events[].attributes` payload. The hierarchy/indentation comes from `parent_id`.

Three reference traces are mapped below — one ADK / Gemini, one LangGraph / OpenAI, and one LangGraph + MCP — to show how the **same visual schema is filled in from engine-specific span vocabularies**.

- [Trace 1 — ADK / Gemini (`okahu_demos_adk_travel_agent_tests`)](#trace-1--adk--gemini)
- [Trace 2 — LangGraph / OpenAI (`test_cc_customer_care_agent`)](#trace-2--langgraph--openai)
- [Trace 3 — LangGraph + MCP tool (`test_lg_travel_agent`)](#trace-3--langgraph-with-mcp-tool)
- [Cross-trace differences](#cross-trace-differences-adk-vs-langgraph)

---

## Trace 1 — ADK / Gemini

Source trace: `c7dcd4c1c338c21075793e3003dc872c` (workflow `okahu_demos_adk_travel_agent_tests`), retrieved via `mcp__okahu-stage__get_trace_spans`.

## Visual reference (the diagram being mapped)

```
workflow: okahu_demos_adk_travel_agent_tests
  agentic.turn / turn
    agentic.invocation / routing
      adk_supervisor_agent
        agentic.invocation / content_processing
          adk_flight_booking_agent
            inference
            agentic.tool.invocation / content_generation
              adk_flight_booking_agent → adk_book_flight
            inference
        agentic.invocation / content_processing
          adk_hotel_booking_agent
            inference
            agentic.tool.invocation / content_generation
              adk_hotel_booking_agent → adk_book_hotel
            inference
        agentic.invocation / content_processing
          adk_trip_summary_agent
            inference
```

## Per-element source

| Visual element | Source field | Example value |
|---|---|---|
| `workflow` (top label) | `attributes.span.type` | `"workflow"` |
| `okahu_demos_adk_travel_agent_tests` | `attributes.workflow.name` (also `attributes.entity[0].name`) | `okahu_demos_adk_travel_agent_tests` |
| `agentic.turn` | `attributes.span.type` | `"agentic.turn"` |
| `turn` (next to `agentic.turn`) | `attributes.span.subtype` | `"turn"` |
| `agentic.invocation` | `attributes.span.type` | `"agentic.invocation"` |
| `routing` / `content_processing` | `attributes.span.subtype` | `"routing"`, `"content_processing"` |
| `adk_supervisor_agent` / `adk_flight_booking_agent` / `adk_hotel_booking_agent` / `adk_trip_summary_agent` | `attributes.entity[0].name` (where `entity[0].type = "agent.adk"`) | e.g. `"adk_supervisor_agent"` |
| `inference` | `attributes.span.type` | `"inference"` |
| `agentic.tool.invocation` | `attributes.span.type` | `"agentic.tool.invocation"` |
| `content_generation` (under tool invocation) | `attributes.span.subtype` | `"content_generation"` |
| `adk_book_flight` / `adk_book_hotel` (right-side label) | The tool `entity[].name` whose `type = "tool.adk"` | `"adk_book_flight"`, `"adk_book_hotel"` |
| Calling agent next to the tool (`adk_flight_booking_agent` / `adk_hotel_booking_agent`) | The agent `entity[].name` (type `agent.adk`) on the same tool span — also matches the parent invocation's `entity[0].name` | `"adk_flight_booking_agent"`, `"adk_hotel_booking_agent"` |
| Tree indentation / nesting | `parent_id` chain (workflow → turn → supervisor → {flight, hotel, summary} → inference/tool) | — |

### Identifiers and payloads (used to wire the tree and feed the detail panel)

| Element | Source field | Notes |
|---|---|---|
| Span identifier | top-level `span_id` (also `context.span_id` in raw OTEL form) | Anchors `parent_id` references |
| Parent link | top-level `parent_id` | Drives nesting; root `workflow` span has `parent_id = null` |
| User / upstream input shown in the detail panel | `events[].name = "data.input"` → `events[].attributes.input` (array of role-tagged JSON strings) | Present on `agentic.turn`, `agentic.invocation` (sub-agents only — supervisor `routing` carries no events), `agentic.tool.invocation`, and `inference` |
| Agent / model / tool output shown in the detail panel | `events[].name = "data.output"` → `events[].attributes.response` | Same span coverage as `data.input`. On tool spans this is the JSON return value of the function |

### Inferring an `inference`-span subtype on ADK

Unlike LangGraph (`inference.framework` carries `span.subtype` of `delegation` / `tool_call` / `turn_end`), ADK `inference` spans **do not have `span.subtype`**. The same role can be inferred from these fields on the inference span:

| Inferred role | Indicator on the ADK `inference` span | Example |
|---|---|---|
| Tool-call inference (LG-equivalent: `tool_call`) | `attributes.entity[]` contains an entry with `type = "tool.function"` (e.g. `adk_book_flight`) **and** `events[name="metadata"].attributes.finish_reason = "FUNCTION_CALL"` (with `finish_type = "tool_call"`) | flight-booking agent's first inference, hotel-booking agent's first inference |
| Final / closing inference (LG-equivalent: `turn_end`) | No `tool.function` entity present **and** `events[name="metadata"].attributes.finish_reason = "STOP"` (with `finish_type = "success"`) | flight-booking and hotel-booking agents' second inferences; trip-summary agent's only inference |
| Delegation inference (LG-equivalent: `delegation`) | **Does not occur in ADK with this orchestration.** ADK's supervisor uses `SequentialAgent`, not a tool-call to hand off, so there is no inference at the supervisor level. The supervisor's `agentic.invocation` carries `span.subtype = "routing"` and has no inference children. | — |

These same `entity[]` / `metadata.finish_reason` / `metadata.finish_type` fields exist on LangGraph's `inference.framework` spans and would yield the same answer — but on LangGraph the explicit `span.subtype` already encodes it, so the inference is redundant there.

## Fields shown on the diagram, summarized

- `attributes.span.type` — top label of every row
- `attributes.span.subtype` — small-text qualifier (`turn`, `routing`, `content_processing`, `content_generation`)
- `attributes.entity[]` — agent name, tool name (selected by `type` = `agent.adk` / `tool.adk`)
- `attributes.workflow.name` — workflow header
- `parent_id` — the indent/nesting structure

## Fields not rendered in the tree

The visual tree omits everything else returned by the API, notably:

- `trace_id`
- `start_time` / `end_time` and `duration_ms`
- `events[name="metadata"]` payloads — `total_tokens`, `prompt_tokens`, `completion_tokens`, `thoughts_tokens`, `finish_reason`, `finish_type` (these are used for the *inferred subtype* on ADK — see above — but not shown as visible chips after the cleanup)
- The model `entity` (e.g. `gemini-2.5-flash`) and inference provider entity (`inference_endpoint`, `provider_name`)
- `scope.*` (session, turn, git, test_name)
- `span_source`
- per-tool `description` text

(Note: `span_id`, `parent_id`, and the `data.input` / `data.output` events *are* used — `span_id` / `parent_id` to wire the tree itself, and `data.input` / `data.output` to populate the per-span detail panel — see the "Identifiers and payloads" table above.)

---

## Trace 2 — LangGraph / OpenAI

Source trace: `621d3a6cc6fe83fc3684c7f8059b9e87` (workflow `test_cc_customer_care_agent`), retrieved via `mcp__okahu-stage__get_trace_spans`. 24 spans total in the JSON; only 17 are rendered in the visual (see the "split" note below).

### Visual reference (the diagram being mapped)

```
workflow: test_cc_customer_care_agent
  agentic.turn / turn
    agentic.invocation / content_processing
      okahu_demo_cc_agent_supervisor
        inference.framework / delegation
    agentic.invocation / content_processing
      okahu_demo_cc_agent_order_lookup
        inference.framework / tool_call
        agentic.tool.invocation / content_generation
          okahu_demo_cc_agent_order_lookup → okahu_demo_cc_tool_lookup_order
        agentic.tool.invocation / content_generation
          okahu_demo_cc_agent_order_lookup → okahu_demo_cc_tool_get_shipping_status
        inference.framework / turn_end
    agentic.invocation / content_processing
      okahu_demo_cc_agent_supervisor
        inference.framework / delegation
    agentic.invocation / content_processing
      okahu_demo_cc_agent_eligibility
        inference.framework / tool_call
        agentic.tool.invocation / content_generation
          okahu_demo_cc_agent_eligibility → okahu_demo_cc_tool_check_eligibility
        inference.framework / turn_end
    agentic.invocation / content_processing
      okahu_demo_cc_agent_supervisor
        inference.framework / turn_end
```

### Per-element source

| Visual element | Source field | Example value |
|---|---|---|
| `workflow` (top label) | `attributes.span.type` | `"workflow"` |
| `test_cc_customer_care_agent` | `attributes.workflow.name` (also `attributes.entity[0].name`, where `entity[0].type = "workflow.langgraph"`) | `test_cc_customer_care_agent` |
| `agentic.turn` | `attributes.span.type` | `"agentic.turn"` |
| `turn` | `attributes.span.subtype` | `"turn"` |
| `agentic.invocation` | `attributes.span.type` | `"agentic.invocation"` |
| `content_processing` | `attributes.span.subtype` | `"content_processing"` |
| `okahu_demo_cc_agent_supervisor` / `okahu_demo_cc_agent_order_lookup` / `okahu_demo_cc_agent_eligibility` | `attributes.entity[0].name` (where `entity[0].type = "agent.langgraph"`) | e.g. `"okahu_demo_cc_agent_supervisor"` |
| `inference.framework` | `attributes.span.type` | `"inference.framework"` |
| `delegation` / `tool_call` / `turn_end` (inference subtype chip) | `attributes.span.subtype` | `"delegation"`, `"tool_call"`, `"turn_end"` |
| `agentic.tool.invocation` | `attributes.span.type` | `"agentic.tool.invocation"` |
| `content_generation` | `attributes.span.subtype` | `"content_generation"` |
| Tool name on the right (`okahu_demo_cc_tool_lookup_order`, `okahu_demo_cc_tool_get_shipping_status`, `okahu_demo_cc_tool_check_eligibility`) | The tool `entity[].name` whose `type = "tool.langgraph"` (here it is `entity[0]`) | `"okahu_demo_cc_tool_lookup_order"` |
| Calling agent on the left (`okahu_demo_cc_agent_order_lookup`, `okahu_demo_cc_agent_eligibility`) | The agent `entity[].name` (type `agent.langgraph`) on the same tool span (here it is `entity[1]`) | `"okahu_demo_cc_agent_order_lookup"` |
| Tree indentation / nesting | `parent_id` chain (workflow → turn → invocation → {inference.framework, tool.invocation}) | — |

### Identifiers and payloads (used to wire the tree and feed the detail panel)

| Element | Source field | Notes |
|---|---|---|
| Span identifier | top-level `span_id` | Anchors `parent_id` references and is the target of `inference.decision.span.id` |
| Parent link | top-level `parent_id` | Drives nesting; root `workflow` span has `parent_id = null` |
| Decision pointer (LangGraph-only) | `attributes["inference.decision.span.id"]` on `agentic.invocation` and `agentic.tool.invocation` spans | Points back to the `inference.framework` span whose tool-call decision caused this invocation/tool call. Lets the renderer link a sub-agent's `agentic.invocation` to the supervisor's `delegation` inference, and a tool span to its triggering `tool_call` inference. **Not present in the ADK trace.** |
| User / upstream input shown in the detail panel | `events[].name = "data.input"` → `events[].attributes.input` (array of role-tagged JSON strings) | Present on `agentic.turn`, `agentic.invocation`, `agentic.tool.invocation`, and `inference.framework`. Absent on `workflow` and `inference.modelapi`. |
| Agent / model / tool output shown in the detail panel | `events[].name = "data.output"` → `events[].attributes.response` | Same span coverage as `data.input`. On tool spans this is the JSON return value of the function. |

#### Decision-pointer examples from this trace

| Span (`span_id`) | Span type / subtype | `inference.decision.span.id` → | Meaning |
|---|---|---|---|
| `3ed3cf228413fd0c` | `agentic.invocation` / `content_processing` (`okahu_demo_cc_agent_order_lookup`) | `6cc4d7c949750c26` (supervisor's `delegation` inference) | Order-lookup agent ran because the supervisor's delegation inference asked for it |
| `788cce506433e5b1` | `agentic.tool.invocation` (`okahu_demo_cc_tool_lookup_order`) | `b1c1842c6fcf559b` (order-lookup agent's `tool_call` inference) | The lookup-order tool was called because of that tool_call inference |
| `1b771ee1dc0b6789` | `agentic.tool.invocation` (`okahu_demo_cc_tool_check_eligibility`) | `f49b21698f1c4fea` (eligibility agent's `tool_call` inference) | Same pattern, eligibility branch |
| `ed800e853bbf8af2` | `agentic.invocation` / `content_processing` (final supervisor turn) | `ddb72841805edde8` (eligibility agent's `turn_end` inference) | Supervisor was re-entered after eligibility returned |

### LangGraph-specific notes

- **Two-layer inference**. Each rendered `inference.framework` row in the JSON has a child `inference.modelapi` span (the raw HTTP-level call). The visual collapses the two and shows only the framework layer. Of the 24 spans returned, the 7 `inference.modelapi` spans are *not* rendered.
- **Handoffs surface as `transfer_to_*` tool functions** in the `entity[2]` slot of the supervisor's `delegation` inferences (e.g. `transfer_to_okahu_demo_cc_agent_order_lookup`, `transfer_to_okahu_demo_cc_agent_eligibility`). The visual hides these.
- **Inference subtypes are meaningful**: `delegation` = supervisor handing off to a sub-agent; `tool_call` = sub-agent invoking a tool; `turn_end` = final response that closes the agent's turn.
- **Entity ordering on `agentic.tool.invocation` is reversed vs the ADK trace.** Here `entity[0]` is the **tool** (`tool.langgraph`) and `entity[1]` is the **agent** (`agent.langgraph`). The renderer therefore selects by `type`, not by index.

---

## Trace 3 — LangGraph with MCP tool

Source trace: `31c3d461f169d6d9eb00cf023c9ab886` (workflow `test_lg_travel_agent`), retrieved via `mcp__okahu-stage__get_trace_spans`. 15 spans total in the JSON; 12 are rendered (3 `inference.modelapi` children hidden, same as Trace 2). This trace introduces the **`agentic.mcp.invocation`** span type — a tool call that is delegated to a remote MCP server (here `WeatherServer` at `http://localhost:8007/weather/mcp/`).

### Visual reference (the diagram being mapped)

```
workflow: test_lg_travel_agent
  agentic.turn / turn
    agentic.invocation / content_processing
      okahu_demo_lg_agent_travel_supervisor
        inference.framework / delegation
    agentic.invocation / content_processing
      okahu_demo_lg_agent_weather_assistant
        inference.framework / tool_call
        agentic.tool.invocation / content_generation
          okahu_demo_lg_agent_weather_assistant → demo_get_weather
          agentic.mcp.invocation / routing
            WeatherServer → demo_get_weather
        inference.framework / turn_end
    agentic.invocation / content_processing
      okahu_demo_lg_agent_travel_supervisor
        inference.framework / turn_end
```

### Per-element source

| Visual element | Source field | Example value |
|---|---|---|
| `workflow` (top label) | `attributes.span.type` | `"workflow"` |
| `test_lg_travel_agent` | `attributes.workflow.name` (also `attributes.entity[0].name`, where `entity[0].type = "workflow.langgraph"`) | `test_lg_travel_agent` |
| `agentic.turn` | `attributes.span.type` | `"agentic.turn"` |
| `turn` | `attributes.span.subtype` | `"turn"` |
| `agentic.invocation` | `attributes.span.type` | `"agentic.invocation"` |
| `content_processing` | `attributes.span.subtype` | `"content_processing"` |
| `okahu_demo_lg_agent_travel_supervisor` / `okahu_demo_lg_agent_weather_assistant` | `attributes.entity[0].name` (where `entity[0].type = "agent.langgraph"`) | e.g. `"okahu_demo_lg_agent_weather_assistant"` |
| `inference.framework` | `attributes.span.type` | `"inference.framework"` |
| `delegation` / `tool_call` / `turn_end` | `attributes.span.subtype` | `"delegation"`, `"tool_call"`, `"turn_end"` |
| `agentic.tool.invocation` | `attributes.span.type` | `"agentic.tool.invocation"` |
| `content_generation` | `attributes.span.subtype` | `"content_generation"` |
| Tool name (`demo_get_weather`) on the right of the tool row | The tool `entity[].name` whose `type = "tool.mcp"` (here `entity[0]`) — note the type is `tool.mcp`, **not** `tool.langgraph`, when the tool is served by an MCP server | `"demo_get_weather"` |
| Calling agent on the left of the tool row (`okahu_demo_lg_agent_weather_assistant`) | The agent `entity[].name` whose `type = "agent.langgraph"` on the same span (here `entity[1]`) | `"okahu_demo_lg_agent_weather_assistant"` |
| **`agentic.mcp.invocation`** (new span type) | `attributes.span.type` | `"agentic.mcp.invocation"` |
| `routing` (under MCP invocation) | `attributes.span.subtype` | `"routing"` |
| MCP server name on the left (`WeatherServer`) | `entity[0].server_name` (where `entity[0].type = "mcp.server"`) | `"WeatherServer"` |
| MCP tool name on the right (`demo_get_weather`) | `entity[0].name` (where `entity[0].type = "mcp.server"`) | `"demo_get_weather"` |
| Tree indentation / nesting | `parent_id` chain (workflow → turn → invocation → {inference.framework, tool.invocation → mcp.invocation}) | — |

### Identifiers and payloads (used to wire the tree and feed the detail panel)

| Element | Source field | Notes |
|---|---|---|
| Span identifier | top-level `span_id` | Anchors `parent_id` references and is the target of `inference.decision.span.id` |
| Parent link | top-level `parent_id` | Drives nesting; root `workflow` span has `parent_id = null` |
| Decision pointer | `attributes["inference.decision.span.id"]` on `agentic.invocation` and `agentic.tool.invocation` spans | Same mechanism as Trace 2. Notable in this trace: the weather tool's `agentic.tool.invocation` (`span_id = 2b113891fa324322`) carries `inference.decision.span.id = e1136a09defc55e0`, pointing back to the weather assistant's `tool_call` inference. The final supervisor `agentic.invocation` (`d1c91ef4bb66f0dd`) points back to `f366a1a0cf68432b` (weather assistant's `turn_end`). |
| User / upstream input shown in the detail panel | `events[name="data.input"].attributes.input` | Present on `agentic.turn`, `agentic.invocation`, `agentic.tool.invocation`, `inference.framework`, **and `agentic.mcp.invocation`**. Absent on `workflow` and `inference.modelapi`. |
| Agent / model / tool output shown in the detail panel | `events[name="data.output"].attributes.response` | Same span coverage *except* for the MCP span — see next row. |
| **MCP invocation output payload** | `events[name="data.output"].attributes.output` (note: `output`, not `response`) | The `agentic.mcp.invocation` span's `data.output` event uses field name `output`. This is the only span type in these three traces that diverges from the `response` convention. |

#### Decision-pointer examples from this trace

| Span (`span_id`) | Span type / subtype | `inference.decision.span.id` → | Meaning |
|---|---|---|---|
| `ad0be2dceb6491ff` | `agentic.invocation` / `content_processing` (`okahu_demo_lg_agent_weather_assistant`) | `b7f6d82fa1d0c69d` (supervisor's `delegation` inference) | Weather assistant ran because the supervisor's delegation inference invoked `transfer_to_okahu_demo_lg_agent_weather_assistant` |
| `2b113891fa324322` | `agentic.tool.invocation` (`demo_get_weather`) | `e1136a09defc55e0` (weather assistant's `tool_call` inference) | The weather tool was called because of that tool_call inference; this span in turn contains the `agentic.mcp.invocation` |
| `d1c91ef4bb66f0dd` | `agentic.invocation` / `content_processing` (final supervisor turn) | `f366a1a0cf68432b` (weather assistant's `turn_end` inference) | Supervisor was re-entered after the weather assistant finished |

### MCP-specific notes

- **New span type — `agentic.mcp.invocation`** (`span.subtype = "routing"`). Wraps the JSON-RPC `tools/call` request to the MCP server. It is a **child of the parent `agentic.tool.invocation`** span (parent_id chain: tool invocation → MCP invocation), so the MCP row appears nested under the tool row in the tree.
- **`entity[0]` on the MCP span carries `type = "mcp.server"`** plus extra fields: `name` (the MCP-exposed tool name), `server_name` (the human-readable server name), and `url` (the MCP endpoint, e.g. `http://localhost:8007/weather/mcp/`). The visual surfaces `server_name` on the left and `name` on the right.
- **The wrapping `agentic.tool.invocation` flips its tool entity type to `tool.mcp`** (vs `tool.langgraph` for in-process tools, as seen in Trace 2). It also gains a top-level boolean attribute `is_mcp = true`. Either of these can be used by the renderer to mark the row as MCP-backed.
- **`data.output` payload field is `output`, not `response`** for `agentic.mcp.invocation` only. This is the lone exception to the otherwise-uniform `response` convention across the trace family.
- **Spans returned vs rendered**: 15 spans returned, 12 rendered. Same fold-rule as Trace 2 — the 3 `inference.modelapi` children are hidden under their `inference.framework` parents.
- **`from_agent` / `from_agent_span_id` on sub-agent invocations.** The weather assistant's `agentic.invocation` and the final supervisor's `agentic.invocation` both carry `entity[0].from_agent` and `from_agent_span_id` pointing to the previous active agent. This complements the decision pointer with an explicit "who handed control to me" link.

---

## Cross-trace differences (ADK vs LangGraph)

The same visual layout is produced from engine-specific span vocabularies. Where they diverge:

| Aspect | ADK trace | LangGraph trace |
|---|---|---|
| Inference span type | `inference` (single layer) | `inference.framework` + child `inference.modelapi` (two layers; only framework is rendered) |
| Inference subtype | **Not present** — must be inferred from `entity[type="tool.function"]` presence + `metadata.finish_reason` / `finish_type` | Present as `span.subtype` = `delegation` / `tool_call` / `turn_end` |
| `inference.decision.span.id` (back-pointer from invocation/tool spans to the inference that decided the call) | **Not present** | Present on `agentic.invocation` and `agentic.tool.invocation` spans |
| Workflow entity type | `workflow.generic` | `workflow.langgraph` |
| Agent entity type | `agent.adk` | `agent.langgraph` |
| Tool entity type | `tool.adk` | `tool.langgraph` |
| Multi-agent orchestration pattern | Sequential agents (no explicit handoff tool) | `transfer_to_*` tool functions at the supervisor's delegation inference |
| Spans returned vs rendered | 13 spans, all 13 rendered | 24 spans, 17 rendered (7 `inference.modelapi` hidden) |

The renderer's contract is therefore:

1. Use `span.type` for the row label and `span.subtype` for the small qualifier.
2. Pick the agent / tool name by walking `entity[]` and selecting by `type` prefix (`agent.*`, `tool.*`), not by index.
3. Fold `inference.modelapi` children into their `inference.framework` parent and only render the parent.
4. When `span.subtype` is missing on an `inference` span (ADK), derive it: a `tool.function` entry in `entity[]` plus `metadata.finish_reason = "FUNCTION_CALL"` ⇒ `tool_call`-equivalent; otherwise `STOP` ⇒ `turn_end`-equivalent.
5. When `inference.decision.span.id` is present on an invocation/tool span (LangGraph), use it to back-link that span to the inference that triggered it.
6. For `agentic.mcp.invocation` spans (Trace 3): label the row with `entity[0].server_name → entity[0].name` (where `entity[0].type = "mcp.server"`) and read the output payload from `events[name="data.output"].attributes.output` — **not** `.response`. The wrapping `agentic.tool.invocation` can be detected as MCP-backed via either `entity[type="tool.mcp"]` or the top-level `is_mcp = true` attribute.
