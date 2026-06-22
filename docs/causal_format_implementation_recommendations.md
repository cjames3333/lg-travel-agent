# Causal Structure Format — Implementation Recommendations

**Date:** 2026-05-05  
**Scope:** BigQuery `okahu-prod.okahu_templates.eval_templates` and `/Users/careyjames/Documents/GitHub/observability/azure-fn`  
**Status:** Recommendations only — no code changes made  
**Reference:** `docs/judge_input_format_causal_structure.md`

---

## Background

The current preamble+CSV format (`id, parent_id, content`) sends a flat conversation log to the judge. Analysis of the claude-sonnet-4-6 evaluation run showed that 28/43 hallucinations were missed because the judge evaluated final→sub-agent relay fidelity instead of sub-agent→tool output grounding. The fix requires the judge to receive a causally structured format (`turn_id, seq, parent_seq, span_type, agent, content`) where `span_type` explicitly labels the reality boundary.

The causal format is controlled by a new `conversation_format` field in the template. The shared preamble (format documentation) lives in code and is selected by `conversation_format`. The evaluation contract (what to look for) lives in the template's `eval_prompt`. These are separate concerns and must remain separate.

---

## 1. BigQuery — `okahu-prod.okahu_templates.eval_templates`

### Current schema (from `templates.py` MERGE statement)

| Column | BQ Type | Description |
|---|---|---|
| `id` | STRING | `evaluation__{group_by}__{name}` |
| `name` | STRING | Template name |
| `version` | INT64 | Unix timestamp — versioning key |
| `type` | STRING | "evaluation" |
| `group_by` | STRING | Aggregation level: traces, inferences, etc. |
| `input` | JSON | Extraction config: `{prompts: [{span_types, events}]}` |
| `output` | JSON | Output table config |
| `llm` | JSON | LLM config: `{prompt_template_name, model_name}` |
| `created_at` | TIMESTAMP | |
| `prompt_schema` | JSON | `structure_output` schema from template JSON file |

### Recommended changes

#### 1a. Add `conversation_format` column

```sql
ALTER TABLE `okahu-prod.okahu_templates.eval_templates`
ADD COLUMN conversation_format STRING OPTIONS (description='Conversation structuring mode: flat_csv (default) or causal_csv');
```

- **Default:** `NULL` treated as `"flat_csv"` — fully backward compatible, all existing templates unaffected
- **New value:** `"causal_csv"` — selects the causal structurer and preamble
- Stored as STRING (not JSON) because it is a scalar enum, not a configuration object

#### 1b. Expand `input.prompts[].span_types` for causal templates

The current hallucination template `input` only targets `inference.framework` spans. The causal format requires additional span types to reconstruct the full agent→tool→agent causal chain.

For any template using `"causal_csv"`, the `input.prompts[].span_types` array must include:

```json
[
  "inference.framework",
  "agentic.tool.invocation",
  "agentic.invocation",
  "agentic.turn"
]
```

These are already known to the shared query infrastructure (`GENAI_SPAN_TYPES` and `AGENT_SPAN_TYPES` in `common/shared_queries.py`) so no new span type registration is needed.

#### 1c. Add `span_subtype` and `agent` to event extraction config

The causal structurer needs two span-level attributes that are currently not extracted: `span.subtype` (maps to span_type enum) and `entity.1.name` (agent/tool name). These live in span `attributes`, not in `data.input`/`data.output` events.

The `input.prompts[].events` schema currently only supports `data.input` and `data.output` event names. A new event entry type is needed:

```json
{
  "name": "attributes",
  "field": "span_subtype",
  "source": "$.\"span.subtype\""
}
```

Alternatively, if changing the events schema is too invasive, add a separate `attributes` key at the prompt level:

```json
{
  "span_types": ["inference.framework", "agentic.tool.invocation", ...],
  "events": [...],
  "attributes": ["span.subtype", "entity.1.name", "entity.1.from_agent"]
}
```

The `attributes` list tells the extraction query which span-level JSON attributes to pull alongside events. The recommended approach is the separate `attributes` key — it does not break the existing `events` schema and `validate_template_data` can treat it as optional.

#### 1d. `group_by` values for new evaluation granularities

The `group_by` column controls what unit of data is evaluated. Causal format adds two new meaningful granularities:

| `group_by` value | Evaluation unit | Existing? |
|---|---|---|
| `traces` | Single trace (1+ spans) | Yes |
| `inferences` | Single inference span | Yes |
| `agent_requests` | Agentic turn (1+ traces) | Partially — maps to `agentic.turn` span |
| `agent_sessions` | Agentic session (1+ turns) | **New** — groups by `scope.agentic.session` attribute |

`agent_sessions` is the only new `group_by` value. It requires the extraction query to group spans by `scope.agentic.session` rather than by `fact_id` (trace ID). Existing `group_by` values are unchanged.

---

## 2. Template JSON Files — `eval-api/core/templates/`

### 2a. New top-level field: `conversation_format`

Add to every template that should use causal structuring. Templates that do not include this field default to `"flat_csv"` (current behavior).

```json
{
  "name": "hallucination",
  "conversation_format": "causal_csv",
  ...
}
```

### 2b. Updated `hallucination.json`

Three changes to the existing file:

**Add `conversation_format`:**
```json
"conversation_format": "causal_csv"
```

**Update `input.prompts` to capture all causal layers:**
```json
"input": {
  "prompts": [
    {
      "span_types": [
        "inference.framework",
        "agentic.tool.invocation",
        "agentic.invocation",
        "agentic.turn"
      ],
      "events": [
        {"name": "data.input",  "field": "input",    "roles": ["user", "system", "tool"]},
        {"name": "data.output", "field": "response",  "roles": ["assistant", "ai"]}
      ],
      "attributes": ["span.subtype", "entity.1.name", "entity.1.from_agent"]
    }
  ]
}
```

**Update `eval_prompt` to include the hallucination-specific causal contract:**

The current `eval_prompt` is generic ("check claims against provided context"). The updated version adds the evaluation contract that operationalizes the causal structure:

```
Evaluate the AI assistant's response for hallucination by checking claims against 
the provided context.

The conversation is provided in causal CSV format with columns: turn_id, seq, 
parent_seq, span_type, agent, content.

Evaluation contract:
- tool_return rows are the authoritative source of truth — they represent exactly 
  what an external system returned, unmediated by any agent.
- Evaluate every claim in agent_claim and final_response rows against all ancestor 
  tool_return rows (trace ancestry via parent_seq).
- A claim that introduces information not present in any ancestor tool_return row 
  is a hallucination, regardless of which layer in the agent chain produced it.
- If a system_prompt row instructs an agent to fabricate or infer missing data 
  (e.g., "fill in missing confirmation details"), flag this as a hallucination 
  risk factor in your explanation even if the agent did not act on it.
- The severity of a hallucination is determined by the impact of the fabricated 
  claim (a wrong financial identifier is major regardless of which agent introduced it), 
  not by how faithfully the supervisor relayed the sub-agent's output.

Identify statements that are not supported by tool_return ancestors or that 
contradict them, distinguishing between factual and hallucinated content.
```

### 2c. New preamble JSON file: `preamble_causal_csv.json`

This file stores the shared format documentation preamble — the "how to read this data" instruction that is format-level, not evaluation-type-specific. It is referenced by code (not by templates directly) and should live alongside the template files for discoverability.

```json
{
  "name": "preamble_causal_csv",
  "version": "v1",
  "description": "Shared preamble for causal CSV conversation format. Format documentation only — no evaluation contract.",
  "preamble": "You are a helpful assistant that analyzes multi-agent conversation traces.\n\nThe conversation is provided as a CSV where each row represents one causal event in the application's execution.\n\nColumn definitions:\n- turn_id: Which user request turn this event belongs to (t1, t2, ...)\n- seq: Chronological sequence number within the turn\n- parent_seq: The seq of the row whose output was the input context when this row was produced\n- span_type: The semantic role of this row (see types below)\n- agent: Who produced this row (agent name, tool name, or 'user')\n- content: The literal data — args, raw return value, or text\n\nspan_type values:\n- user_request: What the user asked\n- system_prompt: The governing instructions active for the agent making the next move\n- routing_decision: Which sub-agent was chosen\n- tool_call: Tool name and arguments — the INPUT side of the reality boundary\n- tool_return: Raw tool output — the OUTPUT side of the reality boundary. This is what an external system literally returned, unmediated by any agent.\n- agent_claim: What a sub-agent said after seeing the tool output\n- final_response: What the user received\n\nHere is the causal CSV:\n"
}
```

This file is loaded by the `CausalConversationStructurer` (see code changes below) at runtime. It is NOT stored in BigQuery — the preamble is shared and format-level, so it belongs in code/config, not per-template data.

---

## 3. Code Changes — `eval-api/`

### 3a. `eval-api/core/okahu_evals/pipeline/common/llm/conversation.py`

**Add `CAUSAL_FORMAT_PROMPT` constant** — the shared preamble loaded from `preamble_causal_csv.json`, or defined inline. Parallel to the existing `CONV_FORMAT_PROMPT`.

**Add `CausalConversationStructurer` class** — a new class alongside `ConversationStructurer` with the same interface (`structure_conversation(conversation) -> str`) but producing `turn_id, seq, parent_seq, span_type, agent, content` output instead of `id, parent_id, content`.

The `CausalConversationStructurer` must:
1. Accept spans enriched with `span.subtype` and `entity.1.name` attributes (provided via the updated extraction pipeline)
2. Map `span.subtype` values to the `span_type` enum:
   - `"delegation"` → `routing_decision`
   - `"tool_call"` → `tool_call` (args) + `tool_return` (from next span's message history)
   - `"turn_end"` (sub-agent) → `agent_claim`
   - `"turn_end"` (supervisor / root agent) → `final_response`
   - System message in input → `system_prompt`
   - User message at turn root → `user_request`
3. Extract `entity.1.name` for the `agent` column
4. Set `parent_seq` by tracking causal ancestry through the span tree (using `parent_id` span attribute)
5. Set `turn_id` from `scope.agentic.turn` attribute when evaluating at session granularity; default `t1` at turn granularity
6. Prepend `CAUSAL_FORMAT_PROMPT` to the generated CSV

The `tool_return` extraction is the trickiest part: in the Monocle traces, tool output does not appear in `agentic.tool.invocation.data.output` (that field is empty). It appears in the subsequent `inference.framework[turn_end].data.input` message history as the `{"tool": "..."}` entry immediately following the `{"ai": "[tool_call...]"}` entry. The `CausalConversationStructurer` must find and promote this entry to a dedicated `tool_return` row.

### 3b. `eval-api/core/okahu_evals/pipeline/common/engine_bq_ray.py`

**In `EvalPipeline.__init__`**, read `conversation_format` from the template config and instantiate the appropriate structurer:

```
conversation_format = self.pipeline.get("conversation_format", "flat_csv")
if conversation_format == "causal_csv":
    self.conversation_structurer = CausalConversationStructurer(desired_roles=self.desired_roles)
else:
    self.conversation_structurer = ConversationStructurer(desired_roles=self.desired_roles)
```

No changes to the extract/load pipeline — only the transform step (conversation structuring) changes.

### 3c. `eval-api/core/okahu_evals/pipeline/common/llm/processor.py`

No interface changes needed. The `LLMBatchProcessor` calls `self.conversation_structurer.structure_conversation(prompt)` — this call is the same regardless of which structurer is used. The output string changes but the call site does not.

### 3d. `eval-api/templates.py` — `validate_template_data`

**Add `conversation_format` to validated fields:**

- Accept `"flat_csv"` and `"causal_csv"` as valid values
- Default to `"flat_csv"` when absent (backward compatible)
- Persist to BigQuery via the existing MERGE/INSERT queries by adding `conversation_format` to the parameter list and the SQL

The MERGE query currently writes: `id, name, version, type, group_by, input, output, llm, created_at, prompt_schema`.  
Updated: add `conversation_format` to this list.

**Add `attributes` to prompt validation** (optional field in `input.prompts`):

Currently `validate_template_data` requires each prompt to have `span_types` and `events`. Add handling for an optional `attributes` list (list of JSON path strings). No maximum length constraint is needed since attribute paths are short.

### 3e. `eval-api/core/okahu_evals/pipeline/common/configuration/parser.py`

**Add `get_required_attributes` method to `TemplateInputParser`:**

Returns the list of span attribute paths needed by the causal structurer (e.g., `["span.subtype", "entity.1.name", "entity.1.from_agent"]`). These are read from `input.prompts[].attributes` when present.

This method is called by `EvalPipeline` to pass the attribute list down to the extraction query builder.

### 3f. `eval-api/core/okahu_evals/pipeline/common/sql_builder.py` (or equivalent BQ query)

The current extraction query selects `data.input` and `data.output` events from spans. For causal format, it must also extract span-level attributes alongside event data.

Specifically, for each span row, the query needs to expose:
- `JSON_VALUE(attributes, '$."span.subtype"')` as `span_subtype`
- `JSON_VALUE(attributes, '$."entity.1.name"')` as `agent_name`
- `JSON_VALUE(attributes, '$."entity.1.from_agent"')` as `from_agent`
- The existing `parent_id` span field (may already be present in the traces table)
- `JSON_VALUE(attributes, '$."scope.agentic.turn"')` as `agentic_turn_id` (for session-level grouping)

The extraction query should conditionally include these columns when `conversation_format = "causal_csv"`. The simplest approach is a separate query template for causal format, parallel to the existing `interactive_eval_query` in `shared_queries.py`.

---

## 4. Code Changes — `okahu-mcp/`

### 4a. `okahu-mcp/tools.py` — `execute_eval_from_json`

The current implementation loads the template from `/okahu-mcp/templates/{template_name}.json`, builds a prompt combining `eval_prompt + structure_output + json_input`, and calls `ctx.sample(prompt)`.

For causal format, the `json_input` structure changes. Currently callers pass unstructured conversation data. With causal format, callers should pass pre-structured causal CSV (or raw spans that the tool can structure). Two options:

**Option A (recommended — simpler):** Accept causal CSV directly as a string field in `json_input`:
```json
{
  "conversation": "<causal CSV string>",
  "format": "causal_csv"
}
```
The tool detects `format: "causal_csv"`, prepends `CAUSAL_FORMAT_PROMPT` instead of the flat preamble, and sends to the judge.

**Option B:** Accept raw spans as JSON and structure them server-side. More powerful but requires shipping the `CausalConversationStructurer` into the MCP server, which currently has no dependency on the eval pipeline.

Option A preserves the MCP tool's current architecture and keeps the structuring logic in the eval pipeline.

### 4b. `/okahu-mcp/templates/hallucination.json`

Update the local MCP copy of the hallucination template to include:
- `"conversation_format": "causal_csv"`
- Updated `eval_prompt` with the hallucination causal contract (same text as section 2b above)

Note: The MCP server loads templates from its own local `/okahu-mcp/templates/` directory, not from the eval-api templates or BigQuery. Both copies must be updated independently.

---

## 5. Change Summary by Layer

| Layer | What changes | Files affected |
|---|---|---|
| **BigQuery schema** | Add `conversation_format` STRING column | `eval_templates` table DDL |
| **BigQuery extraction query** | Conditionally extract span attributes (`span.subtype`, `entity.1.name`, `entity.1.from_agent`, `scope.agentic.turn`) | `common/shared_queries.py` |
| **Template JSON (eval-api)** | Add `conversation_format`, expand `span_types`, update `eval_prompt`, add `attributes` | `eval-api/core/templates/hallucination.json` |
| **New preamble file** | Shared causal format documentation | `eval-api/core/templates/preamble_causal_csv.json` |
| **Conversation structurer** | Add `CAUSAL_FORMAT_PROMPT` and `CausalConversationStructurer` | `conversation.py` |
| **Eval pipeline** | Instantiate correct structurer based on `conversation_format` | `engine_bq_ray.py` |
| **Template validation** | Accept and persist `conversation_format`, accept optional `attributes` | `templates.py`, `parser.py` |
| **MCP server template** | Update local hallucination.json copy | `okahu-mcp/templates/hallucination.json` |
| **MCP tool** | Handle `format: causal_csv` in `json_input`, select correct preamble | `okahu-mcp/tools.py` |

---

## 6. Backward Compatibility

All changes are additive:

- `conversation_format` absent or `null` → existing `ConversationStructurer` and `CONV_FORMAT_PROMPT` are used unchanged
- `attributes` absent from `input.prompts` → no attribute extraction, existing behavior
- Existing templates in BigQuery require no migration — they will continue to use `"flat_csv"` by default
- The `group_by: "agent_sessions"` value is new but is only activated by templates that explicitly set it

The only breaking change risk is if the BigQuery extraction query for causal format returns a different row structure than existing code expects. This is mitigated by making the causal query a separate code path, not a modification of the existing query.

---

## 7. Key Design Decisions

**Why `conversation_format` is a top-level template field, not nested in `input` or `llm`**  
It controls how extracted data is formatted before reaching the judge — it sits between the extraction step (`input`) and the LLM step (`llm`). Nesting it in either would misrepresent its role in the pipeline.

**Why the preamble is in code, not in BigQuery**  
The preamble is shared format documentation — it is identical for every template using `"causal_csv"`. Storing it in BigQuery per-template would duplicate it across rows and create drift risk. Code is the right home for shared constants. The evaluation contract (which varies per template) belongs in `eval_prompt` in the template.

**Why `tool_return` extraction requires special handling**  
Monocle's `agentic.tool.invocation` span has an empty `data.output`. The tool's actual return value is only visible in the next `inference.framework` span's `data.input` message history as a `{"tool": "..."}` entry. The `CausalConversationStructurer` must detect and promote this entry rather than relying on the tool span's own output event.

**Why `agent_sessions` needs a new extraction query grouping**  
Current extraction groups spans by `fact_id` (trace ID). A session spans multiple trace IDs, all sharing the same `scope.agentic.session` attribute value. The query must switch its GROUP BY key for session-level evaluation.
