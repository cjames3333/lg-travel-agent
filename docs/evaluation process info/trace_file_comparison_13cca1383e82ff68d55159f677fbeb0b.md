# Trace File Comparison
**Trace ID:** `13cca1383e82ff68d55159f677fbeb0b`
**Date:** 2026-05-04

## Files Compared

| | File |
|---|---|
| **A (Okahu)** | `okahu_trace_13cca1383e82ff68d55159f677fbeb0b.json` |
| **B (Monocle)** | `monocle_trace_test_fs_financial_agent_13cca1383e82ff68d55159f677fbeb0b_2026-04-17_10.52.52.json` |

---

## 1. Top-Level Structure

| | Okahu (A) | Monocle (B) |
|---|---|---|
| Root type | Single JSON object | JSON array of span objects |
| Trace-level wrapper | Yes — `trace_id`, `start_time`, `end_time`, `duration_ms`, `status` at root | No — spans are the root elements |
| Span count | 14 | 14 |

**Okahu root:**
```json
{
  "trace_id": "13cca1383e82ff68d55159f677fbeb0b",
  "start_time": "2026-04-17T14:52:46.167806Z",
  "end_time": "2026-04-17T14:52:52.195138Z",
  "duration_ms": 6027,
  "status": { "code": "ok", "message": null },
  "spans": [ ... ]
}
```

**Monocle root:**
```json
[ { span }, { span }, ... ]
```

---

## 2. Span Ordering

| | Okahu (A) | Monocle (B) |
|---|---|---|
| Order | Hierarchical — parent spans first, children follow in execution order | Reverse chronological — spans written as they complete, so children appear before parents |

**Okahu span order (first → last):**
1. `workflow` (root)
2. `agentic.turn`
3. `supervisor` invocation
4. `inference.framework` (delegation)
5. `inference.modelapi`
6. `fund_transfer` invocation
7. `inference.framework` (tool_call)
8. `inference.modelapi`
9. `agentic.tool.invocation`
10. `inference.framework` (turn_end)
11. `inference.modelapi`
12. supervisor return invocation
13. `inference.framework` (turn_end)
14. `inference.modelapi`

**Monocle span order (first → last):**
Reverse — starts with the first-completing `openai.AsyncCompletions` child span, ends with the `workflow` root span.

---

## 3. Span Field Naming

| Field | Okahu (A) | Monocle (B) |
|---|---|---|
| Span name | `span_name` | `name` |
| Span ID location | Direct field: `"span_id": "..."` | Nested: `"context": { "span_id": "..." }` |
| Trace ID location | Direct field: `"trace_id": "..."` | Nested: `"context": { "trace_id": "..." }` |
| Parent ID | Direct field: `"parent_id": "..."` (absent on root span) | Direct field: `"parent_id": "..."` (`null` on root span) |

---

## 4. Status Format

| | Okahu (A) | Monocle (B) |
|---|---|---|
| Format | `{"code": "ok", "message": null}` | `{"status_code": "OK"}` |

---

## 5. Attributes Format

This is the most significant structural difference.

**Okahu** stores entity attributes as a nested array of objects:
```json
"entity": [
  { "name": "gpt-4o", "type": "model.llm.gpt-4o" },
  { "inference_endpoint": "https://api.openai.com/v1/", "provider_name": "api.openai.com", "type": "inference.openai" },
  { "name": "okahu_demo_fs_tool_transfer_funds", "type": "tool.function" }
]
```

**Monocle** stores entity attributes as flat dotted key-value pairs:
```json
"entity.1.type": "inference.openai",
"entity.1.provider_name": "api.openai.com",
"entity.1.inference_endpoint": "https://api.openai.com/v1/",
"entity.2.name": "gpt-4o",
"entity.2.type": "model.llm.gpt-4o",
"entity.3.name": "okahu_demo_fs_tool_transfer_funds",
"entity.3.type": "tool.function"
```

---

## 6. Fields Only in Monocle (B)

| Field | Value | Notes |
|---|---|---|
| `kind` | `"SpanKind.INTERNAL"` | Present on every span |
| `links` | `[]` | Present on every span |
| `resource` | `{"attributes": {"service.name": "test_fs_financial_agent"}, "schema_url": ""}` | Present on every span |
| `context.trace_state` | `"[]"` | Inside the `context` object |

---

## 7. Fields Only in Okahu (A)

| Field | Scope | Notes |
|---|---|---|
| `duration_ms` | Per span | Pre-calculated duration in milliseconds |
| Trace-level `trace_id`, `start_time`, `end_time`, `duration_ms`, `status` | Top-level object | Okahu wraps all spans in a trace envelope |

---

## 8. Identical Fields (Both Files)

The following fields are present and have identical values in both files for every span:

- `start_time`
- `end_time`
- `parent_id`
- `span_id` / `context.span_id`
- `trace_id` / `context.trace_id`
- All `events` (name, timestamp, attributes — input, output, metadata)
- `workflow.name`
- `span.type`
- `span.subtype`
- `scope.agentic.turn`
- `scope.agentic.invocation`
- `monocle_apptrace.version`
- `monocle_apptrace.language`
- `span_source`
- `entity.count`

---

## 9. Summary

| Dimension | Okahu (A) | Monocle (B) |
|---|---|---|
| Purpose | Query/API response format | Raw OpenTelemetry export format |
| Root structure | Trace envelope object | Flat span array |
| Span order | Hierarchical (parent-first) | Emission order (child-first) |
| Entity attributes | Nested array of objects | Flat dotted key-value pairs |
| Status schema | `code` / `message` | `status_code` |
| Duration pre-computed | Yes (`duration_ms` per span) | No (must compute from `start_time`/`end_time`) |
| OTel metadata | Omitted (`kind`, `links`, `resource`) | Included |
| Span context | Flat fields | Nested `context` object |
| Semantic content | Identical — same span IDs, times, events, and attribute values |
