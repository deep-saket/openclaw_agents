# Collection Agent Runtime

This directory stores generated runtime artifacts for `agents/collection_agent`.

## Generation Lifecycle

| Source Component | Files it creates/updates | When |
| --- | --- | --- |
| `CollectionRepository` | `conversation_messages.json`, `conversation_states.json`, `tool_logs.json` | On agent startup and every turn |
| `CollectionDataStore` | `contact_attempts.json`, `verification_attempts.json`, `payment_links.json`, `promises.json`, `escalations.json`, `plan_proposals.json` | On startup (ensures files) and during tool execution |
| `CollectionAgent._persist_trace` | `traces/<timestamp>_<trace_id>.json`, `traces/latest_trace.json` | After each completed turn |
| `JSONLTraceSink` (optional) | `traces/events.jsonl` (or configured path) | Real-time during node/tool/llm events |

## File Reference

### Session and turn state

| File | Type | Purpose |
| --- | --- | --- |
| `conversation_messages.json` | object (`session_id -> [messages]`) | Conversation history by session |
| `conversation_states.json` | object (`session_id -> state`) | Working state (`mode`, `active_case_id`, `current_plan`, etc.) |
| `tool_logs.json` | array | Tool executor audit entries (input/output/status/error/timestamp) |

### Longitudinal memory stores

| File | Type | Purpose |
| --- | --- | --- |
| `memory/global_key_event_memory.json` | object | Cross-user successful/unsuccessful procedural cues + counters |
| `memory/user_key_event_memory.json` | object (`user_id -> memory`) | User-specific summary, procedural key points, follow-up considerations, outcome history |

### Tool runtime artifacts

| File | Type | Written by tools |
| --- | --- | --- |
| `contact_attempts.json` | array | `contact_attempt` |
| `verification_attempts.json` | array | `verify_dob`, `verify_mobile` |
| `payment_links.json` | array | `payment_link_create` |
| `promises.json` | array | `promise_capture` |
| `escalations.json` | array | `human_escalation` |
| `plan_proposals.json` | array | `plan_propose` |

### Trace outputs

| File/Path | Type | Purpose |
| --- | --- | --- |
| `traces/<timestamp>_<trace_id>.json` | JSON object | Full per-turn trace snapshot (node order, tool order, timings, llm calls) |
| `traces/latest_trace.json` | JSON object | Last completed turn trace |
| `traces/events.jsonl` | JSONL | Event stream (`turn_started`, `node_started`, `tool_call`, `llm_call`, etc.) |

### Batch evaluation outputs

| File/Path | Type | Purpose |
| --- | --- | --- |
| `eval_runs/<run_id>/run_summary.csv` | CSV | One summary row per executed dataset script |
| `eval_runs/<run_id>/conversations.jsonl` | JSONL | One preserved dataset+execution record per script |
| `eval_runs/<run_id>/traces/<script_id>.json` | JSON | Per-script execution trace payload with dataset linkage |
| `eval_runs/<run_id>.zip` | ZIP | Portable bundle of the whole batch run |

All batch evaluation artifacts must carry both `run_id` and `script_id`.

## Reading Traversal Order

- Node order: `summary.node_hits` in trace JSON.
- Tool order: `tool_calls[].tool_name` in trace JSON.
- Real-time event order: line order in `events.jsonl`.

## Reset for Clean Demo

To reset runtime files before a demo:

- Object files: set to `{}`
  - `conversation_messages.json`
  - `conversation_states.json`
- Array files: set to `[]`
  - all other top-level JSON files in this directory
- Trace files: optional cleanup
  - remove files inside `runtime/traces/`

Do not reset files while an interactive run is active.
