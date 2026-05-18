# Collection Agent

Main customer-facing collections agent.

## First-time setup (run-only)

Use this section when someone is running Collection Agent for the first time.

### 1) Python environment

From repo root (`/Users/saketm10/Projects/openclaw_agents`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Optional voice runtime dependencies:

```bash
pip install ".[voice-realtime]"
```

### 2) Environment variables

Collection Agent reads API keys from environment variables.

Create or update `.env` at repo root:

```bash
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com

# Only needed if you switch config to llm.provider=openai
OPENAI_API_KEY=sk-...
```

Load variables into the current shell:

```bash
set -a
source .env
set +a
```

Notes:

- Current default config uses `llm.provider: nvidia` in `agents/collection_agent/config.yml`.
- If `llm.provider=openai`, `OPENAI_API_KEY` becomes required.

### 3) Run Collection Agent (interactive CLI)

```bash
python agents/collection_agent/main.py --interactive --session-id collection-demo
```

### 4) Run Collection Agent UI

```bash
python -m agents.collection_agent.ui.server
```

Open:

- `http://127.0.0.1:8060/`

Alternative launcher used in this repo:

```bash
./ui-render.sh
```

### 5) Quick health check

```bash
curl http://127.0.0.1:8060/health
```

Expected response:

```json
{"status":"ok"}
```

## Core idea

- Graph handles one internal reasoning pass and always ends at response.
- Response includes both:
  - `response` text
  - `response_target` (`customer` | `self` | `discount_planning_agent`)
  - optional `additional_targets` (for example `collection_memory_helper_agent`)
- Outer orchestration loop in `main.py` decides who gets that response next.

## Collection Graph State Contract

Collection Agent uses:

- Graph state (per turn/pass): `agents/collection_agent/state.py`
- Session memory (persistent across turns): `memory.state` in `SessionStore`

Important: Graph state is rebuilt each run; session memory survives across turns.

### 1) Turn lifecycle and baseline graph state

`CollectionAgent.run_turn()` initializes baseline graph state before the graph starts:

- `session_id`, `turn_id`, `user_input`
- `message_source` (`customer`/`admin`/`self`)
- `user_id`, `case_id`, `channel`
- `memory` (session object)
- `conversation_history` (copied from memory for this turn)
- `observation=None`, `steps=0`
- `node_history=[]`, `conversation_phase="turn_started"`
- `tool_errors=[]`
- `conversation_plan` (copied from `memory.state.active_conversation_plan` when present)
- `turn_index`

### 2) Graph-state keys you will see during execution

| Key | Meaning | Typical writer |
| --- | --- | --- |
| `relevance_intent` | Relevance classification payload (`intent`, `confidence`, `reason`) | `relevance_intent` |
| `pre_plan_intent` | Pre-plan route intent (`plan`/`decide`) | `pre_plan_intent` |
| `execution_path_intent` | Decide execution route (`need_memory`/`need_tool`) | `execution_path_intent` |
| `post_memory_plan_intent` | Post-memory route intent (`plan`/`react`) | `post_memory_plan_intent` |
| `intent` | Compatibility mirror of current intent payload | each intent node |
| `route` | Current routing decision for the node | inferred in node wrapper |
| `node_history` | Exact node traversal order for this run | node wrapper |
| `previous_node` | Previous node in this run | node wrapper |
| `next_node` | Resolved next node (or candidates) | node wrapper |
| `conversation_phase` | Phase label (`entity_extraction`, `plan_proposal`, etc.) | node wrapper |
| `decision` | ReAct decision object (tool call or direct response intent) | `react` |
| `observation` | Tool execution observation payload | `tool_execution` |
| `extracted_entities` | Session-level merged extracted entities | `entity_extract` |
| `extracted_entities_turn` | Entities extracted in current turn only | `entity_extract` |
| `extracted_entity_descriptions` | Descriptions/schema hints for extracted fields | `entity_extract` |
| `extracted_entities_updated_fields` | Fields changed vs previous memory snapshot | `entity_extract` |
| `verification_entities` | Verification-focused entity map (name/dob/phone, etc.) | `entity_extract`, `plan_proposal` |
| `verification_missing_fields` | Verification fields still missing | `entity_extract`, `plan_proposal` |
| `verification_verified_fields` | Verified verification fields | `plan_proposal` |
| `identity_verified` | Whether verification is complete | `entity_extract`, `plan_proposal` |
| `plan_proposal` | Planner proposal payload (target, outline, next actions, tree update) | `plan_proposal` |
| `conversation_plan` | Current plan tree snapshot for this run | `plan_proposal` |
| `routing_context.plan_origin` | Origin of planner invocation (`pre_plan_intent`, `post_memory_plan_intent`, `react`) | node wrapper |
| `reflection_feedback` | Reflect validation payload (`reason`, `is_complete`) | `reflect` |
| `reflection_complete` | Whether reflection accepted current proposal | `reflect` |
| `reflection_retry_count` | Retry counter used by reflect loop guard | `reflect` |
| `reflection_plan_retry_count` | Plan-specific retry counter | `reflect` |
| `failure_type` | Reflection failure class (`none`, `plan_correction_needed`, etc.) | `reflect` |
| `correction_hints` | Reflect hints for planner retry | `reflect` |
| `retry_target` | Retry destination (`plan_proposal` or `none`) | `reflect` |
| `response` | Final customer/system response text for this pass | `relevant_response` / `irrelevant_response` |
| `response_target` | Routing target after graph (`customer`, `self`, `discount_planning_agent`) | `plan_proposal`, normalized in response/finalizer |
| `additional_targets` | Optional extra recipients (for example memory helper) | `plan_proposal` |
| `handoff_payload` | Payload for cross-agent handoff | `plan_proposal` |
| `memory_helper_trigger` | Trigger payload for memory-helper follow-up | `plan_proposal` |
| `conversation_history` | Bounded history also returned in output state | `relevant_response` |
| `prompt` | Rendered prompt for the current node (debug) | node-specific |
| `system_prompt` | Effective system prompt for the current node (debug) | node-specific |
| `llm_response` | Structured/raw LLM output captured for debugging | node-specific |
| `llm_error` | LLM exception text if generation failed | node-specific |
| `llm_status` | `used_llm`, `llm_error`, `prompt_rendered_no_output`, or fallback markers | node wrapper / node-specific |
| `fallback_reason` | Explicit fallback reason (for example provider rate-limit) | intent/response nodes |

### 3) Node ownership map (who updates what)

| Node | Primary graph-state updates |
| --- | --- |
| `relevance_intent` | `relevance_intent`, `intent`, debug keys (`prompt`, `llm_*`) |
| `entity_extract` | `extracted_entities`, `extracted_entities_turn`, `extracted_entity_descriptions`, `verification_entities`, `verification_missing_fields`, `identity_verified`, `memory_context`, debug keys |
| `pre_plan_intent` | `pre_plan_intent`, `intent`, debug keys |
| `execution_path_intent` | `execution_path_intent`, `intent`, debug keys |
| `memory_retrieve` | `memory_context`, `memory_retrievals` |
| `post_memory_plan_intent` | `post_memory_plan_intent`, `intent`, debug keys |
| `react` | `decision`, `steps` (+ debug keys when available) |
| `tool_execution` | `observation` (+ writes session `tool_observations_history` via wrapper side-effect) |
| `plan_proposal` | `plan_proposal`, `conversation_plan`, `route`, `response_target`, verification state mirrors, optional `handoff_payload`/`additional_targets` |
| `reflect` | `reflection_feedback`, `reflection_complete`, retry/failure keys |
| `relevant_response` | `response`, `response_target`, `conversation_history`, debug keys |
| `irrelevant_response` | static `response`, `response_target` |
| wrapper (`_wrap_node`) | `node_history`, `previous_node`, `next_node`, default `conversation_phase`, inferred `route`, computed `llm_status` |

### 4) Persistent session memory keys (cross-turn)

These are not graph-state-only, but they drive graph behavior every turn:

- `active_user_id`, `active_case_id`, `active_channel`
- `active_customer_name`, `active_overdue_amount`, `active_emi_amount`, `active_late_fee`, `active_dpd`
- `active_verification_required_fields`, `active_verification_challenge`
- `verification_entities`, `verification_collected`
- `verification_verified_fields`, `verification_missing_fields`, `verification_last_status`
- `identity_verified`
- `active_conversation_plan`
- `conversation_history` (bounded, currently last 40 entries)
- `tool_observations_history` (bounded, currently last 40 entries)
- `last_user_input`, `last_agent_response`, `last_response_target`, `turn_index`

### 5) Quick debug checks

When behavior looks wrong, inspect these first in order:

1. `node_history` (did graph visit expected nodes?)
2. `route`, `next_node`, `conversation_phase` (was routing correct?)
3. `verification_entities`, `verification_missing_fields`, `identity_verified` (verification stage truth)
4. `plan_proposal` + `conversation_plan.current_node_id` (planner stage)
5. `reflection_feedback` + `reflection_complete` (retry loop reason)
6. `response` + `response_target` (final output contract)

## Graph (single-pass)

```mermaid
flowchart TD
  Start["START"] --> IN["IntentNode (Relevance Gate)\n- relevant | irrelevant | empty"]
  IN -->|irrelevant/empty| IR["IrrelevantResponseNode\n- static response"]
  IR --> End["END"]

  IN -->|relevant| EN["CollectionEntityExtractNode\n- LLM entity extraction\n- verification entity sync"]
  EN --> PG["IntentNode (Pre-Plan Gate)\n- plan | decide"]
  PG -->|plan| PP["PlanProposalNode\n- plan advance / next action"]
  PG -->|decide| XP["IntentNode (Execution Path)\n- need_memory | need_tool"]

  XP -->|need_memory| MR["MemoryRetrieveNode"]
  MR --> PM["IntentNode (Post-Memory Plan Gate)\n- plan | react"]
  PM -->|plan| PP
  PM -->|react| RN["ReactNode"]

  XP -->|need_tool| RN
  RN -->|act| TE["ToolExecutionNode"]
  TE --> RN
  RN -->|respond/end| PP

  PP -->|propose| TE
  PP -->|continue| RF["CollectionReflectNode"]

  RF -->|retry_react| RN
  RF -->|retry_plan| PP
  RF -->|complete| RR["RelevantResponseNode\n- response + response_target + additional_targets"]
  RR --> End
```

Graph assets:

- `graph.mmd`
- `graph.png`
- `graph.jpg`

## Node Definitions

### `IntentNode (Relevance Gate)`

- Classifies whether input is in collections scope.
- Routes to relevant flow or immediate irrelevant response.

### `IrrelevantResponseNode`

- Produces static out-of-scope response.
- Terminates current graph pass.

### `CollectionEntityExtractNode`

- Runs immediately after relevance pass for in-scope turns.
- Extracts entities (LLM-first), syncs verification entities, and updates verification state context before planning/routing.

### `IntentNode (Pre-Plan Gate)`

- Decides whether to start from plan proposal or execution path decision.

### `IntentNode (Execution Path)`

- Chooses between memory-first route or immediate tool/planning route.

### `MemoryRetrieveNode`

- Loads session memory context for better continuity and follow-up handling.

### `IntentNode (Post-Memory Plan Gate)`

- Re-evaluates whether to continue with plan proposal or React flow after memory retrieval.

### `ReactNode`

- Decides next operation (`act`, `respond`, `end`) and tool arguments when needed.

### `ToolExecutionNode`

- Executes selected tool and returns structured observation payload.

### `PlanProposalNode`

- Advances conversation plan, triggers plan revisions, and emits routing targets (`customer`, `self`, `discount_planning_agent`).
- Detects conversation termination and may add `collection_memory_helper_agent` to additional targets.

### `CollectionReflectNode`

- Checks if current pass is complete.
- Routes back for another planning cycle when output is incomplete.

### `RelevantResponseNode`

- Produces final response payload for this graph pass.
- Emits `response_target` and optional `additional_targets` for external orchestration in `main.py`.

## Outer routing (outside graph)

After graph ends at response:

1. if `response_target=customer` -> return to UI/user
2. if `response_target=self` -> re-run collection agent with internal response context
3. if `response_target=discount_planning_agent` -> call discount agent, store result in memory, re-run collection agent
4. if `additional_targets` contains `collection_memory_helper_agent` -> send full conversation payload to memory helper agent after turn completion

Implementation note:

- `CollectionAgent.run()` and `CollectionAgent.run_turn()` perform one forward graph pass only.
- Multi-agent hop orchestration is handled in `agents/collection_agent/main.py` interactive loop (not inside `CollectionAgent.run()`).
- Loop guards in interactive mode:
  - soft cap default: `10` hops (`--agent-hop-soft-cap`)
  - hard cap default: `50` hops (`--agent-hop-hard-cap`)
  - on cap hit, a memory flag is set (`agent_loop_blocked=true`) and collection agent returns a guardrail response.

## Tool Table

| Tool | Description | Typical Inputs | Typical Output | Removed |
| --- | --- | --- | --- | --- |
| `case_fetch` | Retrieves borrower case records with DPD, overdue amount, risk band, and loan linkage so the agent can anchor all downstream decisions in case facts. | `case_id?`, `customer_id?`, `portfolio_id?` | `cases[]`, `total` | `yes` |
| `case_prioritize` | Scores and orders delinquent cases so collections effort starts with highest-risk and highest-recovery opportunities. | `case_ids?`, `portfolio_id?` | `queue[]`, `total` | `yes` |
| `contact_attempt` | Logs outreach attempts across channels and preserves reachability history for follow-up strategy and compliance traceability. | `case_id`, `channel?`, `reached?` | `attempt_id`, `status` | `yes` |
| `verify_dob` | Verifies customer DOB against case-linked challenge record before sensitive dues disclosure. | `case_id?`, `customer_id?`, `dob` | `status`, `field=dob`, `failed_attempts` | `no` |
| `verify_mobile` | Verifies customer mobile number against case-linked challenge record before sensitive dues disclosure. | `case_id?`, `customer_id?`, `phone` | `status`, `field=phone`, `failed_attempts` | `no` |
| `entity_extract` | Extracts generic entities from raw input text (IDs, DOB, phone, PAN-last4, ZIP, name) for downstream orchestration/state updates. | `text` | `entities`, `entity_keys` | `no` |
| `verification_entity_extract` | Filters raw text down to verification-relevant entities required for identity checks. | `text`, `required_fields`, `include_name?` | `entities`, `detected_fields`, `missing_fields` | `no` |
| `verification_memory_verify` | Verifies extracted verification entities against expected challenge values cached in memory state. | `entities`, `expected_challenge`, `required_fields` | `status`, `matched`, `missing_fields`, `mismatched_fields` | `no` |
| `loan_policy_lookup` | Fetches policy constraints (waiver/restructure/promise windows) that govern which offers can legally be proposed. | `case_id?`, `loan_id?` | policy object | `no` |
| `dues_explain_build` | Builds a borrower-friendly dues explanation that summarizes principal overdue, fees, and total payable amount. | `case_id` | `explanation`, `total_due` | `yes` |
| `offer_eligibility` | Checks baseline concession eligibility and produces initial direction (`waiver`, `restructure`, or `none`) under policy rules. | `case_id`, `hardship_flag?`, `requested_waiver_pct?` | `allowed`, `offer_type`, `approved_waiver_pct` | `no` |
| `plan_propose` | Generates or revises repayment plan options (tenure + EMI) for hardship negotiation flows. | `case_id`, `revision_index?`, `max_installment_amount?` | `plan_id`, `monthly_amount`, `first_due_date`, `status` | `no` |
| `payment_link_create` | Creates a payment link for borrowers ready to pay now on digital channels. | `case_id`, `amount`, `channel?` | `payment_reference_id`, `payment_url` | `no` |
| `pay_by_phone_collect` | Simulates assisted payment collection over voice where borrower pays during the call. | `case_id`, `amount`, `consent_confirmed?` | `payment_id`, `status`, `receipt_reference` | `yes` |
| `payment_status_check` | Verifies whether initiated payment is completed/pending/failed and whether more action is required. | `payment_reference_id` | `status`, `needs_additional_action` | `yes` |
| `promise_capture` | Persists promise-to-pay commitments (amount/date/channel) for accountability and future tracking. | `case_id`, `promised_date`, `promised_amount` | `promise_id`, `status` | `no` |
| `followup_schedule` | Creates reminder/callback tasks when immediate payment is not possible. | `case_id`, `scheduled_for`, `preferred_channel?` | `schedule_id` | `yes` |
| `disposition_update` | Writes final interaction disposition and notes for audit, reporting, and queue progression. | `case_id`, `disposition_code`, `notes` | `audit_id`, `updated_at` | `yes` |
| `channel_switch` | Moves conversation to requested channel (voice/sms/email/whatsapp) while retaining interaction context. | `case_id`, `from_channel?`, `to_channel?` | `switch_id`, `carried_context_summary` | `yes` |
| `human_escalation` | Escalates exceptional cases (fraud/legal/dispute/sensitive) to human specialist queues. | `case_id`, `reason` | `escalation_id`, `queue`, `priority` | `no` |

## Specialist agent used outside graph

- `agents/discount_planning_agent`
- `agents/collection_memory_helper_agent`

Collection agent does not treat specialist handoff as an internal graph node or tool.

## Pipecat voice runtime integration

Pipecat integration is added as a runtime adapter, not as an internal graph node.

- shared integration helpers: `src/interfaces/pipecat_runner.py`
- collection voice bot entrypoint: `agents/collection_agent/pipecat_bot.py`

### What this does

- Pipecat handles transport/session/audio IO (WebRTC, Daily, telephony transports).
- OpenAI STT converts caller audio to text.
- Text is routed to existing collection orchestration (`_route_internal_turn`) with discount and memory-helper handoffs unchanged.
- Returned text is converted back to speech via OpenAI TTS.

### Runtime config

Defined in `agents/collection_agent/config.yml` under `voice_runtime`:

- `stt_model`
- `tts_model`
- `tts_voice`
- `pipecat.vad_enabled`
- `pipecat.audio_in_sample_rate`
- `pipecat.audio_out_sample_rate`

### Verification pipeline config

Defined in `agents/collection_agent/config.yml` under `verification`:

- `required_fields`: which fields must match for verification (example: `dob`, `phone`)
- `require_field_in_challenge`: enforce only fields available in challenge payload
- `auto_verify_from_memory_evidence`: allow pre-graph memory verification to set `identity_verified`
- `require_name_match_for_auto_verify`: require extracted name match with active customer
- `tool_only_verification`: disable memory verification success; only `verify_dob`/`verify_mobile` can verify
- `prefer_memory_verify`: verify against cached memory challenge first
- `fallback_to_database_verify`: if memory verify is insufficient/failed, call database-backed `verify_dob`/`verify_mobile`
- `fallback_on_insufficient_entities`: if `false`, skip DB verify when required entities are still missing

### Run

```bash
# install optional dependencies
pip install ".[voice-realtime]"

# local browser WebRTC
python agents/collection_agent/pipecat_bot.py -t webrtc

# daily transport
python agents/collection_agent/pipecat_bot.py -t daily

# telephony transport via twilio (public proxy required)
python agents/collection_agent/pipecat_bot.py -t twilio -x <your-ngrok-domain>
```

## NVIDIA model setup (LLM provider)

Collection agent now supports `llm.provider: nvidia` using NVIDIA hosted Integrate API (OpenAI-compatible chat endpoint).

Update `agents/collection_agent/config.yml`:

```yaml
llm:
  enabled: true
  provider: nvidia
  model_name: meta/llama-3.1-70b-instruct
  base_url: https://integrate.api.nvidia.com
```

Set env:

```bash
export NVIDIA_API_KEY="nvapi-..."
```

Optional override at runtime:

```bash
python agents/collection_agent/main.py --interactive --nvidia-api-key "nvapi-..."
```

## Key-event memory stores

Both stores are physically under collection agent runtime so collection logic can read them directly:

- `agents/collection_agent/runtime/memory/global_key_event_memory.json`
  - cross-user successful/unsuccessful procedural cues
  - event counters and sample signals used as planning hints
- `agents/collection_agent/runtime/memory/user_key_event_memory.json`
  - per-`user_id` latest summary
  - per-`user_id` procedural key points + follow-up considerations
  - per-`user_id` conversation outcome history

## Regenerate graph

```bash
npx -y @mermaid-js/mermaid-cli -i agents/collection_agent/graph.mmd -o agents/collection_agent/graph.png -b white -s 2
python3 -c "from PIL import Image; Image.open('agents/collection_agent/graph.png').convert('RGB').save('agents/collection_agent/graph.jpg', 'JPEG', quality=92)"
```
