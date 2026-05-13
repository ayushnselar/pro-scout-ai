# Pro-Scout AI: Full Codebase Walkthrough

## Purpose of this document

This walkthrough explains the project as if the reader has no prior context. It covers:

- What the system is trying to do
- How each layer is structured
- Why specific implementation choices were made
- How data moves through the app at runtime
- How reliability/fallback behavior works
- What tests validate and what assumptions remain

---

## 1) What this project is

`pro-scout-ai` is an MVP fantasy basketball decision-support app.

At a high level, a user asks a question like:

- `Start Stephen Curry tonight?`
- `Trade LeBron James for Kevin Durant?`

The system then:

1. Extracts player entities from the text
2. Fetches recent qualitative signals (news)
3. Fetches quantitative signals (stats)
4. Combines those signals into one structured decision
5. Renders the result in a Streamlit UI

The app is implemented as a multi-agent pipeline orchestrated by LangGraph.

---

## 2) Repository map and role of each area

## Root

- `requirements.txt`: runtime/test dependencies
- `walkthrough.md`: this file

## `agents/`

Domain agents that do data retrieval and decision synthesis.

- `scout_agent.py`: retrieves and summarizes news signals
- `analyst_agent.py`: retrieves and summarizes performance stats
- `gm_agent.py`: synthesizes final decision from scout + analyst outputs

## `graph/`

Workflow orchestration and strict schemas.

- `schemas.py`: Pydantic models and enums for all structured data
- `state.py`: LangGraph shared state shape (`TypedDict`)
- `entity_extraction.py`: player extraction logic (deterministic + LLM fallback)
- `workflow.py`: graph nodes/edges and execution entrypoints
- `validate.py`: helper validators for JSON/model safety

## `frontend/`

- `app.py`: Streamlit UI and user interaction loop

## `tests/`

Behavioral tests by module:

- `test_entity_extraction.py`
- `test_scout_agent.py`
- `test_analyst_agent.py`
- `test_gm_agent.py`
- `test_workflow.py`

## `docs/`

- `PRD_MVP.md`: product requirements and architecture intent
- `CURSOR_BUILD_STEPS.md`: implementation/build instructions

---

## 3) Dependency stack and why each exists

From `requirements.txt`:

- `streamlit`: web app frontend
- `langgraph`: orchestrates multi-agent flow with fan-out/fan-in
- `pydantic`: schema-first contracts and validation
- `ddgs`: DuckDuckGo search wrapper used by Scout
- `nba_api`: structured NBA team/player/stats data
- `groq`: LLM provider/client for extraction + summarization + decisioning
- `pytest`: tests
- `ruff`: linting
- `python-dotenv`: environment variable support (listed; limited direct usage)

Design intent behind this stack:

- Fast iteration (Streamlit)
- Explicit workflow topology (LangGraph)
- Controlled AI output surface (Pydantic + strict JSON)
- Hybrid signals (news + stats) to reduce single-source bias

---

## 4) Data contracts (schema-first foundation)

`graph/schemas.py` defines the core contract boundary.

### Enums

- `Trend`: `UP | DOWN | STABLE`
- `DataSource`: `nba_api | fallback | unavailable`
- `Action`: `START | SIT | TRADE_ACCEPT | TRADE_REJECT | HOLD`

### Models

- `NewsResult`
  - `player`, `news_summary`, `source_urls`, `retrieved_at_iso`
  - plus structured optional fields: `availability`, `minutes_note`, `role_note`, `fantasy_impact`

- `StatsResult`
  - `player`, `season_avg_points`, `last5_avg_points`, `trend`, `data_source`

- `DecisionResult`
  - `action`, `confidence` (0 to 1), `reasoning`, `sources`

- `DecisionEnvelope`
  - top-level response combining query, extracted players, news, stats, decision, and freshness metadata

Why this matters:

- Ensures all modules agree on format
- Prevents loosely structured LLM output from leaking downstream
- Makes frontend rendering deterministic

`graph/validate.py` adds reusable parsing/validation helpers for raw JSON/model coercion.

---

## 5) Runtime architecture and flow

The orchestration lives in `graph/workflow.py`.

### Graph shape

1. `entity_extraction`
2. Fan-out to:
   - `scout_agent`
   - `analyst_agent`
3. Fan-in at `merge_results`
4. `gm_agent`
5. End

### Why this graph shape

- Entity extraction must happen first because both Scout and Analyst need player names.
- Scout and Analyst are independent once players are known, so they run in parallel for latency.
- GM requires both qualitative and quantitative signals, so explicit fan-in ensures both are complete.

### Public workflow APIs

- `run_workflow(query)` -> returns `DecisionEnvelope`
- `run_workflow_with_debug(query)` -> returns `(DecisionEnvelope, debug_meta)`

The frontend uses the debug variant to show retrieval diagnostics.

---

## 6) Entity extraction internals (`graph/entity_extraction.py`)

This module translates free-form user prompts into player names.

### Deterministic path (`extract_players`)

Uses regex patterns for common fantasy phrasing:

- `trade A for B`
- `start X`
- `sit X`
- `start X or Y`

Normalization and cleanup behavior:

- Alias mapping (e.g., `steph` -> `Stephen Curry`, `lbj` -> `LeBron James`)
- Trims matchup/filler words (`vs`, `tonight`, `this week`, etc.)
- Splits multi-name fragments on `and`, `or`, commas

Reasoning:

- Deterministic extraction is low-latency, low-cost, and robust for common intents.

### LLM fallback (`extract_players_llm_fallback`)

If deterministic extraction returns nothing:

- Calls Groq model (`llama-3.1-8b-instant`)
- Requires strict JSON array output
- Validates with `_PlayersList` Pydantic `RootModel`
- Retries once on failure
- Returns `[]` on repeated failure

Reasoning:

- Adds recall for unstructured phrasing while preserving strict schema safety.

---

## 7) Scout agent internals (`agents/scout_agent.py`)

Scout gathers recent qualitative context for each player and returns structured news signals.

### Stage A: Query generation

For each player, builds three search queries:

- injury update
- status tonight
- minutes restriction

Enhancements:

- Optional team context from `nba_api` (`_get_team`)
- Recency bias (`tonight`) if query implies immediacy/date
- Negative keyword suffix to reduce irrelevant name/bio results

### Stage B: Retrieval + concurrency model

Important implementation choices:

- Uses sync `ddgs` calls inside executor threads
- Wraps each query in async timeout (`_DDG_QUERY_TIMEOUT_S`)
- Batches per-player queries with per-player timeout
- Batches all-player retrieval with global timeout
- Caps concurrent DDG jobs with an `asyncio.Semaphore`

Critical detail:

- Semaphore is created **inside `get_news`** (same active event loop), avoiding event-loop binding issues in Streamlit reruns.

### Stage C: Filtering

`_filter_ddg_results` removes likely junk by:

- Domain denylist (e.g., Wikipedia, baby-name sites)
- Keyword denylist (e.g., `etymology`, `name meaning`)
- Empty snippets

Tracks removal counts for debug metadata.

### Stage D: Batch summarization

After retrieval/filtering, Scout makes **one** Groq call for all players:

- `_groq_summarize_batch(players_snippets)`
- Expects strict JSON map keyed by exact player names
- Normalizes response into structured fields

Why one call:

- Avoid N+1 latency pattern
- Reduce API overhead and rate-limit pressure

### Stage E: Final assembly and fallbacks

For each input player, always returns one `NewsResult`.

Fallback behavior:

- If Groq fails: use concatenated snippets
- If no snippets: `No relevant news found.`
- If retrieval fails/timeouts: still return schema-valid result

Debug metadata includes:

- per-player before/after counts
- filter-reason totals
- whether Groq summarization succeeded

---

## 8) Analyst agent internals (`agents/analyst_agent.py`)

Analyst provides quantitative signal based on NBA points performance.

### Per-player stats process (`fetch_single_player_stats`)

1. Resolve player ID from full name via `nba_api.stats.static.players`
2. Load current season game log via `PlayerGameLog`
3. Pull `PTS` values
4. Compute:
   - season average points
   - last-5 average points
   - trend (`UP/DOWN/STABLE`) using +/-2 threshold

### Parallelism

`get_stats(players)` uses `ThreadPoolExecutor` with `executor.map`.

Reasoning:

- `nba_api` is blocking I/O; thread-based parallelism improves throughput.

### Failure behavior

On any per-player exception, returns safe fallback `StatsResult`:

- points fields = `None`
- trend = `STABLE`
- data_source = `UNAVAILABLE`

No exceptions propagate to crash the workflow.

---

## 9) GM agent internals (`agents/gm_agent.py`)

GM combines Scout + Analyst outputs into one user-facing decision.

### Prompt construction

`_build_decision_prompt` creates structured context including:

- user query
- per-player structured news signals
- per-player stats signals
- explicit priority rules (injury/minutes > trend > season)
- strict output schema instructions

### LLM call and strict parsing

`_call_groq_decision_once`:

- Calls Groq model `llama-3.3-70b-versatile`
- Requires JSON-only output validated into `DecisionResult`
- If parse fails, attempts JSON salvage (`_salvage_first_json_object`)
- Filters model-returned sources to known URLs from Scout

### Retry and fallback strategy

`make_decision`:

- Attempt 1 with timeout
- Attempt 2 with timeout on failure
- If both fail, use deterministic heuristic fallback (`_heuristic_fallback_decision`)

Heuristic fallback supports common intents:

- `start A or B`
- `start A`
- `trade A for B`

Else returns conservative `HOLD`.

Why this exists:

- Keeps app functional when LLM API is down, malformed, or times out.

---

## 10) Frontend behavior (`frontend/app.py`)

Streamlit app is the user interface layer.

### Interaction model

- Text input for question
- Submit button
- Spinner while running workflow

### Async boundary handling

Because Streamlit scripts are synchronous by default:

- wraps async workflow in `asyncio.run`
- wraps workflow call in a hard 15s timeout (`asyncio.wait_for`)

If timeout occurs:

- UI shows error message and stops rendering downstream blocks

### Rendering

Displays:

- decision action
- confidence
- reasoning
- sources
- news cards
- stats cards
- debug sections (raw summaries + scout retrieval metadata)

Design rationale:

- Main output remains simple
- Expanded debug panes aid diagnosis without cluttering default UX

---

## 11) State model and orchestration semantics

`graph/state.py` defines `AgentState` as evolving keys:

- required initially: `query`
- progressively added: `players`, `news`, `stats`, `decision`, `scout_debug`

Nodes in `graph/workflow.py` return partial dictionaries. LangGraph merges these updates into shared state.

Implication:

- Each node focuses on its own responsibility while preserving previous state.

---

## 12) Reliability and fault-tolerance strategy

The codebase intentionally adds layered protection against external failures.

### Timeouts

- DDG query timeout
- per-player DDG timeout
- global DDG timeout
- NBA team lookup timeout
- Groq Scout timeout
- Groq GM timeout
- Frontend workflow timeout

### Defensive defaults

- empty players -> still valid workflow behavior
- retrieval failures -> schema-valid empty/placeholder outputs
- parse failures -> retry and salvage where possible
- total GM failure -> deterministic heuristic fallback (or HOLD)

### Validation boundaries

- LLM outputs are parsed/validated before use
- final envelope is validated before returning to UI

---

## 13) Testing strategy and what is covered

Tests are organized by capability and use monkeypatching/fake modules to isolate behavior.

### `test_entity_extraction.py`

Covers:

- deterministic extraction patterns and alias normalization
- LLM fallback success
- retry-on-invalid-JSON behavior
- empty output on repeated failure

### `test_scout_agent.py`

Covers:

- successful batch summarization
- Groq failure fallback to snippets
- DDG failure behavior
- filtering junk domains/keywords
- timeout resilience while preserving per-player schema output

### `test_analyst_agent.py`

Covers:

- successful points/trend computation from fake `nba_api`
- fallback behavior when player resolution fails

### `test_gm_agent.py`

Covers (inferred from module role and naming):

- strict parsing and retry logic
- salvage/fallback behavior
- source filtering and decision output contract

### `test_workflow.py`

Covers:

- full workflow orchestration with mocked agents
- fan-out/fan-in sequencing assumptions
- behavior when no players are extracted

General testing philosophy:

- Verify not just happy path but failure paths, fallback guarantees, and schema stability.

---

## 14) Implementation choices and design reasoning

This project consistently applies a few engineering principles:

1. **Schema-first boundaries**
   - Makes AI-heavy system predictable and easier to render/test.

2. **Parallel where safe**
   - Independent data retrieval runs concurrently to improve latency.

3. **Graceful degradation over hard failure**
   - User still receives an actionable response under partial outages.

4. **Single-responsibility modules**
   - Agents handle domain tasks; graph handles orchestration; frontend handles presentation.

5. **Conservative defaults for uncertainty**
   - HOLD-like behavior in uncertain states minimizes risky outputs.

---

## 15) Operational notes for a new maintainer

### How to run

Typical local run command:

- `streamlit run frontend/app.py`

### Required secrets

- `GROQ_API_KEY` (env var or Streamlit secrets)

### Main extension points

- Add richer extraction patterns in `graph/entity_extraction.py`
- Add new stat features in `agents/analyst_agent.py`
- Improve news relevance/filtering in `agents/scout_agent.py`
- Refine decision policy/rubric in `agents/gm_agent.py`
- Add new workflow nodes in `graph/workflow.py`

---

## 16) Known gaps / likely next improvements

1. Add a root `README.md` with setup/run/debug instructions.
2. Add real CI workflow in `.github/workflows` to run tests/lint automatically.
3. Expand analyst metrics beyond points (usage, minutes trend, opponent context).
4. Reduce p95 latency to align with strict sub-5s goal in PRD.
5. Add observability (structured logs, per-node timing, error counters).
6. Harden secrets handling and avoid storing plaintext keys in local project files.

---

## 17) End-to-end example mental model

For query: `Start Jayson Tatum or Kevin Durant tonight?`

1. Entity extraction returns `['Jayson Tatum', 'Kevin Durant']`
2. Scout fetches and filters three DDG query streams per player
3. Scout does one Groq batch summarization and structures injury/minutes/role/impact
4. Analyst fetches each player’s current-season game log and computes trend metrics
5. GM receives all signals, applies priority logic, and returns strict JSON decision
6. Workflow validates final envelope
7. Frontend displays action/confidence/reasoning/sources + debug detail

That is the complete project loop.
