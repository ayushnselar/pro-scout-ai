# **Cursor Prompts Document — Pro-Scout AI MVP (FINAL OPTIMIZED VERSION)**

Follow every step in this document sequentially. Do not skip steps. Ensure tests pass before continuing. Stop at BREAKPOINTS for manual actions.

---

# **STEP 0 — Project Initialization**

**PROMPT 0.1**

Create project structure:

pro-scout-ai/

 agents/

 graph/

 frontend/

 tests/

 .github/workflows/

 .streamlit/

 docs/

Create requirements.txt with:

streamlit

langgraph

pydantic

duckduckgo-search

nba\_api

groq

pytest

ruff

python-dotenv

Create frontend/app.py minimal Streamlit hello world.

---

**BREAKPOINT 0.A — LOCAL ENVIRONMENT SETUP**

You manually:

python \-m venv .venv

activate venv

pip install \-r requirements.txt

streamlit run frontend/app.py

Confirm Streamlit loads.

---

# **STEP 1 — Schemas and State**

**PROMPT 1.1**

Create graph/schemas.py using Pydantic:

NewsResult  
 StatsResult  
 DecisionResult  
 DecisionEnvelope

Ensure strict validation.

---

**PROMPT 1.2**

Create graph/state.py TypedDict AgentState including:

query  
 players  
 news  
 stats  
 decision

---

**PROMPT 1.3**

Create graph/validate.py helper functions validating all schemas.

---

# **STEP 2 — Entity Extraction (Deterministic \+ Strict JSON LLM fallback)**

**PROMPT 2.1**

Create graph/entity\_extraction.py.

Implement:

extract\_players(query: str) \-\> list\[str\]

Primary method:

* deterministic parsing for:

  * Trade A for B

  * start A

  * sit A

  * start A or B

Use alias mapping for common names.

Fallback method:

extract\_players\_llm\_fallback(query: str) \-\> list\[str\]

This MUST:

* call Groq llama3-8b-8192

* use strict JSON output mode

* response must be exactly:

\["LeBron James", "Stephen Curry"\]

Use Pydantic validation to enforce structure.

Retry once if invalid.

---

**BREAKPOINT 2.A — ADD GROQ API KEY**

Create:

.streamlit/secrets.toml

Add:

GROQ\_API\_KEY="YOUR\_KEY"

---

**PROMPT 2.2**

Create tests/test\_entity\_extraction.py mocking Groq.

---

# **STEP 3 — Scout Agent (Async \+ Batch Summarization)**

**PROMPT 3.1**

Create agents/scout\_agent.py.

Function:

async def get\_news(players: list\[str\]) \-\> list\[NewsResult\]

Requirements:

Use asyncio.gather for parallel search queries.

For each player:

* query DuckDuckGo asynchronously

* collect snippets and URLs

Then perform ONE SINGLE Groq summarization call:

Pass all player snippets in one prompt:

Summarize the fantasy impact and availability status for each player.

Return structured summaries grouped by player name.

Parse into per-player NewsResult.

Fallback:

If Groq fails, concatenate snippets.

---

**PROMPT 3.2**

Ensure timeouts and exception handling.

Return valid NewsResult even if failures occur.

---

**PROMPT 3.3**

Create tests/test\_scout\_agent.py mocking:

* DuckDuckGo

* Groq

Ensure async test compatibility.

---

# **STEP 4 — Analyst Agent (Parallel via ThreadPoolExecutor)**

**PROMPT 4.1**

Create agents/analyst\_agent.py.

Function:

def get\_stats(players: list\[str\]) \-\> list\[StatsResult\]

Use ThreadPoolExecutor:

with ThreadPoolExecutor() as executor:

   results \= executor.map(fetch\_single\_player\_stats, players)

This parallelizes nba\_api calls.

Compute:

season\_avg\_points  
 last5\_avg\_points  
 trend

---

**PROMPT 4.2**

Fallback:

Return valid StatsResult if nba\_api fails.

---

**PROMPT 4.3**

Create tests/test\_analyst\_agent.py mocking nba\_api.

---

# **STEP 5 — GM Agent (Strict JSON Decision)**

**PROMPT 5.1**

Create agents/gm\_agent.py.

Function:

async def make\_decision(...)

Use Groq llama3-70b-8192.

Prompt must enforce strict JSON schema.

Priority rules:

1 injury status  
 2 minutes restriction  
 3 trend  
 4 season average

Fallback:

Return HOLD if invalid output.

---

**PROMPT 5.2**

Validate with Pydantic.

Retry once if invalid JSON.

---

**PROMPT 5.3**

Create tests/test\_gm\_agent.py mocking Groq.

---

# **STEP 6 — LangGraph Workflow (Parallel fan-out \+ fan-in)**

**PROMPT 6.1**

Create graph/workflow.py using LangGraph.

Graph must:

entity\_extraction node

fan-out parallel nodes:

* scout\_agent (async)

* analyst\_agent (parallel threads internally)

fan-in merge results

gm\_agent node

output DecisionEnvelope

Expose:

async def run\_workflow(query: str)

Ensure proper async handling.

---

**PROMPT 6.2**

Create tests/test\_workflow.py mocking agents.

---

# **STEP 7 — Streamlit Web App**

**PROMPT 7.1**

Update frontend/app.py.

UI must include:

text input  
 submit button  
 spinner

Display:

action  
 confidence  
 reasoning  
 sources  
 news expander  
 stats expander

Call run\_workflow asynchronously.

---

**PROMPT 7.2**

Add logging across system.

---

# **STEP 8 — Linting**

**PROMPT 8.1**

Create pyproject.toml configuring ruff.

Ensure:

ruff check .

passes.

---

# **STEP 9 — CI/CD**

**PROMPT 9.1**

Create .github/workflows/ci.yml.

Pipeline must:

install deps  
 run ruff  
 run pytest

---

**BREAKPOINT 9.A — PUSH TO GITHUB**

You manually:

push repo  
 verify CI passes

---

# **STEP 10 — Deployment**

**PROMPT 10.1**

Create .streamlit/config.toml.

Ensure secrets loaded properly.

---

**PROMPT 10.2**

Create README.md including:

setup  
 deployment instructions  
 architecture

---

**BREAKPOINT 10.A — DEPLOY TO STREAMLIT CLOUD**

You manually:

connect GitHub  
 deploy  
 add GROQ\_API\_KEY secret  
 verify public URL

---

# **STEP 11 — Final Quality Gate**

**PROMPT 11.1**

Create docs/build\_log.md documenting:

successful tests  
 deployment URL  
 example queries

---

# **FINAL RESULT**

* async multi-agent system

* parallel data ingestion

* single batch Groq summarization

* strict schema validation

* CI/CD pipeline

* deployed AI web app

