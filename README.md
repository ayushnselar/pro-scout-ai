# Pro-Scout AI

Multi-agent **fantasy basketball** decision assistant (MVP): combines live-style **news signals** (web search + LLM summarization) with **NBA stats** (`nba_api`) and a **GM** model that returns a structured **START**, **SIT**, **TRADE_ACCEPT**, **TRADE_REJECT**, or **HOLD** decision.

For a full codebase tour, see [walkthrough.md](walkthrough.md). Product intent is in [docs/PRD_MVP.md](docs/PRD_MVP.md).

---

## Setup (local)

**Requirements:** Python **3.11+**

```bash
cd pro-scout-ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### API key (Groq)

The app reads **`GROQ_API_KEY`** from the environment or from Streamlit secrets.

**Local Streamlit**

1. Copy the example file:

   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. Edit `.streamlit/secrets.toml` and set your real key.  
   That path is **gitignored** — do not commit it.

**Optional:** `export GROQ_API_KEY=...` in your shell instead of (or in addition to) `secrets.toml`.

### Run the app

From the repo root:

```bash
streamlit run frontend/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### Tests and lint

```bash
python -m pytest -q
ruff check .
ruff format .
```

---

## Deployment (Streamlit Community Cloud)

High level (manual steps — see **BREAKPOINT 10.A** in [docs/CURSOR_BUILD_STEPs.md](docs/CURSOR_BUILD_STEPs.md)):

1. Push this repo to **GitHub**.
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), **New app** → pick the repo and branch.
3. Set **Main file path** to: `frontend/app.py`
4. Under **App settings → Secrets**, add TOML matching `.streamlit/secrets.toml.example`:

   ```toml
   GROQ_API_KEY = "your-real-key"
   ```

5. **Deploy** and open the public app URL.

Runtime config for the hosted app comes from [`.streamlit/config.toml`](.streamlit/config.toml) in the repo (browser/server defaults). Secrets always come from the Cloud **Secrets** UI (or local `secrets.toml`), not from `config.toml`.

---

## Architecture

| Layer | Role |
|--------|------|
| **`frontend/app.py`** | Streamlit UI; runs the async workflow with a timeout; renders decision, news, stats, debug. |
| **`graph/workflow.py`** | **LangGraph**: entity extraction → **Scout** + **Analyst** in parallel → merge → **GM**. |
| **`graph/entity_extraction.py`** | Regex/alias player extraction; Groq JSON fallback. |
| **`agents/scout_agent.py`** | `ddgs` search + filters + one batch Groq summarization → `NewsResult` list. |
| **`agents/analyst_agent.py`** | `nba_api` game logs → points averages + trend → `StatsResult` list. |
| **`agents/gm_agent.py`** | Groq strict JSON decision + retry + heuristic fallback → `DecisionResult`. |
| **`graph/schemas.py`** | Pydantic contracts (`DecisionEnvelope`, etc.). |
| **`.github/workflows/ci.yml`** | CI: `pip install -r requirements.txt` → `ruff check .` → `pytest`. |

Data flow: **question** → extracted **players** → **news** + **stats** (parallel) → **decision** + sources → UI.

---

## Repository layout

```
agents/          # Scout, Analyst, GM
graph/           # LangGraph workflow, schemas, validation, entity extraction
frontend/        # Streamlit entrypoint
tests/           # pytest suite
docs/            # PRD and build steps
.streamlit/      # config.toml (+ local secrets.toml, gitignored)
```

---

## License / status

MVP / educational use. Tune timeouts, models, and prompts in code and in [docs/PRD_MVP.md](docs/PRD_MVP.md) as needed.
