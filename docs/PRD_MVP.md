# **Pro-Scout AI — MVP Product Requirements Document (Final Version)**

## **1\. Product Overview**

### **1.1 Product Name**

Pro-Scout AI (MVP)

### **1.2 Product Type**

Cloud-hosted AI-powered decision support web application.

### **1.3 Objective**

Build a deployed multi-agent system that retrieves real-time sports news and live statistical performance data, synthesizes both using an LLM, and outputs structured fantasy sports recommendations.

The system must:

* Produce explicit decisions (START, SIT, TRADE\_ACCEPT, TRADE\_REJECT, HOLD)

* Provide confidence score and reasoning

* Complete end-to-end execution in under 5 seconds

* Be publicly accessible via web interface

* Be deployable and testable with CI/CD pipeline

---

## **2\. MVP Scope**

### **Included in MVP**

* Multi-agent orchestration using LangGraph

* Parallel real-time news retrieval

* Parallel live statistical retrieval

* Batch LLM summarization of news signals

* LLM decision synthesis

* Structured decision outputs with strict schema validation

* Streamlit web application frontend

* Cloud deployment (Streamlit Cloud)

* CI/CD pipeline with automated testing

* Robust error handling and fallback behavior

### **Explicitly Excluded (Post-MVP)**

* User accounts

* League integrations (Sleeper, ESPN, Yahoo)

* Database storage

* Long-term learning or feedback loops

* Mobile app

* Authentication system

---

## **3\. System Architecture**

### **3.1 High-Level Architecture**

User (Web Browser)

  ↓

Streamlit Web App

  ↓

LangGraph Orchestration Layer

  ↓

Entity Extraction Node

  ↓

Parallel Execution (Fan-Out)

  ├── Scout Agent (News Retrieval \+ Batch Summarization)

  └── Analyst Agent (Statistical Retrieval)

  ↓

Merge Results (Fan-In)

  ↓

GM Agent (Decision Engine)

  ↓

Structured Decision Output

  ↓

Web Interface Display

---

## **4\. Core Design Principles**

### **Parallelism**

The system must maximize performance by executing independent tasks concurrently:

* Scout Agent processes players concurrently using asyncio

* Analyst Agent processes players concurrently using ThreadPoolExecutor

* Scout and Analyst agents execute in parallel using LangGraph fan-out/fan-in

This ensures consistent sub-5 second response times.

---

### **Batch Processing**

The Scout Agent must perform a single batch LLM summarization call per query to avoid:

* Excessive API latency

* Rate limit issues

* N+1 API call pattern

---

### **Schema-First Validation**

All agent outputs must conform to strict Pydantic schemas.

Invalid outputs must trigger retry or fallback behavior.

No free-form LLM outputs may be used directly.

---

### **Fault Tolerance**

The system must never crash due to external API failure.

Fallback behavior must return valid HOLD decisions if necessary.

---

## **5\. Agent Architecture**

The MVP system consists of three agents:

---

### **5.1 Scout Agent (News Retrieval Agent)**

#### **Objective**

Retrieve real-time qualitative context about players.

#### **Inputs**

List of player names.

#### **Processing Steps**

1. Execute DuckDuckGo searches for each player concurrently using asyncio:

   * "{player} injury update"

   * "{player} minutes restriction"

   * "{player} fantasy outlook"

2. Extract snippets and URLs.

3. Perform a single batch Groq LLM call using llama3-8b-8192 to summarize all player snippets into concise fantasy impact summaries.

4. Parse summaries into structured NewsResult objects.

#### **Output Schema**

{

 "player": "string",

 "news\_summary": "string",

 "source\_urls": \["string"\],

 "retrieved\_at\_iso": "string"

}

#### **Failure Handling**

If search fails:

* Return empty summary

* Return empty URLs

* Maintain valid schema

If LLM fails:

* Use raw snippet fallback summary

---

### **5.2 Analyst Agent (Statistical Retrieval Agent)**

#### **Objective**

Retrieve structured performance metrics.

#### **Inputs**

List of player names.

#### **Processing Steps**

Use ThreadPoolExecutor to fetch player statistics concurrently.

For each player:

* Resolve player ID using nba\_api

* Retrieve game logs

* Calculate:

  * Season average points

  * Last 5 game average points

  * Trend classification (UP, DOWN, STABLE)

#### **Output Schema**

{

 "player": "string",

 "season\_avg\_points": number,

 "last5\_avg\_points": number,

 "trend": "UP | DOWN | STABLE",

 "data\_source": "nba\_api | fallback | unavailable"

}

#### **Failure Handling**

If API fails:  
 Return valid StatsResult with unavailable source.

---

### **5.3 GM Agent (Decision Engine)**

#### **Objective**

Synthesize signals and generate structured recommendation.

#### **Inputs**

* NewsResult list

* StatsResult list

* Original query

#### **Processing Steps**

Call Groq llama3-70b-8192.

Prompt must enforce strict JSON output.

Decision priority:

1. Injury or availability risk

2. Minutes restrictions

3. Recent performance trend

4. Season performance

#### **Output Schema**

{

 "action": "START | SIT | TRADE\_ACCEPT | TRADE\_REJECT | HOLD",

 "confidence": 0.0-1.0,

 "reasoning": "string",

 "sources": \["string"\]

}

#### **Failure Handling**

If output invalid:

Retry once.

If still invalid:

Return HOLD decision.

---

## **6\. Entity Extraction Component**

#### **Objective**

Extract player entities from user query.

#### **Primary Method**

Deterministic parsing using regex and alias mapping.

#### **Fallback Method**

LLM fallback using Groq llama3-8b-8192 with strict JSON output mode.

Output must be:

\["LeBron James", "Stephen Curry"\]

Validated with Pydantic.

---

## **7\. LangGraph Workflow Specification**

The workflow must implement fan-out/fan-in parallel execution.

Graph structure:

entity\_extraction\_node

       ↓

  parallel fan-out

  scout\_agent   analyst\_agent

       ↓

  merge\_results\_node

       ↓

    gm\_agent\_node

       ↓

    output\_node

State must be stored using AgentState schema.

Final output must be DecisionEnvelope.

---

## **8\. Web Application Requirements**

Frontend framework: Streamlit

Features required:

* Text input query box

* Submit button

* Loading indicator

* Decision display

* Confidence display

* Reasoning display

* Source link display

* Expandable news and stats sections

---

## **9\. Performance Requirements**

Maximum end-to-end response time:

5 seconds.

Expected breakdown:

* Scout Agent: 2–3 seconds

* Analyst Agent: \<1 second

* GM Agent: \<1 second

Parallel architecture required to meet this target.

---

## **10\. CI/CD Requirements**

GitHub Actions pipeline must:

* Run Ruff linting

* Run pytest tests

* Validate code on push

Pipeline failure must block deployment.

---

## **11\. Deployment Requirements**

Deployment platform:

Streamlit Cloud

Application must:

* Load Groq API key from environment or secrets

* Be accessible via public URL

* Run without local dependencies

---

## **12\. Folder Structure**

Required structure:

pro-scout-ai/

agents/

graph/

frontend/

tests/

.github/workflows/

.streamlit/

docs/

requirements.txt

README.md

---

## **13\. Success Criteria**

MVP complete when:

* Application deployed publicly

* All agents operational

* Structured decisions generated

* Response time under 5 seconds

* CI/CD pipeline operational

* No crashes from external API failures

---

## **14\. Definition of Done**

System meets all requirements above and produces valid decision outputs for sample queries such as:

* "Start Jayson Tatum?"

* "Trade LeBron James for Stephen Curry?"

---

## **15\. Resume-Level Outcome**

This MVP demonstrates:

* Multi-agent system orchestration

* Parallel distributed processing

* LLM-based decision synthesis

* Real-time data ingestion

* Cloud deployment

* CI/CD automation

