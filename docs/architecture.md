# Architecture — Healthcare Monitoring AI Agent (Track B)

## System overview

```
                         ┌──────────────┐
                         │  supervisor  │   keyword-based intent routing
                         └──────┬───────┘
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
      ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
      │ medical_info  │ │  medication   │ │   fitness     │
      │ (RAG-grounded)│ │  (SQLite CRUD)│ │  (SQLite CRUD)│
      └──────┬────────┘ └──────┬────────┘ └──────┬────────┘
             │ tool loop        │ tool loop        │ tool loop
             ▼                  ▼                  ▼
      medical_tools      medication_tools      fitness_tools
             └──────────────────┴──────────────────┘
                                 ▼
                          ┌─────────────┐
                          │  finalize   │  risk detection + report gen
                          └──────┬──────┘
                                 ▼
                                END
```

Each domain agent is a small LangGraph subgraph: an `agent_node` (Groq LLM
with tools bound) and a `ToolNode` that executes whatever the LLM asks for,
looping until the LLM returns a final answer with no more tool calls. All
three converge on a shared `finalize` node before the graph ends.

## Why keyword routing instead of an LLM router

A fourth LLM call just to pick a branch adds latency, cost, and a new
hallucination surface — in a health app, misrouting "did I take my
metformin" to the fitness agent is a worse failure mode than a rule-based
router occasionally guessing a borderline case wrong. `App/agents/supervisor.py`
counts domain keyword hits and defaults to `medical_info` (the safest branch,
since it only retrieves and answers — it never logs or mutates data) when a
message is ambiguous.

## Why SQLite instead of Postgres/Redis

The brief's Track B stack calls for Postgres + Redis; this build uses SQLite
instead because it satisfies the same requirement — a real relational
schema for a multi-table health record lifecycle — while staying runnable
with zero external infrastructure (no Docker, no hosted DB, no connection
strings). The `App/db/database.py` module is a thin, swappable data-access
layer: every function takes a `patient_id` and returns/accepts plain data, so
swapping the `sqlite3` connection for `psycopg2` later is a localized change,
not a rewrite of the agents or tools.

## Why a rule-based + linear-regression risk engine instead of a trained ML model

`App/tools/analytics_tools.py` flags risk using two techniques, both honestly
scoped:
1. **Threshold rules** — adherence below 70%, 3+ consecutive missed doses,
   average sleep below ~5.5h.
2. **Trend detection via linear regression** (`numpy.polyfit`, degree 1) over
   the last 7 logged fitness entries, to catch a declining steps trend a
   single-day threshold would miss.

A trained model (e.g. a classifier predicting adherence risk) needs
historical labeled outcomes this project has no access to. Claiming one
without that data would be the kind of "invented capability" the course
explicitly warns against. Regression-based trend detection is real,
explainable statistics — a legitimate lightweight stand-in for the
"predictive health analytics" milestone rather than an inflated claim.

## Data privacy & ethical handling

- The SQLite file (`data/health_data.db`) is gitignored and never committed.
- This is a **development/demo database** — no real patient data should ever
  be entered into it. Sample/mock patients only.
- The medical-info agent is explicitly instructed to never diagnose, to
  always ground answers in the local `data/documents/` knowledge base via
  RAG, and to include a disclaimer.
- The medication agent never tells a patient to start, stop, or change a
  dose — interaction findings are surfaced as "confirm with a pharmacist or
  doctor," not medical instructions.
- `check_drug_interactions` uses a small illustrative reference list
  (`KNOWN_INTERACTIONS` in `App/tools/medication_tools.py`), explicitly *not*
  presented as an exhaustive clinical database.
- Every generated health report ends with a "not a substitute for
  professional medical advice" disclaimer.

## What's implemented vs. what's explicitly out of scope for this pass

**Implemented:** 3-agent LangGraph workflow with supervisor routing, 7-table
SQLite schema, medication CRUD + adherence + basic interaction checks,
fitness logging + regression-based trend summaries, goal tracking, rule +
trend-based alerting, aggregated health report generation/export, and a
Streamlit dashboard (chat + medications + fitness charts + goals + alerts +
report download).

**Out of scope for this fast pass** (documented here rather than silently
dropped, per the brief's Track B checklist):
- Multi-service deployment — this is still a single-process app; role-based
  access is implemented (see below), but it isn't yet deployed anywhere.
- A trained ML model for risk prediction (see above — regression trend
  detection is the honest current substitute).
- Voice queries / image-based pill identification (bonus features, not
  attempted here).
- HIPAA-compliant infrastructure (encryption at rest, audit logs) — the
  privacy posture here is "safe for a local dev demo with mock data," not a
  compliance-audited system.

## Role-based access (Patient / Doctor / Caregiver)

`App/auth.py` implements a username + role login, gating what the Streamlit
dashboard shows and allows rather than providing a real security boundary:

- **Patient** — full read/write on their own linked record: chat, add/log
  medications, log fitness, set goals, view/download reports.
- **Doctor** — can switch between any patient's record from a dropdown, and
  can use the chat agent, but every logging action (dose taken/missed, fitness
  entry, goal creation) is hidden — a doctor reviews, they don't log on the
  patient's behalf.
- **Caregiver** — linked to one patient at signup, view-only: alerts,
  adherence rate, fitness charts, goals, and the downloadable report. No chat
  tab and no logging controls, matching the brief's "caregiver notifications"
  framing rather than full account access.

There are deliberately no passwords, sessions, or hashing — a username maps
to a role and (for patient/caregiver) a linked `patient_id` in the `users`
table, and returning to the same username re-establishes the same session
state. This is the right scope for demonstrating role-gated permissions in a
student project; a real deployment would need actual authentication
(password hashing, session tokens, HTTPS) before handling anything beyond
mock data.
