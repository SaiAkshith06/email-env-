---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
tags:
  - openenv
  - email-triage
  - nlp
  - customer-support
  - reinforcement-learning
---

# 📧 Email Triage Environment

[![CI](https://github.com/SaiAkshith06/email-env/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/email-env/actions/workflows/ci.yml)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://saiakshith06-email-env-v2.hf.space)

> *The first OpenEnv environment modelling the two-phase decision loop of expert support triage: investigate uncertainty, then act.*

Customer support agents don't just classify emails — they decide **whether they have enough information to act, or whether it's worth spending time to investigate first**. This environment trains agents to master that tradeoff.

Unlike simple classifiers, agents here face a genuine explore-exploit problem:
- **Investigate** (0 reward, but receive a contextual hint)
- **Classify** (scored on accuracy + efficiency)

Agents that learn *when* to ask for help — and when to act confidently — consistently outscore pure classifiers on the `hard` and `super` tasks.

---

## Why Email Triage?

Customer support teams process **thousands of emails daily**. Misrouting a billing complaint to the technical team — or failing to escalate a production outage buried in a vague subject line — costs companies millions. This environment trains AI agents to handle triage with the precision of an experienced support lead:

- Correctly categorise emails across 5 domains
- Assess urgency across 4 priority levels
- Flag genuinely ambiguous cases for human review
- Learn *when* to request more context vs. act immediately

---

## Baseline Scores

| Agent | Easy | Medium | Hard | Super |
|---|---|---|---|---|
| Random | 0.20 | 0.11 | 0.08 | 0.07 |
| Rule-based | 0.71 | 0.58 | 0.44 | 0.41 |
| LLM (llama-3.1-8b) | 0.89 | 0.76 | 0.63 | 0.58 |

*Scores are average reward across 63 emails. A trained RL agent should exceed the LLM baseline on `hard` and `super` by learning optimal investigate-vs-classify timing.*

---

## Environment Overview

### Action Space

| Field | Type | Values |
|---|---|---|
| `action_type` | string | `"classify"` or `"investigate"` |
| `category` | string | `billing`, `technical`, `bug`, `feature`, `general` |
| `priority` | string | `low`, `medium`, `high`, `urgent` |
| `is_ambiguous` | bool | `true` / `false` |
| `query` | string | Natural language question (investigate only) |

### Observation Space

| Field | Type | Description |
|---|---|---|
| `email_id` | string | Unique email identifier |
| `subject` | string | Email subject line |
| `body` | string | Email body text |
| `sender` | string | Sender email address |
| `sender_tier` | string | `enterprise`, `pro`, `free`, `unknown` |
| `step_count` | int | Current step in episode |
| `episode_progress` | float | Fraction of episode completed (0.0–1.0) |
| `feedback` | string | Hint (after investigate) or grading feedback (after classify) |
| `investigate_used` | bool | Whether investigate was used on current email |
| `prev_rewards` | list[float] | Reward history for current episode |
| `done` | bool | Episode completion flag |

---

## Task Difficulty

| Task | Description | Grading |
|---|---|---|
| `easy` | Predict correct category only | 1.0 for correct, 0.0 for wrong |
| `medium` | Category + priority | 0.5 each; partial credit for off-by-one priority |
| `hard` | Category + priority + ambiguity + overlapping signals | Weighted rubric with keyword reasoning bonus |
| `super` | Full classification with efficiency scoring | Hard score + bonus for no investigation, penalty for over-investigating |

**Hard and super tasks use only the most challenging emails** — those with overlapping signals, mixed intent, and domain ambiguity.

---

## Reward Design

- All rewards are continuous in **(0, 1)** — never exactly 0 or 1
- **Partial credit** for off-by-one priority predictions
- **Ambiguity detection bonus** — correctly flagging unclear emails
- **Keyword reasoning bonus** — reward consistency between email signals and classification
- **Efficiency bonus** (super only) — correct classification without investigating
- **Over-investigation penalty** (super only) — using investigate more than once per email

---

## Dataset

- **63 diverse email samples** across all 5 categories and 4 priority levels
- **All 20 category × priority combinations** covered
- **14 ambiguous emails** spread across categories
- **15 hard-difficulty emails** with overlapping signals (e.g., billing issue that mentions a bug, technical problem that could be billing)
- Average body length: 102 characters
- Seed-based shuffling ensures reproducibility while preventing memorisation

---

## Quick Start

### 1. Reset the Environment

```bash
curl -X POST https://saiakshith06-email-env-v2.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "hard", "seed": 42}'
```

### 2. Investigate (Optional)

```bash
curl -X POST https://saiakshith06-email-env-v2.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{
       "action_type": "investigate",
       "query": "Is this ambiguous?"
     }'
```

### 3. Classify

```bash
curl -X POST https://saiakshith06-email-env-v2.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{
       "action_type": "classify",
       "category": "billing",
       "priority": "high",
       "is_ambiguous": true
     }'
```

### 4. Check Metrics

```bash
curl https://saiakshith06-email-env-v2.hf.space/metrics
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check + version |
| GET | `/health` | Liveness probe |
| GET | `/tasks` | List all available tasks |
| GET | `/metrics` | Current episode statistics |
| GET | `/state` | Current episode state |
| POST | `/reset` | Start new episode |
| POST | `/step` | Submit action, receive observation + reward |

---

## Multi-Step Investigation

Unlike simple classifiers, this environment supports a two-phase agent loop:

**Phase 1 — Investigate (optional)**
```json
{"action_type": "investigate", "query": "What is the priority of this email?"}
```
Returns a hint in `observation.feedback` with **zero reward**. Limited to 1 investigation per email for full reward; over-investigation on `super` results in efficiency penalties.

**Phase 2 — Classify**
```json
{
  "action_type": "classify",
  "category": "billing",
  "priority": "high",
  "is_ambiguous": true
}
```
Returns reward based on task difficulty.

This creates a genuine **explore-exploit tradeoff**: spending a step to investigate reduces uncertainty but costs time. Agents that learn when to investigate vs. classify directly score higher than pure classifiers.

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/email-env
cd email-env
pip install -e ".[dev]"
uvicorn server.app:app --reload --port 7860
```

Run the inference script:
```bash
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

Run tests:
```bash
pytest tests/ -v
```

---

## Project Structure
email-env/
├── inference.py          # Agent inference script (required by OpenEnv)
├── models.py             # Action, Observation, State models
├── client.py             # Environment client
├── openenv.yaml          # OpenEnv configuration
├── Dockerfile            # Container definition
├── server/
│   ├── app.py            # FastAPI server
│   ├── email_env_environment.py  # Core environment logic
│   ├── grader.py         # Reward computation
│   ├── tasks.py          # Task definitions
│   └── data.json         # Email dataset (63 samples)
└── tests/
└── test_grader.py    # Grader unit tests

---

## OpenEnv Compatibility

This environment fully implements the OpenEnv spec:
- Inherits from `openenv.core.env_server.interfaces.Environment`
- `EmailAction` inherits from `openenv Action`
- `EmailObservation` inherits from `openenv Observation`
- Exposes standard `/reset`, `/step`, `/tasks`, `/health` endpoints
- Compatible with `openenv validate`
