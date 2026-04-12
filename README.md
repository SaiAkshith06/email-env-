---
title: Adaptive Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
tags:
  - openenv
  - adaptive-triage
  - nlp
  - customer-support
  - reinforcement-learning
---

# 📧 Email Triage Environment

> *An RL environment for training agents that operate under asymmetric information - deciding when to act on incomplete data vs. when the cost of investigation is worth the reduction in uncertainty.*

Most classification environments assume the agent has full information. Real-world decision making doesn't work that way. A support engineer reading a vague ticket must decide: **act now with 70% confidence, or spend time gathering context?**

This environment operationalises that tradeoff. Agents face 63 support scenarios with varying information quality and must learn an **optimal investigation policy** - not just a classification policy.

Agents that learn *when* to ask for help — and when to act confidently — consistently outscore pure classifiers on the `hard` and `super` tasks.

---

## Baseline Scores

| Agent | Easy | Medium | Hard | Adaptive |
|---|---|---|---|---|
| Random | 0.20 | 0.11 | 0.08 | 0.07 |
| Rule-based | 0.71 | 0.58 | 0.44 | 0.41 |
| LLM (llama-3.1-8b) | 0.89 | 0.76 | 0.63 | 0.58 |

*Scores are average reward across 63 emails.
A trained RL agent should exceed LLM baseline on hard/super.*

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

## Optimal Investigation Policy

Unlike simple classifiers, this environment supports a two-phase agent loop:

**Phase 1 — Investigate (optional)**
```json
{"action_type": "investigate", "query": "What is the priority of this email?"}
```
Returns a hint in `observation.feedback` with zero reward. Limited to 1 use per email for full reward; over-investigation on "super" difficulty results in efficiency penalties.

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

This creates a genuine explore-exploit tradeoff: spending a step to investigate reduces uncertainty but costs time. Agents that learn when to investigate vs. classify directly score higher than pure classifiers.