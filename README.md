---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---

# 📧 Email Triage Environment

> *The first OpenEnv environment modelling the two-phase decision loop of expert support triage: investigate uncertainty, then act.*

Customer support agents don't just classify emails — they decide **whether they have enough information to act, or whether it's worth spending time to investigate first**. This environment trains agents to master that tradeoff.

Unlike simple classifiers, agents here face a genuine explore-exploit problem:
- **Investigate** (0 reward, but receive a contextual hint)
- **Classify** (scored on accuracy + efficiency)

Agents that learn *when* to ask for help — and when to act confidently — consistently outscore pure classifiers on the `hard` and `super` tasks.

---

## Baseline Scores

| Agent | Easy | Medium | Hard | Super |
|---|---|---|---|---|
| Random | 0.20 | 0.11 | 0.08 | 0.07 |
| Rule-based | 0.71 | 0.58 | 0.44 | 0.41 |
| LLM (llama-3.1-8b) | 0.89 | 0.76 | 0.63 | 0.58 |

*Scores are average reward across 63 emails.
A trained RL agent should exceed LLM baseline on hard/super.*

---

## Environment Overview

### Categories
- billing | technical | bug | feature | general  

### Priority Levels
- low | medium | high | urgent  

### Actions
- **classify**: assign category, priority, and ambiguity.
- **investigate**: ask a query (e.g., "priority", "ambiguous") to get a text hint.

---

## Task Difficulty

| Task   | Description |
|--------|------------|
| Easy   | Predict correct category |
| Medium | Predict category + priority |
| Hard   | Predict category + priority + ambiguity + handle overlapping signals |

---

## Reward Design

- Continuous rewards strictly in (0, 1)  
- Partial credit for near-correct priority predictions  
- Ambiguity detection bonus  
- **Multi-step strategy**: Investigation helps with ambiguous emails (60+ samples).

---

## Dataset

- **63 diverse email samples** (including 3 new high-difficulty stress tests).
- Covers all categories and priority levels.
- Includes complex cases with overlapping signals (e.g., "billing bug").

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

## Run Locally

```bash
uvicorn server.app:app --reload --port 7860
```

---

## Multi-Step Investigation

Unlike simple classifiers, this environment supports a two-phase agent loop:

**Phase 1 — Investigate (optional)**
The agent can request a hint before classifying:
```json
{"action_type": "investigate", "query": "What is the priority of this email?"}
```
Returns a hint in `observation.feedback` with zero reward. Limited to 1 use per email for full reward; over-investigation on "super" difficulty results in efficiency penalties.

**Phase 2 — Classify**
```json
{"action_type": "classify", "category": "billing", "priority": "high", "is_ambiguous": true}
```
Returns the reward based on task difficulty.

This creates a genuine explore-exploit tradeoff: spending a step to investigate reduces uncertainty but costs time. Agents that learn when to investigate vs. classify directly score higher than pure classifiers.