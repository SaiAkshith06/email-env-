---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---

# Email Triage Environment (v3.0)

## Why Email Triage?

Customer support teams handle thousands of emails daily. Misclassifying a billing issue as technical — or missing an urgent production bug — can directly impact revenue and user trust.

This environment is now a **genuine RL task** with sequential decision-making:
- **Classify**: Take a risk and categorize immediately.
- **Investigate**: Request more context first (costs a step, but returns a valuable hint).

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