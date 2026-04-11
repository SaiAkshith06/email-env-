---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Email Triage Environment

## Why Email Triage?

Customer support teams handle thousands of emails daily. Misclassifying a billing issue as technical — or missing an urgent production bug — can directly impact revenue and user trust.

This environment simulates real-world email triage by requiring an agent to:
- Categorise emails correctly  
- Assign appropriate priority  
- Detect ambiguous cases for human escalation  

---

## Environment Overview

### Categories
- billing  
- technical  
- bug  
- feature  
- general  

### Priority Levels
- low  
- medium  
- high  
- urgent  

---

## Task Difficulty

| Task   | Description |
|--------|------------|
| Easy   | Predict correct category |
| Medium | Predict category + priority |
| Hard   | Predict category + priority + ambiguity |

---

## Reward Design

- Continuous rewards strictly in (0, 1)  
- Partial credit for near-correct predictions  
- Priority proximity scoring  
- Ambiguity detection bonus  
- Deterministic and reproducible grading  

---

## Architecture

Email Input → Hybrid Agent (LLM + Rules) → Environment → Feedback → Grader → Reward  

---

## Features

- Hybrid LLM + rule-based inference  
- Feedback-aware environment loop  
- Deterministic reward computation  
- Support for easy / medium / hard tasks  
- OpenEnv-compatible design  

---

## Dataset

- 60 diverse email samples  
- Covers all categories and priority levels  
- Includes ambiguous and edge cases  
- Designed to simulate real-world support scenarios  

---

## Run Locally

```bash
uvicorn server.app:app --reload
```

## Try It Now

```bash
curl -X POST https://saiakshith06-email-env-v2.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "easy", "seed": 42}'
```