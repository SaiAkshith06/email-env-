# Email Triage Environment

CI

## Why Email Triage?

Customer support teams handle thousands of emails daily. Misclassifying a billing issue as technical — or missing an urgent production bug — can cost companies revenue and customer trust.

This environment simulates real-world email triage:
- Categorising emails correctly
- Assigning proper priority
- Detecting ambiguous cases for human review

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

| Task | Description |
|------|------------|
| Easy | Category classification |
| Medium | Category + Priority |
| Hard | Category + Priority + Ambiguity + Reasoning |

---

## Reward Design

- Weighted scoring for category, priority, ambiguity
- Partial credit for near-correct answers
- Bonus for keyword-based reasoning
- All scores strictly in (0,1)

---

## Architecture

Email Input → LLM + Rule Hybrid → Feedback → Grader → Reward

---

## Try It Locally

uvicorn server.app:app --reload

---

## Features

- Hybrid LLM + rule-based inference
- Feedback-driven learning loop
- Deterministic grading system
- Realistic dataset with ambiguity
- OpenEnv compatible

---

## Dataset

- 40+ diverse emails
- All categories and priorities covered
- Includes ambiguous and edge cases

---

## Future Improvements

- Larger dataset (100+ emails)
- Multi-language support
- Real-time agent training

---