---
title: Email Env Environment Server
emoji: 📧
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Email Triage Environment

An OpenEnv environment that simulates **email triage** — a real-world customer support task where an AI agent must classify incoming support emails by category, priority, and ambiguity.

## Motivation

Customer support teams process hundreds of emails daily. Correctly triaging emails (routing to the right team, assigning urgency) is crucial for fast resolution times. This environment lets agents learn to automate that process.

## Quick Start

```python
from email_env import EmailEnv
from email_env.models import EmailAction, Category, Priority

# Connect to the deployed environment
env = EmailEnv(base_url="https://saiakshith06-email-env-v2.hf.space")

# Reset with a task and seed
result = env.reset(task_id="easy", seed=42)
print(f"Email: {result.observation.subject}")
print(f"Body: {result.observation.body}")

# Classify the email
action = EmailAction(
    category=Category.BILLING,
    priority=Priority.HIGH,
    is_ambiguous=False
)
result = env.step(action)
print(f"Reward: {result.reward}")
print(f"Done: {result.done}")
```

## Action Space

**EmailAction** — the agent's classification decision:

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `category` | `Category` (enum) | `billing`, `technical`, `general`, `bug`, `feature` | Email topic category |
| `priority` | `Priority` (enum) | `low`, `medium`, `high`, `urgent` | Urgency level |
| `is_ambiguous` | `bool` | `true` / `false` | Whether the email is unclear or vague |

## Observation Space

**EmailObservation** — what the agent sees each step:

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | `str` | Unique email identifier |
| `subject` | `str` | Email subject line |
| `body` | `str` | Email body text |
| `sender` | `str` | Sender address |
| `step_count` | `int` | Current step number |
| `done` | `bool` | Whether the episode is finished |
| `task_id` | `str` | Current task (`easy`, `medium`, `hard`) |

## Tasks & Difficulty

| Task | Difficulty | What the Agent Must Predict | Scoring |
|------|------------|----------------------------|---------|
| `easy` | ★☆☆ | Category only | 1.0 if correct, 0.0 if wrong |
| `medium` | ★★☆ | Category + Priority | 0.5 category + 0.5 priority (0.25 partial credit for ±1 priority) |
| `hard` | ★★★ | Category + Priority + Ambiguity | 0.4 category + 0.4 priority + 0.2 ambiguity (−0.1 penalty for false positive, clamped to [0, 1]) |

### Why the hard task is hard:
- Ambiguity detection requires understanding nuance ("I think", "maybe", "not sure")
- Priority assignment requires reasoning about urgency in context
- Some emails have overlapping signals (e.g., a billing email mentioning a "bug" in charges)

## Reward Design

- **Partial credit**: The medium and hard tasks give partial rewards, not just binary 0/1
- **Progressive signal**: Each step immediately returns a reward — no sparse end-of-episode signal
- **Penalty for over-flagging**: Hard task penalizes always guessing `is_ambiguous=True`
- **Score range**: All graders return scores in [0.0, 1.0]

## Baseline Scores

Baseline inference using Groq LLaMA 3.1 8B (seed=42):

| Task | Total Reward | Steps |
|------|-------------|-------|
| easy | ~18.00 | 20 |
| medium | ~16.50 | 20 |
| hard | ~14.00 | 20 |

Run the baseline yourself:

```bash
export OPENAI_API_KEY=<your-groq-api-key>
python inference.py
```

## Setup & Development

### Run locally

```bash
# Install dependencies
pip install -e .

# Start the server
uvicorn server.app:app --reload --port 8000
```

### Build & run with Docker

```bash
# Build
docker build -t email-env:latest .

# Run
docker run -p 8000:8000 email-env:latest
```

### Deploy to Hugging Face Spaces

```bash
openenv push
```

## Project Structure

```
email_env/
├── openenv.yaml              # OpenEnv manifest
├── models.py                 # EmailAction, EmailObservation, EmailState (Pydantic)
├── client.py                 # EmailEnv client (extends EnvClient)
├── inference.py              # Baseline inference script (runs all 3 tasks)
├── __init__.py               # Package exports
├── pyproject.toml            # Dependencies & metadata
└── server/
    ├── app.py                # FastAPI server (reset/step/state endpoints)
    ├── email_env_environment.py  # Core environment logic
    ├── grader.py             # Task-specific grading functions
    ├── tasks.py              # Task definitions
    ├── data.json             # Email dataset (20 emails)
    ├── Dockerfile            # Container image
    └── requirements.txt      # Server dependencies
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/reset` | Reset environment. Body: `{"task_id": "easy", "seed": 42}` |
| `POST` | `/step` | Submit action. Body: `{"category": "billing", "priority": "high", "is_ambiguous": false}` |
| `GET` | `/state` | Get current environment state |
