import requests
import os
import json
from openai import OpenAI

BASE_URL = os.getenv("ENV_BASE_URL", os.getenv("HOST", "http://localhost:8000"))

try:
    client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", "dummy_key"))
    )
except Exception:
    client = None

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

TASK_IDS = ["easy", "medium", "hard"]


def safe_score(x):
    EPS = 1e-6
    try:
        x = float(x)
    except:
        return EPS
    if x <= 0:
        return EPS
    if x >= 1:
        return 1 - EPS
    return x


def rule_based(obs):
    text = (obs.get("subject", "") + " " + obs.get("body", "")).lower()

    if "payment" in text or "billing" in text:
        return {"category": "billing", "priority": "high", "is_ambiguous": False}
    if "error" in text or "bug" in text:
        return {"category": "bug", "priority": "urgent", "is_ambiguous": False}
    if "login" in text:
        return {"category": "technical", "priority": "high", "is_ambiguous": False}
    if "feature" in text:
        return {"category": "feature", "priority": "low", "is_ambiguous": False}

    return {"category": "general", "priority": "medium", "is_ambiguous": False}


def llm_action(obs):
    if client is None:
        return rule_based(obs)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": str(obs)}],
            temperature=0
        )
        text = response.choices[0].message.content.strip()
        return json.loads(text)
    except:
        return rule_based(obs)


def run_task(task_id):
    print(f"[START] task={task_id}", flush=True)

    rewards = []

    obs = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id, "seed": 42}
    ).json()

    for step in range(20):
        action = llm_action(obs)

        result = requests.post(
            f"{BASE_URL}/step",
            json=action
        ).json()

        reward = safe_score(result.get("reward", 0.0))
        done = result.get("done", False)

        rewards.append(reward)

        print(
            f"[STEP] task={task_id} step={step+1} reward={reward:.6f} done={str(done).lower()}",
            flush=True
        )

        if done:
            break

        obs = result.get("observation", {})

    raw_total = sum(rewards)

    # squash to (0,1)
    score = raw_total / (raw_total + 1)
    score = safe_score(score)

    print(
        f"[END] task={task_id} score={score:.6f} steps={len(rewards)}",
        flush=True
    )


def main():
    for task in TASK_IDS:
        run_task(task)


if __name__ == "__main__":
    main()