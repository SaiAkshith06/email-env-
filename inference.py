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


# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are an expert customer support triage agent.

Classify emails into:
- billing (payments, invoices, refunds)
- bug (crashes, errors, failures)
- technical (login, account, setup issues)
- feature (feature requests, improvements)
- general (unclear or mixed intent)

Priority:
- urgent → system down, crash, critical failure
- high → major issue
- medium → normal issue
- low → feature request

Ambiguity:
- true ONLY if unclear or mixed intent

Return ONLY JSON:
{"category":"...","priority":"...","is_ambiguous":false}
"""


# ---------------- SAFE SCORE ----------------
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


# ---------------- RULE-BASED ----------------
def rule_based(obs):
    text = (obs.get("subject", "") + " " + obs.get("body", "")).lower()

    if any(w in text for w in ["payment", "invoice", "billing", "refund"]):
        return {"category": "billing", "priority": "high", "is_ambiguous": False}

    if any(w in text for w in ["crash", "error", "bug", "fail"]):
        return {"category": "bug", "priority": "urgent", "is_ambiguous": False}

    if any(w in text for w in ["login", "account", "password", "access"]):
        return {"category": "technical", "priority": "high", "is_ambiguous": False}

    if any(w in text for w in ["feature", "request", "improve"]):
        return {"category": "feature", "priority": "low", "is_ambiguous": False}

    ambiguous = any(w in text for w in [
        "not sure", "maybe", "i think", "seems", "unclear"
    ])

    return {"category": "general", "priority": "medium", "is_ambiguous": ambiguous}


# ---------------- LLM ACTION ----------------
def get_llm_action(obs):
    if client is None:
        return rule_based(obs)

    user_msg = (
        f"Subject: {obs.get('subject','')}\n"
        f"From: {obs.get('sender','')}\n"
        f"Body: {obs.get('body','')}\n"
        f"Previous feedback: {obs.get('feedback','')}\n"
        "Classify the email. Return ONLY JSON."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0,
            max_tokens=100
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        # ---------------- HYBRID CORRECTION ----------------
        rule = rule_based(obs)

        if result.get("category") != rule["category"]:
            result["category"] = rule["category"]

        if result.get("priority") not in ["urgent", "high", "medium", "low"]:
            result["priority"] = rule["priority"]

        # consistency rules
        if result["category"] == "bug":
            result["priority"] = "urgent"
        if result["category"] == "feature":
            result["priority"] = "low"
        if result["category"] == "billing":
            result["priority"] = "high"

        return result

    except Exception:
        return rule_based(obs)


# ---------------- RUN TASK ----------------
def run_task(task_id):
    print(f"[START] task={task_id}", flush=True)

    rewards = []

    obs = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id, "seed": 42}
    ).json()

    for step in range(20):
        action = get_llm_action(obs)

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

    # squash into (0,1)
    score = raw_total / (raw_total + 1)
    score = safe_score(score)

    print(
        f"[END] task={task_id} score={score:.6f} steps={len(rewards)}",
        flush=True
    )


# ---------------- MAIN ----------------
def main():
    for task in TASK_IDS:
        run_task(task)


if __name__ == "__main__":
    main()