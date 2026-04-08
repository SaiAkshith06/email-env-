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


# ---------------- SAFE SCORE ----------------
def safe_score(score):
    EPS = 1e-6
    try:
        score = float(score)
    except:
        return EPS

    if score <= 0:
        return EPS
    if score >= 1:
        return 1 - EPS

    return score


# ---------------- RULE-BASED ----------------
def rule_based(obs):
    text = (obs.get("subject", "") + " " + obs.get("body", "")).lower()

    if any(w in text for w in ["payment", "invoice", "billing", "refund"]):
        return {"category": "billing", "priority": "high", "is_ambiguous": False}

    if any(w in text for w in ["crash", "bug", "error"]):
        return {"category": "bug", "priority": "urgent", "is_ambiguous": False}

    if any(w in text for w in ["login", "account", "password"]):
        return {"category": "technical", "priority": "high", "is_ambiguous": False}

    if any(w in text for w in ["feature", "request"]):
        return {"category": "feature", "priority": "low", "is_ambiguous": False}

    return {"category": "general", "priority": "medium", "is_ambiguous": False}


# ---------------- LLM ----------------
def llm_action(obs):
    if client is None:
        return rule_based(obs)

    prompt = f"""
Classify this email.

Return ONLY JSON:
{{
  "category": "billing|bug|technical|feature|general",
  "priority": "urgent|high|medium|low",
  "is_ambiguous": true/false
}}

Subject: {obs.get('subject')}
Body: {obs.get('body')}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)
        return result

    except Exception:
        return rule_based(obs)


# ---------------- RUN TASK ----------------
def run_task(task_id: str):
    rewards = []

    try:
        obs = requests.post(
            f"{BASE_URL}/reset",
            json={"task_id": task_id, "seed": 42}
        ).json()

        for _ in range(20):
            action = llm_action(obs)

            result = requests.post(
                f"{BASE_URL}/step",
                json=action
            ).json()

            reward = safe_score(result.get("reward", 0.0))
            rewards.append(reward)

            if result.get("done", False):
                break

            obs = result.get("observation", {})

        # ---------------- FINAL NORMALIZATION ----------------
        raw_total = sum(rewards)

        # squash into (0,1)
        total = raw_total / (raw_total + 1)

        total = safe_score(total)

    except Exception:
        total = safe_score(0.0)

    return total


# ---------------- MAIN ----------------
def main():
    final_scores = {}

    for task in TASK_IDS:
        final_scores[task] = run_task(task)

    # ✅ ONLY JSON OUTPUT (VERY IMPORTANT)
    print(json.dumps(final_scores))


if __name__ == "__main__":
    main()