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

    cat_scores = {
        "billing": 0,
        "bug": 0,
        "technical": 0,
        "feature": 0,
        "general": 0,
    }

    for w in ["payment", "invoice", "billing", "refund", "charge"]:
        if w in text:
            cat_scores["billing"] += 2

    for w in ["crash", "bug", "error", "exception", "fails", "not working"]:
        if w in text:
            cat_scores["bug"] += 2

    for w in ["login", "account", "password", "auth", "signup"]:
        if w in text:
            cat_scores["technical"] += 2

    for w in ["feature", "request", "add", "improve", "enhancement"]:
        if w in text:
            cat_scores["feature"] += 2

    category = max(cat_scores, key=cat_scores.get)
    if cat_scores[category] == 0:
        category = "general"

    if any(w in text for w in ["crash", "down", "urgent", "asap"]):
        priority = "urgent"
    elif any(w in text for w in ["fail", "error", "issue"]):
        priority = "high"
    elif any(w in text for w in ["feature", "request"]):
        priority = "low"
    else:
        priority = "medium"

    is_ambiguous = any(w in text for w in [
        "not sure", "maybe", "i think", "seems"
    ])

    return {
        "category": category,
        "priority": priority,
        "is_ambiguous": is_ambiguous
    }


# ---------------- LLM ----------------
def llm_action(obs):
    prompt = f"""
Classify this email STRICTLY.

Rules:
- Billing issues → billing + high
- Bugs/crashes/errors → bug + urgent
- Feature requests → feature + low
- Login/account issues → technical + high
- Ambiguous queries → general + medium + ambiguous=true

Return ONLY JSON:
{{
  "category": "billing|bug|technical|feature|general",
  "priority": "urgent|high|medium|low",
  "is_ambiguous": true/false
}}

Email:
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

        rb = rule_based(obs)

        if result["category"] == "bug":
            result["priority"] = "urgent"
        if result["category"] == "feature":
            result["priority"] = "low"
        if result["category"] == "billing":
            result["priority"] = "high"

        if result["priority"] not in ["urgent", "high", "medium", "low"]:
            result["priority"] = rb["priority"]

        return result

    except Exception as e:
        print(f"[DEBUG] Groq failed: {e}")
        return rule_based(obs)


# ---------------- RUN TASK ----------------
def run_task(task_id: str, seed: int = 42):
    print(f"[START] task={task_id}")

    rewards = []

    try:
        obs = requests.post(
            f"{BASE_URL}/reset",
            json={"task_id": task_id, "seed": seed}
        ).json()

        for step in range(20):

            action = llm_action(obs)

            result = requests.post(
                f"{BASE_URL}/step",
                json=action
            ).json()

            raw_reward = result.get("reward", 0.0)

            # FIX APPLIED HERE
            reward = safe_score(raw_reward)

            done = result.get("done", False)

            rewards.append(reward)

            if done:
                break

            obs = result.get("observation", {})

        total = safe_score(sum(rewards))  # FIX HERE ALSO

        success = total > 0

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        total = safe_score(0.0)
        success = False
        rewards = []

    return task_id, total, rewards


# ---------------- MAIN ----------------
def main():
    all_results = {}

    for task_id in TASK_IDS:
        tid, total, rewards = run_task(task_id)
        all_results[tid] = safe_score(total)

    # 🔥 FINAL OUTPUT SAFE
    final_scores = {
        k: safe_score(v) for k, v in all_results.items()
    }

    print(json.dumps(final_scores))


if __name__ == "__main__":
    main()