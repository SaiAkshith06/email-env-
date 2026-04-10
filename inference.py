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


# ---------------- SYSTEM PROMPT (UPGRADED) ----------------
SYSTEM_PROMPT = """
You are a senior customer support triage expert.

Follow this reasoning internally:
1. Identify intent
2. Detect strong keywords
3. Decide category
4. Decide priority
5. Check ambiguity

CATEGORIES:
billing, bug, technical, feature, general

PRIORITY:
urgent = crash/data loss/system down
high = major issue (login/payment failure)
medium = normal issue
low = feature request

AMBIGUITY:
true ONLY if unclear, vague, or mixed intent

STRICT RULES:
- bug → usually urgent
- billing → usually high
- feature → low

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
        f"Previous feedback: {obs.get('feedback','')}\n\n"
        "Classify this email accurately."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0,
            max_tokens=120
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        # ---------------- HYBRID CORRECTION ----------------
        rule = rule_based(obs)

        # fix category disagreement
        if result.get("category") not in ["billing","bug","technical","feature","general"]:
            result["category"] = rule["category"]

        # fix priority
        if result.get("priority") not in ["urgent","high","medium","low"]:
            result["priority"] = rule["priority"]

        # enforce consistency rules
        if result["category"] == "bug":
            result["priority"] = "urgent"
        elif result["category"] == "feature":
            result["priority"] = "low"
        elif result["category"] == "billing":
            result["priority"] = "high"

        # ambiguity refinement
        text_lower = (obs.get("subject","") + obs.get("body","")).lower()
        if any(w in text_lower for w in ["maybe","not sure","seems","unclear"]):
            result["is_ambiguous"] = True

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