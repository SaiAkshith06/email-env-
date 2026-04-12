import requests
import os
import json
from openai import OpenAI

BASE_URL = os.getenv("ENV_BASE_URL", os.getenv("HOST", "http://localhost:7860"))

try:
    client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", "dummy_key"))
    )
except Exception:
    client = None

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

TASK_IDS = ["easy", "medium", "hard", "adaptive"]

BENCHMARK = "email_triage_env"
SUCCESS_THRESHOLD = 0.6


SYSTEM_PROMPT = """
You are a senior customer support triage expert.

Classify emails into:
billing, bug, technical, feature, general

Priority:
urgent, high, medium, low

Return ONLY JSON:
{"category":"...","priority":"...","is_ambiguous":false}
"""


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

    if any(w in text for w in ["payment", "invoice", "billing", "refund"]):
        return {"action_type": "classify", "category": "billing", "priority": "high", "is_ambiguous": False}

    if any(w in text for w in ["crash", "error", "bug", "fail"]):
        return {"action_type": "classify", "category": "bug", "priority": "urgent", "is_ambiguous": False}

    if any(w in text for w in ["login", "account", "password", "access"]):
        return {"action_type": "classify", "category": "technical", "priority": "high", "is_ambiguous": False}

    if any(w in text for w in ["feature", "request", "improve"]):
        return {"action_type": "classify", "category": "feature", "priority": "low", "is_ambiguous": False}

    ambiguous = any(w in text for w in ["not sure", "maybe", "seems", "unclear"])

    return {"action_type": "classify", "category": "general", "priority": "medium", "is_ambiguous": ambiguous}


def validate_action(result, obs):
    rule = rule_based(obs)

    valid_categories = ["billing", "bug", "technical", "feature", "general"]
    valid_priorities = ["urgent", "high", "medium", "low"]

    if result.get("category") not in valid_categories:
        result["category"] = rule["category"]

    if result.get("priority") not in valid_priorities:
        result["priority"] = rule["priority"]

    if "is_ambiguous" not in result:
        result["is_ambiguous"] = rule["is_ambiguous"]

    return result


def get_llm_action(obs):
    feedback = obs.get("feedback", "")
    investigate_used = obs.get("investigate_used", False)

    # If we haven't investigated yet and the email looks tricky, investigate first
    body = obs.get("body", "").lower()
    subject = obs.get("subject", "").lower()
    looks_ambiguous = any(w in body + subject for w in ["not sure", "maybe", "unclear", "seems", "think", "either"])

    if looks_ambiguous and not investigate_used and client is not None:
        return {
            "action_type": "investigate",
            "query": "Is this ambiguous? What is the category?",
            "category": None,
            "priority": None,
            "is_ambiguous": False
        }

    # Otherwise classify normally
    if client is None:
        return rule_based(obs)

    user_msg = (
        f"Subject: {obs.get('subject','')}\n"
        f"Body: {obs.get('body','')}\n"
        f"Feedback: {obs.get('feedback','')}\n"
        "Return classification JSON."
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
        result = validate_action(result, obs)
        result["action_type"] = "classify"

        return result

    except Exception:
        return rule_based(obs)


def run_task(task_id):
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards = []

    obs = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id, "seed": 42}
    ).json()

    for step in range(1, 61):
        action = get_llm_action(obs)
        error = None

        try:
            result = requests.post(
                f"{BASE_URL}/step",
                json=action
            ).json()
        except Exception as e:
            error = str(e)
            break

        reward = safe_score(result.get("reward", 0.0))
        done = result.get("done", False)

        rewards.append(reward)

        action_str = json.dumps(action, separators=(",", ":"))

        print(
            f"[STEP] step={step} action={action_str} reward={reward:.4f} done={str(done).lower()} error={error or 'null'}",
            flush=True
        )

        if done:
            break

        obs = result.get("observation", {})

    score = sum(rewards) / len(rewards) if rewards else 0.0
    success = score >= SUCCESS_THRESHOLD
    rewards_str = ",".join([f"{r:.4f}" for r in rewards])

    print(
        f"[END] task={task_id} success={str(success).lower()} steps={len(rewards)} score={score:.4f} rewards={rewards_str}",
        flush=True
    )


def main():
    for task in TASK_IDS:
        run_task(task)


if __name__ == "__main__":
    main()