import requests
import os
import json
from openai import OpenAI

# Use ENV_BASE_URL for the environment server
BASE_URL = os.getenv("ENV_BASE_URL", os.getenv("HOST", "http://localhost:8000"))

_hf_token = os.getenv("HF_TOKEN")  # read from env if expected by spec

try:
    # Use API_BASE_URL and API_KEY injected by evaluating proxy
    client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", "dummy_key"))
    )
except Exception:
    client = None


MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

TASK_IDS = ["easy", "medium", "hard"]


# ---------------- RULE-BASED FALLBACK ----------------
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


# ---------------- GROQ LLM ----------------
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

        # POST-CORRECTION
        rb = rule_based(obs)

        # fix wrong priorities
        if result["category"] == "bug":
            result["priority"] = "urgent"

        if result["category"] == "feature":
            result["priority"] = "low"

        if result["category"] == "billing":
            result["priority"] = "high"

        # fallback correction if mismatch
        if result["priority"] not in ["urgent", "high", "medium", "low"]:
            result["priority"] = rb["priority"]

        return result

    except Exception as e:
        print(f"[DEBUG] Groq failed: {e}")
        return rule_based(obs)


# ---------------- RUN SINGLE TASK ----------------
def run_task(task_id: str, seed: int = 42):
    print(f"[START] task={task_id} env=email_env_v2 model={MODEL_NAME} seed={seed}")

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

            reward = result.get("reward", 0.0)
            done = result.get("done", False)

            rewards.append(reward)

            action_str = json.dumps(action, separators=(',', ':'))
            print(f"[STEP] step={step+1} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

            if done:
                break

            obs = result.get("observation", {})

        total = sum(rewards)
        success = total > 0

    except Exception as e:
        print(f"[STEP] step=0 action=noop reward=0.00 done=true error={str(e)}")
        total = 0.0
        success = False
        rewards = []

    print(f"[END] task={task_id} success={str(success).lower()} steps={len(rewards)} total_reward={total:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}")
    return task_id, total, rewards


# ---------------- MAIN ----------------
def main():
    print("=" * 60)
    print("BASELINE INFERENCE — Email Env (all 3 tasks)")
    print("=" * 60)

    all_results = {}

    for task_id in TASK_IDS:
        tid, total, rewards = run_task(task_id, seed=42)
        all_results[tid] = {"total_reward": total, "steps": len(rewards)}

    print("\n" + "=" * 60)
    print("SUMMARY — Baseline Scores")
    print("=" * 60)
    for tid in TASK_IDS:
        r = all_results[tid]
        print(f"  {tid:>8s}:  total_reward={r['total_reward']:.2f}  steps={r['steps']}")
    print("=" * 60)


if __name__ == "__main__":
    main()