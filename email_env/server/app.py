# FORCE REBUILD: 2026-04-04T22:00:00 (v3.0)
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from email_env.server.email_env_environment import EmailEnvironment
from email_env.models import EmailAction

app = FastAPI()

# create env
print("--- INITIALIZING EMAIL ENVIRONMENT V2 (60 EMAILS) ---")
env = EmailEnvironment()


@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {"id": "easy", "description": "Predict category only"},
            {"id": "medium", "description": "Predict category + priority"},
            {"id": "hard", "description": "Predict category + priority + ambiguity"}
        ]
    }


# ✅ Request model for reset
class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int | None = None


@app.get("/")
def root():
    return {"message": "Email Env Running"}

@app.get("/health")
def health():
    return {"status": "ok"}


from typing import Optional

# ✅ RESET endpoint (deterministic + configurable)
@app.post("/reset")
def reset(req: dict = None):
    print(f"--- RESET REQUEST RECEIVED: {req} ---")
    
    # Extract task_id and seed from dict manually to be lenient
    task_id = "easy"
    seed = None
    
    if req:
        # Check both "task_id" and "task" (common variations)
        task_id = req.get("task_id") or req.get("task") or "easy"
        seed = req.get("seed")

    return env.reset(task_id=task_id, seed=seed)


# ✅ STEP endpoint (safe handling)
@app.post("/step")
def step(action: EmailAction):

    # 🔒 prevent step before reset
    if env.state is None:
        return {
            "error": "Environment not initialized. Call /reset first."
        }

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }


# ✅ STATE endpoint (safe handling)
@app.get("/state")
def get_state():
    state = env.state

    # 🔒 prevent state before reset
    if state is None:
        return {"error": "Environment not initialized. Call /reset first."}

    return {
        "current_index": state.current_index,
        "total_reward": state.total_reward,
        "done": state.done,
        "task_id": state.task_id
    }


def main():
    uvicorn.run("email_env.server.app:app", host="0.0.0.0", port=8000)


# ✅ REQUIRED for OpenEnv validation
if __name__ == "__main__":
    main()