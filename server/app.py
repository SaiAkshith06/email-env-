from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn, traceback

from server.email_env_environment import EmailEnvironment
from models import EmailAction

app = FastAPI(title="Email Triage Environment", version="2.0.0")

print("--- INITIALIZING EMAIL ENVIRONMENT V2 (60 EMAILS) ---")
env = EmailEnvironment()

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


@app.get("/")
def root():
    return {"status": "ok", "message": "Email Triage Environment Running", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def get_tasks():
    return {"tasks": [
        {"id": "easy", "description": "Predict category only"},
        {"id": "medium", "description": "Predict category + priority"},
        {"id": "hard", "description": "Predict category + priority + ambiguity"}
    ]}

@app.post("/reset")
def reset(req: ResetRequest = None):
    try:
        if req is None:
            req = ResetRequest()
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        return obs.dict()           # fallback to flat dict, no nesting
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/step")
def step(action: EmailAction):
    try:
        if env.state is None:
            return JSONResponse(status_code=400, content={"error": "Call /reset first"})
        obs, reward, done, info = env.step(action)
        return {"observation": obs.dict(), "reward": reward, "done": done, "info": info}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/state")
def get_state():
    if env.state is None:
        return JSONResponse(status_code=400, content={"error": "Call /reset first"})
    return {
        "current_index": env.state.current_index,
        "total_reward": env.state.total_reward,
        "done": env.state.done,
        "task_id": env.state.task_id
    }

def main():
    uvicorn.run("email_env.server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()