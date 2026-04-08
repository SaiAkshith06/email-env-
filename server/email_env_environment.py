import json
import os
import random
from typing import Tuple
from uuid import uuid4
from pathlib import Path

from openenv.core.env_server.interfaces import Environment

from email_env.models import EmailAction, EmailObservation, EmailState
from email_env.server.grader import grade_easy, grade_medium, grade_hard


# ✅ FIXED DATA PATH: Prioritize the Docker COPY'd source over the
# wheel-installed copy in site-packages (which can be stale due to uv cache).
_DOCKER_DATA_PATH = Path("/app/env/server/data.json")
_LOCAL_DATA_PATH = Path(__file__).parent / "data.json"

if _DOCKER_DATA_PATH.exists():
    DATA_PATH = _DOCKER_DATA_PATH
else:
    DATA_PATH = _LOCAL_DATA_PATH

print(f"[EmailEnv] Loading data from: {DATA_PATH}")


def load_data():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    print(f"[EmailEnv] Loaded {len(data)} emails. First subject: {data[0].get('subject', 'N/A')}")
    return data


class EmailEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.data = load_data()
        self._state = None
        self._episode_id = str(uuid4())

    # ✅ FIXED RESET (deterministic)
    def reset(self, task_id="easy", seed=None) -> EmailObservation:

        if seed is not None:
            random.seed(seed)

        self._state = EmailState(
            email_queue=self.data,
            current_index=0,
            total_reward=0.0,
            done=False,
            task_id=task_id
        )

        email = self._state.email_queue[0]

        return EmailObservation(
            email_id=email["email_id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            step_count=0,
            done=False,
            task_id=task_id
        )

    # ✅ CLEAN REWARD ROUTING
    def compute_reward(self, action: EmailAction, email: dict) -> float:
        task = self._state.task_id

        if task == "easy":
            return grade_easy(action, email)

        elif task == "medium":
            return grade_medium(action, email)

        elif task == "hard":
            return grade_hard(action, email)
        EPS = 1e-6
        return max(EPS, min(1 - EPS, float(0.0)))

    def step(self, action: EmailAction) -> Tuple[EmailObservation, float, bool, dict]:

        if self._state.done:
            return None, 0.0, True, {}

        current_email = self._state.email_queue[self._state.current_index]

        reward = self.compute_reward(action, current_email)

        self._state.total_reward += reward
        self._state.current_index += 1

        done = self._state.current_index >= len(self._state.email_queue)
        self._state.done = done

        if not done:
            next_email = self._state.email_queue[self._state.current_index]

            observation = EmailObservation(
                email_id=next_email["email_id"],
                subject=next_email["subject"],
                body=next_email["body"],
                sender=next_email["sender"],
                step_count=self._state.current_index,
                done=False,
                task_id=self._state.task_id
            )
        else:
            observation = EmailObservation(
                email_id="",
                subject="",
                body="",
                sender="",
                step_count=self._state.current_index,
                done=True,
                task_id=self._state.task_id
            )

        return observation, reward, done, {}

    @property
    def state(self) -> EmailState:
        return self._state