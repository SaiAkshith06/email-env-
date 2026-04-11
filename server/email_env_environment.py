import json
import random
from typing import Tuple
from uuid import uuid4
from pathlib import Path

from openenv.core.env_server.interfaces import Environment

from email_env.models import EmailAction, EmailObservation, EmailState
from email_env.server.grader import grade_easy, grade_medium, grade_hard


_DOCKER_DATA_PATH = Path("/app/env/server/data.json")
_LOCAL_DATA_PATH = Path(__file__).parent / "data.json"

DATA_PATH = _DOCKER_DATA_PATH if _DOCKER_DATA_PATH.exists() else _LOCAL_DATA_PATH


def load_data():
    try:
        if not DATA_PATH.exists():
            return []
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                return []
            return data
    except Exception:
        return []


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


class EmailEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def init(self):
        self.data = load_data()
        self._state = None
        self._episode_id = str(uuid4())

    def reset(self, task_id="easy", seed=None) -> EmailObservation:

        if seed is not None:
            random.seed(seed)

        shuffled = self.data.copy()
        random.shuffle(shuffled)

        if not self.data:
            shuffled = [{
                "email_id": "fallback",
                "subject": "Missing Data",
                "body": "No data found",
                "sender": "system",
                "category": "general",
                "priority": "low",
                "is_ambiguous": False
            }]

        self._state = EmailState(
            email_queue=shuffled,
            current_index=0,
            total_reward=0.0,
            done=False,
            task_id=task_id
        )

        self._state.reward_history = []

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

    def compute_reward(self, action: EmailAction, email: dict) -> float:
        task = self._state.task_id

        if task == "easy":
            score = grade_easy(action, email)
        elif task == "medium":
            score = grade_medium(action, email)
        elif task == "hard":
            score = grade_hard(action, email)
        else:
            score = 0.0

        return safe_score(score)

    def generate_feedback(self, action: EmailAction, email: dict, reward: float) -> str:
        parts = []

        if action.category == email["category"]:
            parts.append(f"[CORRECT] Category: {action.category}")
        else:
            parts.append(f"[WRONG] Category: got {action.category}, expected {email['category']}")

        if action.priority == email["priority"]:
            parts.append(f"[CORRECT] Priority: {action.priority}")
        else:
            parts.append(f"[WRONG] Priority: got {action.priority}, expected {email['priority']}")

        true_amb = email.get("is_ambiguous", False)
        if action.is_ambiguous == true_amb:
            parts.append("[CORRECT] Ambiguity detected")
        else:
            parts.append(f"[WRONG] Ambiguity: got {action.is_ambiguous}, expected {true_amb}")

        parts.append(f"Reward: {reward:.3f}")

        return " | ".join(parts)

    def step(self, action: EmailAction) -> Tuple[EmailObservation, float, bool, dict]:

        if self._state.done:
            return EmailObservation(
                email_id="",
                subject="",
                body="",
                sender="",
                step_count=self._state.current_index,
                done=True,
                task_id=self._state.task_id,
                feedback="",
                prev_rewards=self._state.reward_history
            ), safe_score(0.0), True, {}

        current_email = self._state.email_queue[self._state.current_index]

        reward = self.compute_reward(action, current_email)

        self._state.reward_history.append(reward)

        self._state.total_reward += reward
        self._state.current_index += 1

        done = self._state.current_index >= len(self._state.email_queue)
        self._state.done = done

        feedback = self.generate_feedback(action, current_email, reward)

        if not done:
            next_email = self._state.email_queue[self._state.current_index]

            observation = EmailObservation(
                email_id=next_email["email_id"],
                subject=next_email["subject"],
                body=next_email["body"],
                sender=next_email["sender"],
                step_count=self._state.current_index,
                done=False,
                task_id=self._state.task_id,
                feedback=feedback,
                prev_rewards=self._state.reward_history
            )
        else:
            observation = EmailObservation(
                email_id="",
                subject="",
                body="",
                sender="",
                step_count=self._state.current_index,
                done=True,
                task_id=self._state.task_id,
                feedback=feedback,
                prev_rewards=self._state.reward_history
            )

        return observation, reward, done, {}

    @property
    def state(self) -> EmailState:
        return self._state