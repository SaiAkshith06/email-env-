import json
import random
from typing import Tuple, List
from uuid import uuid4
from pathlib import Path

from openenv.core.env_server.interfaces import Environment

from models import EmailAction, EmailObservation, EmailState, ActionType
from server.grader import grade_easy, grade_medium, grade_hard, grade_adaptive


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

    def __init__(self):
        super().__init__()
        self.data = load_data()
        self._state = None
        self._episode_id = str(uuid4())
        print(f"[EmailEnv] Loaded {len(self.data)} emails")

    def reset(self, task_id="easy", seed=None) -> EmailObservation:

        if seed is not None:
            random.seed(seed)

        if not self.data:
            self.data = [{
                "email_id": "fallback",
                "subject": "Missing Data",
                "body": "No data found",
                "sender": "system",
                "category": "general",
                "priority": "low",
                "is_ambiguous": False
            }]

        if task_id in ["hard", "adaptive"]:
            pool = [e for e in self.data if e.get("difficulty") == "hard"] or self.data
        else:
            pool = [e for e in self.data if e.get("difficulty") != "hard"] or self.data

        shuffled = pool.copy()
        random.shuffle(shuffled)

        self._state = EmailState(
            email_queue=shuffled,
            current_index=0,
            total_reward=0.0,
            done=False,
            task_id=task_id,
            reward_history=[],
            current_email_investigated=False,
            current_email_investigate_count=0
        )

        email = self._state.email_queue[0]

        return EmailObservation(
            email_id=email["email_id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            sender_tier=email.get("sender_tier", "unknown"),
            hours_since_received=email.get("hours_since_received", 0),
            step_count=0,
            done=False,
            task_id=task_id,
            investigate_used=False,
            episode_progress=0.0
        )

    def compute_reward(self, action: EmailAction, email: dict) -> float:
        task = self._state.task_id

        if task == "easy":
            score = grade_easy(action, email)
        elif task == "medium":
            score = grade_medium(action, email)
        elif task == "hard":
            score = grade_hard(action, email)
        elif task == "adaptive":
            score = grade_adaptive(action, email, self._state.current_email_investigate_count)
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

    def _generate_hint(self, email: dict, query: str) -> str:
        q = (query or "").lower()
        if "priority" in q or "urgent" in q:
            return f"The sender's tone suggests this is {email['priority']} priority."
        if "category" in q or "type" in q:
            body = email['body'].lower()
            if "payment" in body or "invoice" in body or "billing" in body:
                return "This email mentions financial transactions."
            if "crash" in body or "error" in body or "technical" in body:
                return "This email describes a technical failure."
            return "The email topic is unclear without more context."
        if "ambiguous" in q:
            return "Yes, this email has mixed signals." if email.get('is_ambiguous') else "This email seems straightforward."
        return "The email requires careful reading of the full body."

    def step(self, action: EmailAction) -> Tuple[EmailObservation, float, bool, dict]:

        if self._state.done:
            return self._finalize_observation("Episode already finished."), safe_score(0.0), True, {}

        current_email = self._state.email_queue[self._state.current_index]

        # --- HANDLE INVESTIGATE ---
        if action.action_type == ActionType.INVESTIGATE:
            hint = self._generate_hint(current_email, action.query)
            self._state.current_email_investigated = True
            self._state.current_email_investigate_count += 1
            
            progress = self._state.current_index / max(1, len(self._state.email_queue))
            obs = EmailObservation(
                email_id=current_email["email_id"],
                subject=current_email["subject"],
                body=current_email["body"],
                sender=current_email["sender"],
                sender_tier=current_email.get("sender_tier", "unknown"),
                hours_since_received=current_email.get("hours_since_received", 0),
                step_count=self._state.current_index,
                done=False,
                task_id=self._state.task_id,
                feedback=f"[HINT] {hint}",
                prev_rewards=self._state.reward_history,
                investigate_used=True,
                episode_progress=progress
            )
            return obs, 0.0, False, {}

        # --- HANDLE CLASSIFY ---
        reward = self.compute_reward(action, current_email)
        self._state.reward_history.append(reward)
        self._state.total_reward += reward
        self._state.current_index += 1
        self._state.current_email_investigated = False # reset for next email
        self._state.current_email_investigate_count = 0

        done = self._state.current_index >= len(self._state.email_queue)
        self._state.done = done

        feedback = self.generate_feedback(action, current_email, reward)

        if not done:
            next_email = self._state.email_queue[self._state.current_index]
            progress = self._state.current_index / max(1, len(self._state.email_queue))
            observation = EmailObservation(
                email_id=next_email["email_id"],
                subject=next_email["subject"],
                body=next_email["body"],
                sender=next_email["sender"],
                sender_tier=next_email.get("sender_tier", "unknown"),
                hours_since_received=next_email.get("hours_since_received", 0),
                step_count=self._state.current_index,
                done=False,
                task_id=self._state.task_id,
                feedback=feedback,
                prev_rewards=self._state.reward_history,
                investigate_used=False,
                episode_progress=progress
            )
        else:
            observation = self._finalize_observation(feedback)

        return observation, reward, done, {}

    def _finalize_observation(self, last_feedback: str) -> EmailObservation:
        avg_reward = sum(self._state.reward_history) / len(self._state.reward_history) if self._state.reward_history else 0.0
        summary = (
            f"{last_feedback} | "
            f"Episode complete. {len(self._state.reward_history)} emails processed. "
            f"Average reward: {avg_reward:.3f}. "
            f"Total score: {self._state.total_reward:.3f}"
        )
        return EmailObservation(
            email_id="", subject="", body="", sender="",
            sender_tier="unknown",
            hours_since_received=0,
            step_count=self._state.current_index,
            done=True,
            task_id=self._state.task_id,
            feedback=summary,
            prev_rewards=self._state.reward_history,
            investigate_used=False,
            episode_progress=1.0
        )

    @property
    def state(self) -> EmailState:
        return self._state