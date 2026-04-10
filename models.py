from typing import List
from enum import Enum
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    BUG = "bug"
    FEATURE = "feature"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class EmailAction(Action):
    category: Category = Field(...)
    priority: Priority = Field(...)
    is_ambiguous: bool = Field(default=False)

    def to_dict(self):
        return {
            "category": str(self.category),
            "priority": str(self.priority),
            "is_ambiguous": self.is_ambiguous
        }


class EmailObservation(Observation):
    email_id: str = Field(default="")
    subject: str = Field(default="")
    body: str = Field(default="")
    sender: str = Field(default="")
    step_count: int = Field(default=0)
    done: bool = Field(default=False)
    task_id: str = Field(default="task1")
    feedback: str = Field(default="")
    prev_rewards: List[float] = Field(default_factory=list)


class EmailState:
    def init(
        self,
        email_queue: List[dict],
        current_index: int,
        total_reward: float,
        done: bool,
        task_id: str
    ):
        self.email_queue = email_queue
        self.current_index = current_index
        self.total_reward = total_reward
        self.done = done
        self.task_id = task_id
        self.reward_history: List[float] = []