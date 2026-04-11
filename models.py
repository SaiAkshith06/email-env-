from typing import List
from enum import Enum
from pydantic import BaseModel, Field
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


class EmailState(BaseModel):
    email_queue: List[dict] = Field(default_factory=list)
    current_index: int = Field(default=0)
    total_reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    task_id: str = Field(default="easy")
    reward_history: List[float] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True