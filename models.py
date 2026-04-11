from typing import List, Optional
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


class ActionType(str, Enum):
    INVESTIGATE = "investigate"
    CLASSIFY    = "classify"


class EmailAction(Action):
    action_type: ActionType = Field(default=ActionType.CLASSIFY)
    # classify fields (required for classify, optional for investigate)
    category: Optional[Category] = Field(default=None)
    priority: Optional[Priority] = Field(default=None)
    is_ambiguous: bool = Field(default=False)
    # investigate field
    query: Optional[str] = Field(default=None, description="What do you want to check? (e.g., 'priority', 'category')")

    def to_dict(self):
        return {
            "action_type": str(self.action_type),
            "category": str(self.category) if self.category else None,
            "priority": str(self.priority) if self.priority else None,
            "is_ambiguous": self.is_ambiguous,
            "query": self.query
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
    investigate_used: bool = Field(default=False)


class EmailState(BaseModel):
    email_queue: List[dict] = Field(default_factory=list)
    current_index: int = Field(default=0)
    total_reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    task_id: str = Field(default="easy")
    reward_history: List[float] = Field(default_factory=list)
    current_email_investigated: bool = Field(default=False)

    class Config:
        arbitrary_types_allowed = True