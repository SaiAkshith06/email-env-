from typing import List
from enum import Enum
from pydantic import Field

from openenv.core.env_server.types import Action, Observation


# ---------------- CATEGORY ----------------
class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    BUG = "bug"
    FEATURE = "feature"


# ---------------- PRIORITY ----------------
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# ---------------- ACTION ----------------
class EmailAction(Action):
    category: Category = Field(description="Email category")
    priority: Priority = Field(description="Priority level")
    is_ambiguous: bool = Field(default=False, description="Ambiguity flag")


# ---------------- OBSERVATION ----------------
class EmailObservation(Observation):
    email_id: str = Field(default="", description="Email ID")
    subject: str = Field(default="", description="Subject")
    body: str = Field(default="", description="Body")
    sender: str = Field(default="", description="Sender")
    step_count: int = Field(default=0)
    done: bool = Field(default=False)
    task_id: str = Field(default="task1")

    # Step 2 additions
    feedback: str = Field(default="", description="Feedback from environment")
    prev_rewards: List[float] = Field(default_factory=list)


# ---------------- STATE ----------------
class EmailState:
    def __init__(self, email_queue, current_index, total_reward, done, task_id):
        self.email_queue = email_queue
        self.current_index = current_index
        self.total_reward = total_reward
        self.done = done
        self.task_id = task_id

        # Step 2 addition
        self.reward_history: List[float] = []