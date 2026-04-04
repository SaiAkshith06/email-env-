from pydantic import BaseModel
from enum import Enum
from typing import List

# Email categories
class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    BUG = "bug"
    FEATURE = "feature"

# Priority levels
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

# Agent action
class EmailAction(BaseModel):
    category: Category
    priority: Priority
    is_ambiguous: bool = False

# What agent sees
class EmailObservation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    step_count: int
    done: bool
    task_id: str = "task1"

# Internal state
class EmailState(BaseModel):
    email_queue: List[dict]
    current_index: int
    total_reward: float
    done: bool
    task_id: str = "task1"