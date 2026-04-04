# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Email Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EmailAction, EmailObservation


class EmailEnv(
    EnvClient[EmailAction, EmailObservation, State]
):
    """
    Client for the Email Env Environment.

    This client connects to the environment server and allows the agent
    to interact with the email classification task.

    The agent predicts:
    - category (billing, technical, etc.)
    - priority (low, medium, high, urgent)
    - whether the email is ambiguous

    Example:
        >>> from email_env.client import EmailEnv
        >>> from email_env.models import EmailAction, Category, Priority
        >>>
        >>> with EmailEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.subject)
        ...
        ...     action = EmailAction(
        ...         category=Category.BILLING,
        ...         priority=Priority.HIGH,
        ...         is_ambiguous=False
        ...     )
        ...     result = client.step(action)
        ...     print(result.reward)
    """

    def _step_payload(self, action: EmailAction) -> Dict:
        """
        Convert EmailAction into JSON payload for /step.

        Args:
            action: EmailAction instance

        Returns:
            Dict payload
        """
        return {
            "category": action.category.value,
            "priority": action.priority.value,
            "is_ambiguous": action.is_ambiguous,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EmailObservation]:
        """
        Parse server response into StepResult.

        Args:
            payload: response JSON from server

        Returns:
            StepResult object
        """
        obs = payload.get("observation", {})

        observation = EmailObservation(
            email_id=obs.get("email_id"),
            subject=obs.get("subject"),
            body=obs.get("body"),
            sender=obs.get("sender"),
            step_count=obs.get("step_count"),
            done=obs.get("done"),
            task_id=obs.get("task_id"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse state endpoint response.

        Args:
            payload: JSON response

        Returns:
            State object
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )