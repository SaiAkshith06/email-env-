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
    """

    def _step_payload(self, action: EmailAction) -> Dict:
        """
        Convert EmailAction into JSON payload for /step.
        """
        payload = {
            "action_type": action.action_type.value,
            "is_ambiguous": action.is_ambiguous,
        }
        
        if action.category:
            payload["category"] = action.category.value
        if action.priority:
            payload["priority"] = action.priority.value
        if action.query:
            payload["query"] = action.query
            
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[EmailObservation]:
        """
        Parse server response into StepResult.
        """
        obs = payload.get("observation", {}) or payload # handles flat or nested

        observation = EmailObservation(
            email_id=obs.get("email_id"),
            subject=obs.get("subject"),
            body=obs.get("body"),
            sender=obs.get("sender"),
            step_count=obs.get("step_count"),
            done=obs.get("done"),
            task_id=obs.get("task_id"),
            feedback=obs.get("feedback"),
            investigate_used=obs.get("investigate_used", False)
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse state endpoint response.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )