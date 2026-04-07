"""
CodeReviewEnv — OpenEnv-compliant environment.

An AI agent acts as a code reviewer. It is shown a diff one step at a time
and must flag issues (or approve clean lines) before submitting the review.

API
---
  env = CodeReviewEnv(task="easy")  # "easy" | "medium" | "hard"
  obs = env.reset()
  obs, reward, done, info = env.step(action)
  state = env.state()
"""
from __future__ import annotations

import logging
from typing import Any

from env.data import TASKS
from env.grader import grade_action, grade_episode
from env.models import (
    Action, ActionType, CodeLine, FlaggedIssue,
    Observation, Reward,
)

logger = logging.getLogger(__name__)

MAX_STEPS = 120  # safety cap


class CodeReviewEnv:
    """OpenEnv environment for code review."""

    def __init__(self, task: str = "easy") -> None:
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASKS)}")
        self.task = task
        self._task_data = TASKS[task]
        self._ground_truth: dict[int, dict] = self._task_data["ground_truth"]

        # Episode state — initialised by reset()
        self._diff_lines: list[CodeLine] = []
        self._current_line_index: int = 0
        self._step_number: int = 0
        self._flagged_issues: list[FlaggedIssue] = []
        self._flagged_line_numbers: set[int] = set()
        self._done: bool = False
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        raw_diff = self._task_data["diff"]
        self._diff_lines = self._parse_diff(raw_diff)
        self._current_line_index = 0
        self._step_number = 0
        self._flagged_issues = []
        self._flagged_line_numbers = set()
        self._done = False
        self._cumulative_reward = 0.0

        logger.info("Environment reset. Task=%s, lines=%d", self.task, len(self._diff_lines))
        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Apply action and return (observation, reward, done, info).

        reward is a float in [-1.0, 1.0].
        done is True when the agent submits or runs out of steps.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_number += 1

        # Compute per-step reward
        reward_obj: Reward = grade_action(
            action=action,
            ground_truth=self._ground_truth,
            already_flagged=self._flagged_line_numbers,
        )
        reward_val = float(reward_obj.value)
        self._cumulative_reward += reward_val

        # Update state based on action
        if action.action_type == ActionType.FLAG_ISSUE:
            if (
                action.line_number is not None
                and action.line_number not in self._flagged_line_numbers
                and action.severity is not None
                and action.category is not None
                and action.explanation
            ):
                issue = FlaggedIssue(
                    line_number=action.line_number,
                    severity=action.severity,
                    category=action.category,
                    explanation=action.explanation,
                )
                self._flagged_issues.append(issue)
                self._flagged_line_numbers.add(action.line_number)
            self._advance_line()

        elif action.action_type == ActionType.APPROVE_LINE:
            self._advance_line()

        elif action.action_type == ActionType.SUBMIT_REVIEW:
            self._done = True

        # Auto-done when all lines reviewed or step cap hit
        if self._current_line_index >= len(self._diff_lines):
            self._done = True
        if self._step_number >= MAX_STEPS:
            self._done = True

        # Final episode score
        final_score = 0.0
        if self._done:
            final_score = grade_episode(self._flagged_issues, self._ground_truth)

        info: dict[str, Any] = {
            "step": self._step_number,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "reward_breakdown": reward_obj.model_dump(),
            "issues_flagged": len(self._flagged_issues),
            "final_score": final_score if self._done else None,
            "task": self.task,
        }

        obs = self._make_observation()
        logger.debug("Step=%d action=%s reward=%.3f done=%s",
                     self._step_number, action.action_type, reward_val, self._done)
        return obs, reward_val, self._done, info

    def state(self) -> dict[str, Any]:
        """Return the full internal state (for debugging / spec compliance)."""
        return {
            "task": self.task,
            "step_number": self._step_number,
            "current_line_index": self._current_line_index,
            "total_lines": len(self._diff_lines),
            "flagged_issues": [f.model_dump() for f in self._flagged_issues],
            "flagged_line_numbers": sorted(self._flagged_line_numbers),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "done": self._done,
            "file_name": self._task_data["file_name"],
            "language": self._task_data["language"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_diff(self, raw: str) -> list[CodeLine]:
        lines = []
        for i, content in enumerate(raw.splitlines(), start=1):
            lines.append(CodeLine(line_number=i, content=content, is_changed=True))
        return lines

    def _advance_line(self) -> None:
        self._current_line_index = min(
            self._current_line_index + 1, len(self._diff_lines)
        )

    def _make_observation(self) -> Observation:
        return Observation(
            diff_lines=self._diff_lines,
            file_name=self._task_data["file_name"],
            language=self._task_data["language"],
            current_line_index=self._current_line_index,
            lines_reviewed=self._current_line_index,
            total_lines=len(self._diff_lines),
            issues_flagged=list(self._flagged_issues),
            step_number=self._step_number,
            task=self.task,
            done=self._done,
        )
