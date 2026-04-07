"""
Typed Pydantic models for the Code Review OpenEnv environment.
All OpenEnv-compliant: Observation, Action, Reward.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class IssueSeverity(str, Enum):
    CRITICAL = "critical"   # security vulnerability, crash-risk
    HIGH = "high"           # logic bug, data loss
    MEDIUM = "medium"       # performance, bad practice
    LOW = "low"             # style, minor suggestion
    INFO = "info"           # informational note


class IssueCategory(str, Enum):
    SECURITY = "security"
    BUG = "bug"
    PERFORMANCE = "performance"
    STYLE = "style"
    LOGIC = "logic"
    DOCUMENTATION = "documentation"
    NONE = "none"           # used when no issue exists on a line


class ActionType(str, Enum):
    FLAG_ISSUE = "flag_issue"       # mark a line as having a problem
    APPROVE_LINE = "approve_line"   # explicitly mark a line as clean
    SUBMIT_REVIEW = "submit_review" # finish the review session


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class CodeLine(BaseModel):
    line_number: int
    content: str
    is_changed: bool = True  # True = part of the diff


class FlaggedIssue(BaseModel):
    line_number: int
    severity: IssueSeverity
    category: IssueCategory
    explanation: str = Field(..., min_length=5, max_length=500)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    diff_lines: list[CodeLine]
    file_name: str
    language: str
    current_line_index: int        # which line the agent is focused on
    lines_reviewed: int
    total_lines: int
    issues_flagged: list[FlaggedIssue] = Field(default_factory=list)
    step_number: int
    task: str                      # "easy" | "medium" | "hard"
    done: bool = False


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """What the agent does at each step."""
    action_type: ActionType
    # Required for FLAG_ISSUE
    line_number: int | None = None
    severity: IssueSeverity | None = None
    category: IssueCategory | None = None
    explanation: str | None = None
    # Required for APPROVE_LINE
    approved_line: int | None = None


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Scalar reward with breakdown for interpretability."""
    value: float = Field(..., ge=-1.0, le=1.0)
    true_positive: float = 0.0    # correctly flagged real issue
    false_positive: float = 0.0   # flagged clean line
    false_negative: float = 0.0   # missed a real issue
    severity_bonus: float = 0.0   # extra credit for correct severity
    explanation_quality: float = 0.0
