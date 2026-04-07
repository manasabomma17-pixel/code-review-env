"""
Deterministic grader for the Code Review environment.

grade_action()  — called after every step, returns a dense Reward
grade_episode() — called at episode end, returns final score 0.0–1.0

Reward logic:
  +0.40  True positive  — correctly flagging a real issue
  +0.15  Severity bonus — if the severity level is exactly correct
  +0.10  Category bonus — if the category is exactly correct
  -0.20  False positive — flagging a clean line
  -0.05  Per missing issue at episode end (false negative penalty)

Score is clipped to [-1.0, 1.0] per step and [0.0, 1.0] for final.
"""
from __future__ import annotations

from env.models import (
    Action, ActionType, IssueSeverity, Reward, FlaggedIssue
)

# Severity ordering for partial credit (within one level = half penalty)
_SEV_ORDER = [
    IssueSeverity.INFO,
    IssueSeverity.LOW,
    IssueSeverity.MEDIUM,
    IssueSeverity.HIGH,
    IssueSeverity.CRITICAL,
]

# Category match bonus
_CAT_BONUS = 0.10
# TP base reward
_TP_REWARD = 0.40
# Exact severity bonus
_SEV_BONUS = 0.15
# FP penalty
_FP_PENALTY = -0.20
# Per-step action reward (for approve_line on a clean line)
_APPROVE_CLEAN_BONUS = 0.02
# Final FN penalty per missed issue
_FN_PENALTY_PER_MISS = 0.10


def _severity_distance(predicted: IssueSeverity, actual: IssueSeverity) -> int:
    try:
        pi = _SEV_ORDER.index(predicted)
        ai = _SEV_ORDER.index(actual)
        return abs(pi - ai)
    except ValueError:
        return 4


def grade_action(
    action: Action,
    ground_truth: dict[int, dict],
    already_flagged: set[int],
) -> Reward:
    """
    Returns a dense Reward for a single action.

    Parameters
    ----------
    action        : The agent's Action
    ground_truth  : Mapping of line_number → ground truth dict
    already_flagged: Set of line numbers already flagged this episode
    """
    if action.action_type == ActionType.SUBMIT_REVIEW:
        return Reward(value=0.0)

    if action.action_type == ActionType.APPROVE_LINE:
        line = action.approved_line
        if line is not None and line not in ground_truth:
            return Reward(value=_APPROVE_CLEAN_BONUS)
        # approved a line that has an issue → small penalty
        return Reward(value=-0.05)

    # FLAG_ISSUE
    line = action.line_number
    if line is None:
        return Reward(value=0.0)

    # Duplicate flag
    if line in already_flagged:
        return Reward(value=-0.05)

    if line not in ground_truth:
        # False positive
        return Reward(
            value=_FP_PENALTY,
            false_positive=_FP_PENALTY,
        )

    # True positive — compute bonuses
    gt = ground_truth[line]
    sev_actual = gt["severity"]
    cat_actual = gt["category"]

    tp = _TP_REWARD
    sev_bonus = 0.0
    cat_bonus = 0.0

    if action.severity == sev_actual:
        sev_bonus = _SEV_BONUS
    elif action.severity is not None:
        dist = _severity_distance(action.severity, sev_actual)
        if dist == 1:
            sev_bonus = _SEV_BONUS * 0.5  # partial credit

    if action.category == cat_actual:
        cat_bonus = _CAT_BONUS

    total = min(1.0, tp + sev_bonus + cat_bonus)
    return Reward(
        value=total,
        true_positive=tp,
        severity_bonus=sev_bonus,
    )


def grade_episode(
    flagged_issues: list[FlaggedIssue],
    ground_truth: dict[int, dict],
) -> float:
    """
    Final episode score in [0.0, 1.0].

    Precision × Recall F1-like score:
      - Start from 1.0
      - Subtract FN penalty for each missed real issue
      - Subtract FP penalty for each spurious flag
      - Divide by number of ground truth issues to normalize
    """
    if not ground_truth:
        return 1.0 if not flagged_issues else 0.5

    flagged_lines = {f.line_number for f in flagged_issues}
    gt_lines = set(ground_truth.keys())

    true_positives = flagged_lines & gt_lines
    false_positives = flagged_lines - gt_lines
    false_negatives = gt_lines - flagged_lines

    n = len(gt_lines)
    tp_score = len(true_positives) / n
    fp_penalty = (len(false_positives) * 0.2) / max(n, 1)
    fn_penalty = (len(false_negatives) * _FN_PENALTY_PER_MISS) / n

    raw = tp_score - fp_penalty - fn_penalty
    return round(max(0.0, min(1.0, raw)), 4)
