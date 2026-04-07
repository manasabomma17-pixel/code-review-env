import pytest
from env import (
    CodeReviewEnv, Action, ActionType,
    IssueSeverity, IssueCategory,
)
from env.grader import grade_action, grade_episode
from env.models import FlaggedIssue


@pytest.fixture
def easy_env():
    env = CodeReviewEnv(task="easy")
    env.reset()
    return env


class TestAPICompliance:
    def test_reset_returns_observation(self, easy_env):
        obs = easy_env.reset()
        assert obs.diff_lines
        assert obs.task == "easy"
        assert obs.step_number == 0
        assert obs.done is False

    def test_step_returns_correct_types(self, easy_env):
        action = Action(
            action_type=ActionType.FLAG_ISSUE,
            line_number=2,
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SECURITY,
            explanation="SQL injection via string concatenation",
        )
        obs, reward, done, info = easy_env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert -1.0 <= reward <= 1.0

    def test_state_returns_dict(self, easy_env):
        s = easy_env.state()
        assert isinstance(s, dict)
        assert "task" in s
        assert "step_number" in s
        assert "flagged_issues" in s

    def test_submit_review_ends_episode(self, easy_env):
        action = Action(action_type=ActionType.SUBMIT_REVIEW)
        _, _, done, _ = easy_env.step(action)
        assert done is True

    def test_reset_clears_state(self, easy_env):
        action = Action(
            action_type=ActionType.FLAG_ISSUE,
            line_number=2,
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SECURITY,
            explanation="SQL injection via string concat",
        )
        easy_env.step(action)
        easy_env.reset()
        s = easy_env.state()
        assert s["step_number"] == 0
        assert s["flagged_issues"] == []

    def test_error_on_step_after_done(self, easy_env):
        easy_env.step(Action(action_type=ActionType.SUBMIT_REVIEW))
        with pytest.raises(RuntimeError):
            easy_env.step(Action(action_type=ActionType.SUBMIT_REVIEW))

    def test_all_three_tasks_initialize(self):
        for task in ["easy", "medium", "hard"]:
            env = CodeReviewEnv(task=task)
            obs = env.reset()
            assert obs.task == task
            assert len(obs.diff_lines) > 0

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            CodeReviewEnv(task="impossible")


class TestRewards:
    def test_true_positive_gives_positive_reward(self):
        gt = {2: {"severity": IssueSeverity.CRITICAL, "category": IssueCategory.SECURITY, "desc": "SQLi"}}
        action = Action(
            action_type=ActionType.FLAG_ISSUE,
            line_number=2,
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SECURITY,
            explanation="SQL injection vulnerability found",
        )
        r = grade_action(action, gt, set())
        assert r.value > 0

    def test_false_positive_gives_negative_reward(self):
        gt = {}
        action = Action(
            action_type=ActionType.FLAG_ISSUE,
            line_number=5,
            severity=IssueSeverity.LOW,
            category=IssueCategory.STYLE,
            explanation="clean line flagged incorrectly",
        )
        r = grade_action(action, gt, set())
        assert r.value < 0

    def test_exact_severity_gives_bonus(self):
        gt = {7: {"severity": IssueSeverity.CRITICAL, "category": IssueCategory.SECURITY, "desc": "x"}}
        action_exact = Action(
            action_type=ActionType.FLAG_ISSUE,
            line_number=7,
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SECURITY,
            explanation="exact severity match found here",
        )
        action_wrong = Action(
            action_type=ActionType.FLAG_ISSUE,
            line_number=7,
            severity=IssueSeverity.LOW,
            category=IssueCategory.SECURITY,
            explanation="wrong severity assigned here",
        )
        r_exact = grade_action(action_exact, gt, set())
        r_wrong = grade_action(action_wrong, gt, set())
        assert r_exact.value > r_wrong.value

    def test_duplicate_flag_penalized(self):
        gt = {2: {"severity": IssueSeverity.HIGH, "category": IssueCategory.BUG, "desc": "x"}}
        action = Action(
            action_type=ActionType.FLAG_ISSUE,
            line_number=2,
            severity=IssueSeverity.HIGH,
            category=IssueCategory.BUG,
            explanation="duplicate flag attempt here",
        )
        r = grade_action(action, gt, {2})
        assert r.value < 0

    def test_reward_always_in_range(self, easy_env):
        actions = [
            Action(action_type=ActionType.FLAG_ISSUE, line_number=1,
                severity=IssueSeverity.HIGH, category=IssueCategory.BUG,
                explanation="potential bug found here"),
            Action(action_type=ActionType.APPROVE_LINE, approved_line=3),
            Action(action_type=ActionType.SUBMIT_REVIEW),
        ]
        for a in actions:
            _, reward, done, _ = easy_env.step(a)
            assert -1.0 <= reward <= 1.0
            if done:
                break


class TestGrader:
    def test_perfect_score_all_found(self):
        gt = {
            2: {"severity": IssueSeverity.CRITICAL, "category": IssueCategory.SECURITY, "desc": "x"},
            7: {"severity": IssueSeverity.HIGH, "category": IssueCategory.BUG, "desc": "y"},
        }
        flagged = [
            FlaggedIssue(line_number=2, severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.SECURITY, explanation="found it"),
            FlaggedIssue(line_number=7, severity=IssueSeverity.HIGH,
                        category=IssueCategory.BUG, explanation="found it"),
        ]
        score = grade_episode(flagged, gt)
        assert score == 1.0

    def test_zero_score_nothing_found(self):
        gt = {2: {"severity": IssueSeverity.CRITICAL, "category": IssueCategory.SECURITY, "desc": "x"}}
        score = grade_episode([], gt)
        assert score == 0.0

    def test_partial_score_some_found(self):
        gt = {
            2: {"severity": IssueSeverity.CRITICAL, "category": IssueCategory.SECURITY, "desc": "x"},
            7: {"severity": IssueSeverity.HIGH, "category": IssueCategory.BUG, "desc": "y"},
        }
        flagged = [
            FlaggedIssue(line_number=2, severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.SECURITY, explanation="found one"),
        ]
        score = grade_episode(flagged, gt)
        assert 0.0 < score < 1.0

    def test_final_score_in_valid_range(self, easy_env):
        easy_env.reset()
        action = Action(action_type=ActionType.SUBMIT_REVIEW)
        _, _, done, info = easy_env.step(action)
        assert done
        assert 0.0 <= info["final_score"] <= 1.0