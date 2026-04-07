from env import CodeReviewEnv, Action, ActionType, IssueSeverity, IssueCategory
env = CodeReviewEnv(task='easy')
obs = env.reset()
obs, reward, done, info = env.step(Action(
    action_type=ActionType.FLAG_ISSUE, line_number=2,
    severity=IssueSeverity.CRITICAL, category=IssueCategory.SECURITY,
    explanation='SQL injection'))
print('reward:', reward, type(reward))
print('done:', done, type(done))
print('info keys:', list(info.keys()))
