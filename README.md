# CodeReviewEnv 🔍

> **OpenEnv × Scaler Hackathon** — Team Lakehouse

An OpenEnv-compliant reinforcement learning environment where an AI agent acts as a **code reviewer**, identifying security vulnerabilities, bugs, and logic errors in realistic code diffs.

---

## Problem Description

Code review is one of the most critical — and expensive — tasks in software engineering. A single missed SQL injection or hardcoded secret can lead to catastrophic data breaches. This environment trains agents to perform systematic, accurate security-focused code review across three difficulty levels.

The agent must:
1. Inspect a realistic code diff (Python or JavaScript)
2. Flag lines containing issues with the correct **severity** and **category**
3. Provide a clear explanation for each flag
4. Submit the review when complete

---

## Environment Design

### Architecture

```
my-openenv/
├── env/
│   ├── __init__.py
│   ├── models.py       ← Pydantic: Observation, Action, Reward
│   ├── data.py         ← Code diffs + ground truth issues
│   ├── grader.py       ← Deterministic scoring logic
│   └── environment.py  ← CodeReviewEnv class
├── tests/
│   └── test_environment.py
├── inference.py        ← Baseline agent (MANDATORY NAME, ROOT LOCATION)
├── app.py              ← FastAPI server for HF Spaces
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

### OpenEnv API

```python
from env import CodeReviewEnv, Action, ActionType, IssueSeverity, IssueCategory

env = CodeReviewEnv(task="easy")  # "easy" | "medium" | "hard"
obs = env.reset()

action = Action(
    action_type=ActionType.FLAG_ISSUE,
    line_number=2,
    severity=IssueSeverity.CRITICAL,
    category=IssueCategory.SECURITY,
    explanation="SQL injection via string concatenation — use parameterised queries",
)
obs, reward, done, info = env.step(action)
state = env.state()
```

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `flag_issue \| approve_line \| submit_review` | What the agent does |
| `line_number` | `int` | Line to flag (for `flag_issue`) |
| `severity` | `critical\|high\|medium\|low\|info` | Issue severity |
| `category` | `security\|bug\|performance\|style\|logic\|documentation` | Issue type |
| `explanation` | `str` | Human-readable reason |
| `approved_line` | `int` | Line to approve (for `approve_line`) |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `diff_lines` | `list[CodeLine]` | Full diff with line numbers |
| `file_name` | `str` | File under review |
| `language` | `str` | Programming language |
| `current_line_index` | `int` | Agent's current focus |
| `lines_reviewed` | `int` | Progress counter |
| `total_lines` | `int` | Total lines in diff |
| `issues_flagged` | `list[FlaggedIssue]` | Issues found so far |
| `step_number` | `int` | Steps taken this episode |
| `done` | `bool` | Episode complete |

---

## Tasks

### Easy — `user_service.py` (Python)
3 issues in a user management service. All issues are textbook and clearly visible.
- **Line 2**: SQL injection via string concatenation → `CRITICAL / security`
- **Line 7**: MD5 used for password hashing (broken crypto) → `CRITICAL / security`
- **Line 11**: `json` module used but not imported → `MEDIUM / bug`

### Medium — `server.js` (JavaScript / Express.js)
5 issues in a REST API server. Mix of severity levels.
- **Line 4**: Admin route with no auth middleware → `CRITICAL / security`
- **Line 11**: SQL injection via template literal → `CRITICAL / security`
- **Line 13**: Plaintext password comparison → `HIGH / security`
- **Line 21**: Silent catch swallows errors → `MEDIUM / bug`
- **Line 24**: Hardcoded JWT secret → `CRITICAL / security`

### Hard — `concurrency_utils.py` (Python)
5 subtle issues in concurrent code. Requires understanding of threading models.
- **Line 9**: TOCTOU race condition (double-checked locking bug) → `HIGH / bug`
- **Line 22**: Missing lock on shared `calls` dict → `HIGH / bug`
- **Line 30**: Lambda closure captures loop variable → `HIGH / bug`
- **Line 33**: Thread-unsafe `list.append` → `MEDIUM / bug`
- **Line 38**: `pickle.loads` on untrusted data → RCE → `CRITICAL / security`

---

## Reward Function

Dense rewards at every step encourage systematic review:

| Event | Reward |
|-------|--------|
| True positive (correct flag) | +0.40 |
| Exact severity match | +0.15 |
| Off-by-one severity | +0.075 |
| Exact category match | +0.10 |
| False positive (clean line flagged) | −0.20 |
| Duplicate flag | −0.05 |
| Approve a genuinely clean line | +0.02 |

**Final episode score** = F1-like metric: `TP/N − FP_penalty − FN_penalty` clamped to `[0.0, 1.0]`

---

## Setup & Usage

### Local

```bash
# 1. Clone and set up
git clone <your-repo>
cd code-review-env
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run tests
pytest tests/ -v

# 3. Smoke test
python -c "
from env import CodeReviewEnv, Action, ActionType, IssueSeverity, IssueCategory
env = CodeReviewEnv(task='easy')
obs = env.reset()
action = Action(action_type=ActionType.FLAG_ISSUE, line_number=2,
    severity=IssueSeverity.CRITICAL, category=IssueCategory.SECURITY,
    explanation='SQL injection')
obs, reward, done, info = env.step(action)
print('Reward:', reward, type(reward))
print('Done:', done, type(done))
print('Info:', list(info.keys()))
"

# 4. Start server
python app.py
# Visit http://localhost:7860/health

# 5. Run baseline inference
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_your_token_here
python inference.py
```

### Docker

```bash
docker build -t code-review-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api-inference.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  -e HF_TOKEN=hf_your_token_here \
  code-review-env
```

### HF Spaces Deployment

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create code-review-env --type space --space_sdk docker
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/code-review-env
git push hf main
```

Set secrets in HF Spaces Settings:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

---

## Baseline Results

| Task | Score | Notes |
|------|-------|-------|
| easy | ~0.85 | All 3 issues typically found |
| medium | ~0.60 | Auth bypass and template SQLi sometimes missed |
| hard | ~0.35 | Race conditions and closure bugs are challenging |

Model: `meta-llama/Llama-3.3-70B-Instruct` via HF Inference API

---

## License

MIT
