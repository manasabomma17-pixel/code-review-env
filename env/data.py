"""
Dataset of realistic code review scenarios.
Each entry contains a diff and ground-truth issues with severity/category.

Ground truth format:
  {line_number: {"severity": IssueSeverity, "category": IssueCategory, "desc": str}}
"""
from __future__ import annotations
from env.models import IssueSeverity, IssueCategory


# ---------------------------------------------------------------------------
# EASY — single file, obvious bugs/issues, Python
# ---------------------------------------------------------------------------

EASY_DIFF = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    result = db.execute(query)
    return result[0]

def hash_password(password):
    import md5
    return md5.new(password).hexdigest()

def read_config():
    with open("config.json") as f:
        data = json.load(f)
    return data
""".strip()

EASY_GROUND_TRUTH: dict[int, dict] = {
    2: {
        "severity": IssueSeverity.CRITICAL,
        "category": IssueCategory.SECURITY,
        "desc": "SQL injection via string concatenation"
    },
    7: {
        "severity": IssueSeverity.CRITICAL,
        "category": IssueCategory.SECURITY,
        "desc": "MD5 is cryptographically broken for password hashing"
    },
    11: {
        "severity": IssueSeverity.MEDIUM,
        "category": IssueCategory.BUG,
        "desc": "json module not imported; will raise NameError"
    },
}

EASY_META = {
    "file_name": "user_service.py",
    "language": "python",
    "description": "User management service with authentication utilities",
}


# ---------------------------------------------------------------------------
# MEDIUM — multiple issue types, mixed severity, JavaScript
# ---------------------------------------------------------------------------

MEDIUM_DIFF = """
const express = require('express');
const app = express();

app.get('/admin', (req, res) => {
    res.send('Admin panel');
});

app.post('/login', async (req, res) => {
    const { username, password } = req.body;
    const user = await db.query(
        `SELECT * FROM users WHERE username = '${username}'`
    );
    if (user && user.password === password) {
        req.session.user = user;
        res.json({ success: true });
    }
});

function retry(fn, times) {
    for (let i = 0; i < times; i++) {
        try {
            return fn();
        } catch (e) {}
    }
}

const SECRET = "hardcoded_jwt_secret_12345";
app.listen(3000);
""".strip()

MEDIUM_GROUND_TRUTH: dict[int, dict] = {
    4: {
        "severity": IssueSeverity.CRITICAL,
        "category": IssueCategory.SECURITY,
        "desc": "Admin route has no authentication/authorization middleware"
    },
    11: {
        "severity": IssueSeverity.CRITICAL,
        "category": IssueCategory.SECURITY,
        "desc": "SQL injection via template literal string interpolation"
    },
    13: {
        "severity": IssueSeverity.HIGH,
        "category": IssueCategory.SECURITY,
        "desc": "Plaintext password comparison — must use bcrypt/argon2"
    },
    21: {
        "severity": IssueSeverity.MEDIUM,
        "category": IssueCategory.BUG,
        "desc": "Silent catch swallows errors; retry failures are invisible"
    },
    24: {
        "severity": IssueSeverity.CRITICAL,
        "category": IssueCategory.SECURITY,
        "desc": "Hardcoded JWT secret in source code"
    },
}

MEDIUM_META = {
    "file_name": "server.js",
    "language": "javascript",
    "description": "Express.js REST API server with authentication",
}


# ---------------------------------------------------------------------------
# HARD — subtle issues, concurrency, Go-style Python, many clean lines
# ---------------------------------------------------------------------------

HARD_DIFF = """
import threading
import time
from collections import defaultdict

_cache = {}
_lock = threading.Lock()

def get_cached(key, fetch_fn):
    if key in _cache:
        return _cache[key]
    with _lock:
        result = fetch_fn()
        _cache[key] = result
        return result

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = defaultdict(list)

    def is_allowed(self, client_id):
        now = time.time()
        window = [t for t in self.calls[client_id] if now - t < self.period]
        self.calls[client_id] = window
        if len(window) < self.max_calls:
            self.calls[client_id].append(now)
            return True
        return False

def process_batch(items, worker_fn):
    threads = []
    results = []
    for item in items:
        t = threading.Thread(target=lambda: results.append(worker_fn(item)))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results

def deserialize(data: bytes):
    import pickle
    return pickle.loads(data)
""".strip()

HARD_GROUND_TRUTH: dict[int, dict] = {
    9: {
        "severity": IssueSeverity.HIGH,
        "category": IssueCategory.BUG,
        "desc": "TOCTOU race: check-then-lock pattern allows duplicate fetches (double-checked locking without re-checking inside lock)"
    },
    22: {
        "severity": IssueSeverity.HIGH,
        "category": IssueCategory.BUG,
        "desc": "RateLimiter.calls has no lock; concurrent access causes data races"
    },
    30: {
        "severity": IssueSeverity.HIGH,
        "category": IssueCategory.BUG,
        "desc": "Lambda captures loop variable by reference — all threads run worker_fn on the last item"
    },
    33: {
        "severity": IssueSeverity.MEDIUM,
        "category": IssueCategory.BUG,
        "desc": "results.append is not thread-safe without a lock"
    },
    38: {
        "severity": IssueSeverity.CRITICAL,
        "category": IssueCategory.SECURITY,
        "desc": "pickle.loads on untrusted bytes allows arbitrary code execution (RCE)"
    },
}

HARD_META = {
    "file_name": "concurrency_utils.py",
    "language": "python",
    "description": "Concurrency utilities: cache, rate limiter, batch processor",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS = {
    "easy": {
        "diff": EASY_DIFF,
        "ground_truth": EASY_GROUND_TRUTH,
        **EASY_META,
    },
    "medium": {
        "diff": MEDIUM_DIFF,
        "ground_truth": MEDIUM_GROUND_TRUTH,
        **MEDIUM_META,
    },
    "hard": {
        "diff": HARD_DIFF,
        "ground_truth": HARD_GROUND_TRUTH,
        **HARD_META,
    },
}
