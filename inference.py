"""
inference.py — Baseline inference script for CodeReviewEnv.
MUST be in the project root directory.

Required environment variables:
  API_BASE_URL  — e.g. https://api-inference.huggingface.co/v1
  MODEL_NAME    — e.g. meta-llama/Llama-3.3-70B-Instruct
  HF_TOKEN      — your Hugging Face token (used as API key)

Uses OpenAI client (pointed at HF Inference API).
Emits structured [START], [STEP], [END] logs to stdout.
"""
from __future__ import annotations

import json
import os
import sys
import time

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment / client setup
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set — requests may fail for gated models", file=sys.stderr)

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# Import environment AFTER path is set
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import (  # noqa: E402
    CodeReviewEnv, Action, ActionType,
    IssueSeverity, IssueCategory,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert code reviewer specializing in security vulnerabilities and bugs.

You will be shown a code diff. For each step you must output a JSON action with EXACTLY this schema:

For flagging an issue:
{
  "action_type": "flag_issue",
  "line_number": <int>,
  "severity": "<critical|high|medium|low|info>",
  "category": "<security|bug|performance|style|logic|documentation>",
  "explanation": "<clear explanation of the issue>"
}

For approving a clean line:
{
  "action_type": "approve_line",
  "approved_line": <int>
}

For finishing the review:
{
  "action_type": "submit_review"
}

Rules:
- Output ONLY valid JSON. No prose, no markdown, no backticks.
- Focus on SECURITY issues first (SQL injection, hardcoded secrets, insecure crypto, RCE).
- Then BUG issues (race conditions, unhandled errors, wrong logic).
- When all critical issues are flagged, submit the review.
"""


def build_user_message(obs_dict: dict) -> str:
    lines = obs_dict["diff_lines"]
    code_block = "\n".join(
        f"{l['line_number']:>3}: {l['content']}" for l in lines
    )
    flagged = obs_dict.get("issues_flagged", [])
    flagged_summary = (
        "\n".join(f"  Line {f['line_number']}: [{f['severity']}] {f['explanation']}" for f in flagged)
        if flagged else "  (none yet)"
    )
    return (
        f"File: {obs_dict['file_name']} ({obs_dict['language']})\n"
        f"Step: {obs_dict['step_number']} | "
        f"Lines reviewed: {obs_dict['lines_reviewed']}/{obs_dict['total_lines']}\n\n"
        f"CODE DIFF:\n{code_block}\n\n"
        f"Issues flagged so far:\n{flagged_summary}\n\n"
        "What is your next action? Output JSON only."
    )


def call_llm(messages: list[dict]) -> str:
    """Call LLM via OpenAI client, return text content."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=300,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str) -> Action:
    """Parse LLM JSON output into an Action, with fallback."""
    # Strip markdown fences if present
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        data = json.loads(raw)
        return Action(**data)
    except Exception:
        # Fallback: submit review if parsing fails
        return Action(action_type=ActionType.SUBMIT_REVIEW)


def run_task(task: str) -> float:
    """Run one full episode on a task, return final score."""
    env = CodeReviewEnv(task=task)
    obs = env.reset()
    obs_dict = obs.model_dump()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(obs_dict)},
    ]

    final_score = 0.0
    step = 0
    max_steps = 40

    while step < max_steps:
        step += 1

        # Call LLM
        try:
            raw_response = call_llm(messages)
        except Exception as e:
            print(f"[WARN] LLM call failed at step {step}: {e}", file=sys.stderr)
            break

        # Parse action
        action = parse_action(raw_response)

        # Emit [STEP] log
        print(json.dumps({
            "type": "[STEP]",
            "task": task,
            "step": step,
            "action": action.model_dump(),
            "raw_llm": raw_response[:200],
        }), flush=True)

        # Execute in environment
        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()

        # Append to conversation
        messages.append({"role": "assistant", "content": raw_response})
        if not done:
            messages.append({
                "role": "user",
                "content": (
                    f"Step result: reward={reward:.3f}\n"
                    + build_user_message(obs_dict)
                ),
            })

        if done:
            final_score = info.get("final_score") or 0.0
            break

        time.sleep(0.3)  # rate limit courtesy pause

    return final_score


def main() -> None:
    tasks = ["easy", "medium", "hard"]
    results = {}

    # [START] log
    print(json.dumps({
        "type": "[START]",
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "tasks": tasks,
    }), flush=True)

    for task in tasks:
        print(f"\n{'='*50}", file=sys.stderr)
        print(f"Running task: {task}", file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)

        try:
            score = run_task(task)
        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}", file=sys.stderr)
            score = 0.0

        results[task] = score
        print(f"Task '{task}' score: {score:.4f}", file=sys.stderr)

    # [END] log
    print(json.dumps({
        "type": "[END]",
        "results": results,
        "average_score": round(sum(results.values()) / len(results), 4),
    }), flush=True)


if __name__ == "__main__":
    main()
