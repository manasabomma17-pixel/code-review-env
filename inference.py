"""
inference.py — Baseline inference script for CodeReviewEnv.
"""
from __future__ import annotations

import os
import sys
import json
import time
from typing import Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import (
    CodeReviewEnv, Action, ActionType,
    IssueSeverity, IssueCategory,
)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
BENCHMARK = "code-review-env"
MAX_STEPS = 30

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert code reviewer specializing in security vulnerabilities and bugs.

Output a JSON action with EXACTLY this schema:

For flagging an issue:
{"action_type":"flag_issue","line_number":<int>,"severity":"<critical|high|medium|low|info>","category":"<security|bug|performance|style|logic|documentation>","explanation":"<clear explanation>"}

For approving a clean line:
{"action_type":"approve_line","approved_line":<int>}

For finishing the review:
{"action_type":"submit_review"}

Output ONLY valid JSON. No prose, no markdown, no backticks.
Focus on SECURITY issues first, then BUG issues.
When all critical issues are flagged, submit the review."""


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
        f"CODE DIFF:\n{code_block}\n\n"
        f"Issues flagged so far:\n{flagged_summary}\n\n"
        "Output your next action as JSON only."
    )


def call_llm(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=300,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str) -> Action:
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        data = json.loads(raw)
        return Action(**data)
    except Exception:
        return Action(action_type=ActionType.SUBMIT_REVIEW)


def run_task(task: str) -> None:
    env = CodeReviewEnv(task=task)
    obs = env.reset()
    obs_dict = obs.model_dump()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(obs_dict)},
    ]

    rewards: list[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        for step in range(1, MAX_STEPS + 1):
            error_msg = None
            try:
                raw_response = call_llm(messages)
            except Exception as e:
                error_msg = str(e)[:50]
                raw_response = '{"action_type":"submit_review"}'

            action = parse_action(raw_response)
            action_str = action.action_type.value

            try:
                obs, reward, done, info = env.step(action)
                obs_dict = obs.model_dump()
            except Exception as e:
                error_msg = str(e)[:50]
                reward = 0.0
                done = True
                info = {"final_score": 0.01}

            rewards.append(float(reward))
            steps_taken = step

            done_val = str(done).lower()
            error_val = error_msg if error_msg else "null"
            print(
                f"[STEP] step={step} action={action_str} reward={float(reward):.2f} done={done_val} error={error_val}",
                flush=True,
            )

            if not done:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": f"reward={reward:.3f}\n" + build_user_message(obs_dict),
                })

            if done:
                raw_score = info.get("final_score") or 0.01
                score = max(0.01, min(0.99, float(raw_score)))
                success = score >= 0.3
                break

            time.sleep(0.3)

    except Exception as e:
        print(f"[DEBUG] Task {task} exception: {e}", file=sys.stderr, flush=True)
        score = 0.01

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.01"
        print(
            f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}",
            flush=True,
        )


def main() -> None:
    for task in ["easy", "medium", "hard"]:
        print(f"\n=== Running task: {task} ===", file=sys.stderr, flush=True)
        try:
            run_task(task)
        except Exception as e:
            print(f"[DEBUG] Fatal: {e}", file=sys.stderr, flush=True)
            print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)
            print(f"[STEP] step=1 action=submit_review reward=0.01 done=true error=fatal", flush=True)
            print(f"[END] success=false steps=1 score=0.010 rewards=0.01", flush=True)


if __name__ == "__main__":
    main()
