---
title: Code Review Env
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# CodeReviewEnv

An OpenEnv-compliant RL environment for code review.

## API
- POST /reset
- POST /step
- GET /state
- GET /health

## Tasks
- Easy: Python service with SQL injection
- Medium: Express.js API with auth bypass
- Hard: Concurrency bugs and RCE