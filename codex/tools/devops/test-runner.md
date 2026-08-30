---
id: devops.test-runner
title: Deterministic Test Runner
summary: Select and report the cheapest deterministic verification gates for a local code change.
tags: [devops, tests, pytest, unittest, lint, verification, ci]
---
# Inputs

Changed files, project instructions, and the supported build/test commands.

# Method

Run narrow syntax and unit gates first, then affected integration gates. Keep failures verbatim enough
to reproduce. Distinguish an environmental block from a product failure. Never stage or commit.

# Output

Commands run, concise outcomes, remaining untested risks, and the next smallest recovery action.
