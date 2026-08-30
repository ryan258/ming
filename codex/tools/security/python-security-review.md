---
id: security.python-review
title: Python Security Review
summary: Review Python boundaries for injection, path traversal, secret exposure, and unsafe deserialization.
tags: [security, python, review, injection, secrets, filesystem]
---
# Inputs

A focused Python snippet, module, or diff plus its trust boundaries.

# Method

Trace untrusted inputs to filesystem, process, SQL, template, network, and deserialization sinks.
Check canonicalized path confinement, parameterized queries, subprocess argument boundaries, secret
redaction, timeout behavior, and error messages. Rank concrete findings by impact and reproducibility.

# Output

A concise severity-ranked review with exact locations, exploit preconditions, and a bounded fix.
Do not claim safety from static pattern matching alone.
