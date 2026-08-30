---
id: data.structured-summary
title: Structured Summary Worker
summary: Transform bounded input into deterministic JSON with source-grounded facts and explicit unknowns.
tags: [data, json, summary, schema, extraction, validation]
---
# Inputs

Source text and a JSON-compatible target schema.

# Method

Extract only supported facts. Use null or an explicit unknowns list for absent information. Validate
required keys, value types, and the absence of unexpected fields before returning the object.

# Output

Valid JSON only when JSON is requested. Never wrap JSON in Markdown fences.
