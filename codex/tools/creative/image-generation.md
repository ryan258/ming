---
id: creative.image-generation
title: Need-Driven Image Artist
summary: Turn a communication need into a bounded art brief and one cost-authorized raster image.
tags: [creative, image, illustration, art-direction, visual, hero, openrouter]
---
# Inputs

The communication need, intended use, audience, art direction, things to avoid, frame ratio, and any
exact text that must appear.

# Method

Translate the need into a single-image composition with an immediate focal hierarchy. Default to no
rendered text. Generate one image only. Prefer a currently free route when one exists; otherwise show
the expected cheap-model cost and require explicit per-image authorization before calling the API.

# Output

A PNG, JPEG, or WebP raster confined to `outputs/images/`, plus a JSON provenance manifest containing
the final prompt, requested and actual model, SHA-256 digest, media type, usage, and observed cost.
The preview or generation iteration is also saved beneath `outputs/iterations/`. Human visual review
remains required.
