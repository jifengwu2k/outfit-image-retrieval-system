# AGENTS.md

## Purpose

This repository stores outfit images, perceptual hashes, and descriptions for agent-assisted retrieval. Your job is to help a user find relevant outfit images from a natural-language query.

## Retrieval interface

Use these files as the retrieval interface:

- `outfit_descriptions.json` — maps outfit image perceptual hashes to descriptions
- `images/` — stores the corresponding outfit images as `images/<perceptual hash>.jpg`

## How to retrieve relevant outfit images

When a user asks for an outfit:

1. Read the user's natural-language query.
2. Read `outfit_descriptions.json` and identify the most relevant descriptions. Use semantic matching rather than exact keyword matching. Then identify the corresponding perceptual hash values.
3. Fetch the corresponding images from `images/<perceptual hash>.jpg`.
4. Return the best matches first, including the image path and a brief explanation.

Do not modify repository files during retrieval.
