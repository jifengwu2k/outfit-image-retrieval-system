# Outfit Image Retrieval System

## Overview

An agentic system maintaining collection of **outfit images** that can be **retrieved** with **natural-language queries**. When new outfit images are **added**, the system computes a **perceptual hash**, generates a **description**, and stores both for later **retrieval**.

## Main use cases

1. **Add outfit images**
2. **Retrieve outfit images**

## Repository layout

- `add_outfit_images.py` — command-line entrypoint for adding outfit images
- `images/` — outfit image storage directory (JPEG format, perceptual hashes as filenames)
- `outfit_descriptions.json` — JSON metadata mapping outfit image perceptual hashes to descriptions
- `requirements.txt` — Python dependencies

## Add outfit images

`add_outfit_images.py`:

- accepts one or more outfit image paths;
- computes a perceptual hash for each outfit image;
- generates a description with a Hugging Face multimodal model;
- stores the image in `images/` as `<perceptual hash>.jpg`;
- stores the mapping from perceptual hash to description in `outfit_descriptions.json`.

### Model interface

Description generation uses the Hugging Face multimodal APIs:

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM
```

The entrypoint requires:

- `--pretrained-transformers-multimodal-lm-model`

which specifies the pretrained multimodal model used to describe images. The selected model must be compatible with `AutoProcessor` and `AutoModelForMultimodalLM`.

### Install

```bash
pip install -r requirements.txt
```

### Usage

```bash
python add_outfit_images.py \
  --device cuda \
  --pretrained-transformers-multimodal-lm-model Qwen/Qwen2.5-VL-3B-Instruct \
  path/to/look1.jpg path/to/look2.png
```

Optional flags:

- `--prompt` — customize the description instruction
- `--overwrite-existing-description` — regenerate metadata for an existing perceptual hash

If an image with the same perceptual hash is already indexed, it is skipped unless `--overwrite-existing-description` is provided.

### Output format

`outfit_descriptions.json` is a JSON file mapping outfit image perceptual hashes to descriptions.

Example:

```json
{
  "abc123def456": "navy blazer over a white shirt with dark trousers"
}
```

Indexed images are stored canonically in `images/` as `<perceptual hash>.jpg`.

## Retrieving relevant outfit images

Retrieval is designed around agentic workflows rather than a dedicated retrieval script. In practice, you can implicitly program a user's agent of choice, such as Claude Code, with an `AGENTS.md` file that instructs the agent to:

- take the user's natural-language query;
- query `outfit_descriptions.json` for the most relevant descriptions;
- identify the matching perceptual hashes;
- fetch the corresponding outfit images from `images/` using those perceptual hashes.

This makes the retrieval layer flexible: an agent can translate free-form user intent into the concrete steps needed to locate and return the best matching outfit images.

## Philosophy

This project deliberately embraces the uncertainty and fluidity of the agentic AI era.

Rather than hard-coding a rigid retrieval pipeline, it keeps the core representation simple, legible, and easy for both humans and agents to work with:

- outfit images are stored canonically by perceptual hash;
- descriptions are stored in a plain JSON mapping;
- retrieval is performed semantically from natural-language queries rather than through a brittle fixed taxonomy.

The goal is not to prematurely optimize for a single retrieval architecture, but to create a durable substrate that capable agents can interpret, inspect, and use flexibly. In that sense, the repository is intentionally minimal: it favors transparent artifacts and stable conventions over heavy infrastructure.

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
