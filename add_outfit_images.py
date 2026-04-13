#!/usr/bin/env python3
# Copyright (c) 2026 Jifeng Wu
# Licensed under the Apache-2.0 License. See LICENSE file in the project root for full license information.
'''Command-line entrypoint for adding new outfit images.'''
from __future__ import print_function
import argparse
import base64
import codecs
import io
import json
import os
from typing import Dict

import imagehash
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images')
METADATA_JSON = os.path.join(SCRIPT_DIR, 'outfit_descriptions.json')
DEFAULT_PROMPT = 'Describe the outfit in this image.'


def load_existing_records():
    # type: () -> Dict[str, str]
    if not os.path.exists(METADATA_JSON):
        return {}

    with codecs.open(METADATA_JSON, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_records(records):
    # type: (Dict[str, str]) -> None
    with codecs.open(METADATA_JSON, 'w', encoding='utf-8') as file:
        json.dump(records, file, indent=2, ensure_ascii=False, sort_keys=True)


def main():
    # type: () -> int
    parser = argparse.ArgumentParser(
        description='Add outfit images, compute perceptual hashes and index descriptions.'
    )
    parser.add_argument(
        'images',
        nargs='+',
        help='Paths to one or more image files to add.',
    )
    parser.add_argument(
        '--pretrained-transformers-multimodal-lm-model',
        required=True,
        help='Hugging Face model name or local path for AutoModelForMultimodalLM.',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='Torch device to use for multimodal inference (default: cpu).',
    )
    parser.add_argument(
        '--prompt',
        default=DEFAULT_PROMPT,
        help='Instruction prompt used for image description generation (default: %s).' % (DEFAULT_PROMPT,),
    )
    parser.add_argument(
        '--overwrite-existing-description',
        action='store_true',
        help='If a perceptual hash already exists in the JSON, regenerate and overwrite its description.',
    )

    args = parser.parse_args()

    image_paths = args.images

    if not os.path.isdir(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    existing_records = load_existing_records()

    device = torch.device(args.device)

    print('Using device:', device)
    print('Loading multimodal model:', args.pretrained_transformers_multimodal_lm_model)

    processor = AutoProcessor.from_pretrained(args.pretrained_transformers_multimodal_lm_model)

    model = AutoModelForMultimodalLM.from_pretrained(
        args.pretrained_transformers_multimodal_lm_model,
    )
    model = model.to(device)
    model.eval()

    processed = 0
    skipped = 0

    for image_path in image_paths:
        opened_image = Image.open(image_path)
        image = opened_image.convert('RGB')
        opened_image.close()

        phash = str(imagehash.phash(image))

        existing = existing_records.get(phash)
        if existing and not args.overwrite_existing_description:
            skipped += 1
            print('[skip] %s -> phash=%s already indexed' % (image_path, phash))
            continue

        image_buffer = io.BytesIO()
        image.save(image_buffer, format='JPEG')
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
        image_url = 'data:image/jpeg;base64,' + image_base64

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'url': image_url},
                    {'type': 'text', 'text': args.prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            add_generation_prompt=True,
        ).to(model.device)
        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(**inputs)

        response = processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=False,
        )
        description = response

        stored_path = os.path.join(IMAGES_DIR, '%s.jpg' % (phash,))
        if not os.path.exists(stored_path):
            image.save(stored_path, format='JPEG')

        existing_records[phash] = description
        save_records(existing_records)

        processed += 1
        print('[ok] %s -> %s (phash=%s)' % (image_path, stored_path, phash))
        print('     description: %s' % (description,))

    print('Done. processed=%s skipped=%s' % (processed, skipped))


if __name__ == '__main__':
    main()
