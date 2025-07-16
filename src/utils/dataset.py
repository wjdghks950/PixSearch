import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.segment_anything.utils.transforms import ResizeLongestSide
from pycocotools import mask
from transformers import CLIPImageProcessor


def collate_fn_llama4(batch, processor=None, local_rank=-1, turn_type="single"):
    """
    Note: Either 'image' or 'image_url' will be provided in the dataset, but not necessarily both.
    When the actual image cannot be included, only the image_url will be available.
    The evaluation servers will nevertheless always include the loaded 'image' field.

    Each example contains the following fields:

    {
    "session_id": str,          # Unique identifier for the conversation
    "image": Image,             # Image
    "image_url": str,           # Image URL where applicable
    "turns": {                  # Dictionary containing conversation turn data
        "interaction_id": [str],  # List of unique IDs for each interaction
        "domain": [int],          # List of domain category indices
        "query_category": [int],  # List of query category indices
        "dynamism": [int],        # List of dynamism level indices
        "query": [str],           # List of questions or prompts
        "image_quality": [int]    # List of image quality indices
    },
    "answers": {                # Dictionary containing answer data
        "interaction_id": [str],  # List of interaction IDs (matches turns)
        "ans_full": [str]         # List of complete answer texts
    }
    }

    """
    session_ids = []  # 'session_ids' in cragmm
    images_list = []  # list of PIL.Image images
    image_urls_list = []  # list of image urls
    turns_list = []  # list of turns
    answers_list = []  # list of answers

    if turn_type == "single":
        for example in batch:
            session_ids.append(example["session_id"])
            images_list.append(example["image"])
            image_urls_list.append(example["image_url"])
            turns_list.append(example["turns"])
            answers_list.append(example["answers"])
    else:
        # TODO: Implement 'turn_type' = 'multi' case for multi-turn conversations
        # TODO: Use llama4 processor to change images to tensors of same size
        for example in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                    ],
                }
            ]

    return {
        "session_ids": session_ids,
        "images": images_list,  # torch.stack(images_list, dim=0),
        "image_urls": image_urls_list,
        "turns_list": turns_list,
        "answers_list": answers_list,
    }
