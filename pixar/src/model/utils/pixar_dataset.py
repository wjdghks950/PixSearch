import json
import os
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor

from pycocotools import mask
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import DEFAULT_IMAGE_TOKEN, _imread_rgb_sans_icc

# helpers
def _resize_mask_like(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1), int(y1)


# ------------------------------
# PixarDataset
# ------------------------------

class PixarDataset(torch.utils.data.Dataset):
    """
    JSONL-driven dataset for region/image/text retrieval per <search>... </search>.

    Returns the SAME tuple fields and order as SemSegDataset:
        (
            image_path,         # str
            image,              # Tensor [3, H_sam_pad(=1024), W_sam_pad(=1024)] normalized + square-padded
            image_clip,         # Tensor [3, H_clip, W_clip] (CLIP-preprocessed)
            conversations,      # List[str] prompts (constructed as in SemSeg)
            masks,              # Tensor [K, H_sam, W_sam] (REGION qtype masks), or empty
            label,              # Tensor [H_sam, W_sam] (ignore_label), or dummy
            resize,             # (H_sam, W_sam) before padding
            questions,          # List[str] (uses decomposed_questions if provided; else [question])
            sampled_classes,    # List[str] labels for REGION masks (from region_query_span)
            per_token_label_dict,  # [{label: mask_tensor}, ...] aligned with conversations length
        )

    PLUS two appended items:
        search_calls_norm: List[Dict]  # normalized search_calls with resized masks, positions, qtype, query, spans, etc.
        retrieval_items:   List[Dict]  # retriever-ready items:
                                        #   {'modality': 'text'|'region'|'image',
                                        #    'position': int,
                                        #    'query_text': Optional[str],
                                        #    'region_mask': Optional[Tensor[H_sam, W_sam]],
                                        #    'clip_image': Optional[Tensor[3,Hc,Wc]]}
    """

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)   # 0–255 space
    pixel_std  = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size   = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir: str,
        tokenizer,
        vision_tower: str,
        precision: str = "fp32",
        image_size: int = 224,
        pixar_data="oven||infoseek||aokvqa||okvqa||textvqa||webqa",
    ):
        self.base_image_dir = base_image_dir
        self.precision = precision
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.pixar_data = pixar_data.split("||")
        self.transform = ResizeLongestSide(image_size)  # SAM-style resize (longest side -> image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.jsonl_version = "v2"

        self.samples: List[Dict[str, Any]] = []
        for ds in self.pixar_data:
            with open(os.path.join(self.base_image_dir, ds, f"{ds}_retrieve_k_5_interleaved_{self.jsonl_version}.jsonl"), "r") as reader:
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    # Image preprocess (SAM normalization + square padding to 1024)
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):

        s = self.samples[idx]

        image_path = s.get("image_path", "")
        if isinstance(image_path, list):
            image_path = image_path[0]
        image_path_suffix = "/".join(image_path.split("/")[-3:])
        image_path = os.path.join(self.base_image_dir, image_path_suffix)
        image_file_exts = ["jpg", "png", "jpeg", "JPEG", "JPG", "PNG", "webp"]
        if not image_path.endswith(tuple(image_file_exts)):
            image_path = [f"{image_path}.{ext}" for ext in image_file_exts if os.path.exists(f"{image_path}.{ext}")][0]

        question = s.get("question", "")
        decomposed_questions = s.get("decomposed_questions", None)
        answer_text = s.get("answer", s.get("answer_draft", ""))  # e.g., "The building in <search> ... </search> <information> ... </information> is a ..."
        search_calls = s.get("search_calls", [])

        # load transform image
        img_rgb = _imread_rgb_sans_icc(image_path)

        # clip image processing
        clip_proc = self.clip_image_processor.preprocess(img_rgb, return_tensors="pt")
        image_clip = clip_proc["pixel_values"][0]  # [3, Hc, Wc]

        # sam image processing
        image_resized = self.transform.apply_image(img_rgb)      # H x W x 3 (uint8)
        resize = image_resized.shape[:2]                         # (H_sam, W_sam)
        image = torch.from_numpy(image_resized).permute(2, 0, 1).contiguous()
        image = self.preprocess(image)                           # [3, 1024, 1024]

        # build conversations (similar to SemSegDataset)
        conversations: List[str] = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []

        # NOTE: Check if there are <image> tokens in the question and replace with "<img>"
        question = question.replace("<image>", "<img>")
        answer_text = answer_text.replace("<image>", "<img>")

        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], answer_text)
        conversations.append(conv.get_prompt())

        # normalize search calls and prepare masks
        region_masks: List[torch.Tensor] = []
        region_labels: List[str] = []
        region_label_offsets: List[List[int]] = []

        search_calls_norm: List[Dict[str, Any]] = []

        for call in search_calls:
            qtype = (call.get("qtype", "") or "").upper()  # 'REGION' | 'IMAGE' | 'TEXT'
            pos   = int(call.get("position", -1))
            qtext = call.get("query_text", "")

            # region_query span metadata
            region_query_span = call.get("region_query_span", None)
            region_query_preceding_span = call.get("region_query_preceding_span", None)
            region_query_preceding_span_offsets = call.get("region_query_preceding_span_offsets", None)
            region_query_span_source = call.get("region_query_span_source", None)
            region_query_span_offsets = call.get("region_query_span_offsets", None)
            qtype_reasoning = call.get("qtype_reasoning", None)

            # new mask_path structure: list of dicts with {'bbox': path, 'mask': path}
            crop_paths = call.get("mask_path", [])

            entry: Dict[str, Any] = {
                "position": pos,
                "qtype": qtype,
                "qtype_reasoning": qtype_reasoning,
                "query_text": qtext,
                "masks": [],
                "mask_labels": [],
                "mask_labels_offsets": [],
                "crop_paths": crop_paths,
            }

            if qtype == "REGION":
                # NOTE: text span for mask comes from 'region_query_span' (e.g., "the city shown in the image")
                label_text = region_query_span if region_query_span is not None else region_query_preceding_span
                label_text_offsets = region_query_span_offsets if region_query_span is not None else region_query_preceding_span_offsets
                for m in call.get("mask", []):
                    try:
                        rle = m
                        if not isinstance(rle["counts"], bytes):
                            rle["counts"] = rle["counts"].encode()
                        mask_np = mask.decode(rle)
                        mask_np = mask_np.astype(np.uint8)
                    except Exception:
                        mask_np = np.zeros(resize, dtype=np.uint8)

                    # resize to sam grid
                    mask_rs = _resize_mask_like(mask_np, resize[0], resize[1])  # [H_sam, W_sam]
                    mask_t = torch.from_numpy(mask_rs.astype(np.uint8))

                    entry["masks"].append(mask_t)
                    entry["mask_labels"].append(label_text)
                    entry["mask_labels_offsets"].append(label_text_offsets)

                    region_masks.append(mask_t)
                    region_labels.append(label_text)
                    region_label_offsets.append(label_text_offsets)

            elif qtype == "IMAGE":
                pass

            elif qtype == "TEXT":
                pass

            search_calls_norm.append(entry)

        # masks: stack REGION masks (if any); otherwise empty [0, H_sam, W_sam]
        if len(region_masks) > 0:
            masks = torch.stack(region_masks, dim=0)  # [K, H_sam, W_sam]
            sampled_classes = region_labels  # from region_query_span
        else:
            masks = torch.zeros((0, resize[0], resize[1]), dtype=torch.uint8)
            sampled_classes = []

        # label: dummy ignore map (SemSegDataset provides 2D map)
        label = torch.ones(resize[0], resize[1], dtype=torch.long) * self.ignore_label

        questions = [question] if question else []

        # per_token_label_dict: [{class: mask_tensor}, ...] aligned with conversations
        # TODO: the label to mask mapping should be a single dict, not a list of dicts
        if len(sampled_classes) > 0:
            per_token_label_dict = [{lbl: m for lbl, m in zip(sampled_classes, masks)}]
            per_token_label_dict += [{} for _ in range(max(0, len(conversations) - len(per_token_label_dict)))]
        else:
            per_token_label_dict = [{} for _ in range(len(conversations))]

        return (
            image_path,           # str
            image,                # Tensor [3, 1024, 1024]
            image_clip,           # Tensor [3, Hc, Wc]
            conversations,        # List[str]
            masks,                # Tensor [K, H_sam, W_sam]
            label,                # Tensor [H_sam, W_sam] (255)
            resize,               # (H_sam, W_sam) pre-pad
            questions,            # List[str]
            sampled_classes,      # List[str] (from region_query_span)
            per_token_label_dict, # List[Dict[str, Tensor]]
            search_calls_norm,    # List[Dict], normalized with spans/paths
            # TODO: Decide whether to include "retrieved_items" in the return tuple
        )
