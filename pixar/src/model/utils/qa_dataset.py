# qa_dataset.py
import os
import json
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide


class _QABase(torch.utils.data.Dataset):
    '''
    Text-only QA base dataset that keeps parity with the VQA-style return format:
        (
            image_path: str,                  # "" for text-only
            image: FloatTensor[C,H,W],        # SAM-preprocessed dummy image (zeros) for shape compatibility
            image_clip: FloatTensor[3,h,w],   # CLIP-preprocessed dummy image
            conversations: List[str],         # 1 prompt (or prompt+answer if inference=False)
            masks: FloatTensor[0,H,W],        # empty
            label: FloatTensor[H,W],          # ignore label map
            resize: Tuple[int,int],           # (H,W) after SAM resize
            questions: List[str],             # same as conversations
            sampled_answer_choices: List[str],# normalized references (all golds)
            per_token_label_dict: List[dict], # [{}]
            search_calls_norm: List[dict],    # []
            inference: bool,                  # see above
            dataset_name: str,                # 'nq' | 'hotpotqa' | 'musique' | 'triviaqa' | 'popqa'
        )
    '''
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std  = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size   = 1024
    ignore_label = 255

    def __init__(
        self,
        tokenizer,
        vision_tower: str,
        split: str,
        image_size: int = 1024,
        inference: bool = False,   # if True: user-only prompt; answers in sampled_answer_choices (eval-like)
        limit: Optional[int] = None,
        seed: int = 7,
    ):
        self.tokenizer = tokenizer
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.transform = ResizeLongestSide(image_size)
        self.image_size = image_size
        self._inference = inference
        self._rng = random.Random(seed)
        self._limit = limit

        # Prebuild a single dummy RGB image to reuse
        self._dummy_rgb = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def __len__(self):
        n = len(self.rows)
        return min(n, self._limit) if self._limit is not None else n

    def preprocess_image_pair(self, rgb: np.ndarray):
        '''Produce SAM-resized (for 'image') and CLIP-processed (for 'image_clip').'''
        # CLIP
        image_clip = self.clip_image_processor.preprocess(rgb, return_tensors="pt")["pixel_values"][0]  # [3,h,w]

        # SAM-style path (ResizeLongestSide + normalize + square-pad to 1024)
        resized = self.transform.apply_image(rgb)                          # HxWx3 (np)
        resize_hw = resized.shape[:2]
        image = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float()
        image = self._normalize_and_square_pad(image)                      # [3,1024,1024]
        return image, image_clip, resize_hw

    def _normalize_and_square_pad(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @staticmethod
    def _first_nonempty_str(candidates: List[str]) -> str:
        for s in candidates:
            if isinstance(s, str) and s.strip():
                return s.strip()
        return ""

    def _build_conversation(self, question: str, gold_answers: List[str]) -> List[str]:
        '''
        Build a single-turn conversation. If inference=False, include assistant turn with 1 canonical gold.
        '''
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        # USER: question only (no <image> token for pure text QA)
        # You can add a small instruction if desired:
        text_instructions = "Answer concisely."
        conv.append_message(conv.roles[0], f"{question.strip()} {text_instructions}")

        if not self._inference:
            # ASSISTANT: provide one canonical gold (first non-empty)
            canonical = self._first_nonempty_str(gold_answers) if len(gold_answers) > 0 else ""
            conv.append_message(conv.roles[1], canonical)

        return [conv.get_prompt()]

    # each subclass must implement: _hf_load() and _extract_qa(i) -> (question, answers_list)

    def __getitem__(self, idx: int):
        item = self.rows[idx]
        question, answers = self._extract_qa(item)

        # some questions don't end with a question mark - adding one here
        if not question.strip().endswith("?"):
            question += "?"

        if isinstance(answers, str):
            answers = [answers]
        answers = [str(a).strip() for a in answers if str(a).strip()]
        conversations = self._build_conversation(question, answers)
        questions = conversations

        # Dummy image pair to satisfy collate_fn structure
        image, image_clip, resize_hw = self.preprocess_image_pair(self._dummy_rgb)

        # No masks/labels for text QA
        H, W = self._dummy_rgb.shape[:2]
        masks = torch.rand(0, H, W)                  # empty
        label = torch.ones(H, W) * self.ignore_label # ignore
        per_token_label_dict = [{}]
        search_calls_norm = []

        image_path = ""  # text-only
        dataset_name = self._dataset_name

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize_hw,
            questions,
            answers,                 # sampled_answer_choices
            per_token_label_dict,
            search_calls_norm,
            self._inference,
            dataset_name,
        )


class NaturalQuestionsDataset(_QABase):
    '''
    Hugging Face: "natural_questions"
    Common fields across configs:
      - 'question': str
      - Gold answers may appear as:
        * item['answers']['text'] (list[str])
        * item['annotations']['short_answers'] -> list[{text,start_token,end_token}]
        * item['answer'] (str)
    We'll try several fallbacks.
    '''
    def __init__(self, tokenizer, vision_tower, split: str = "validation", **kwargs):
        super().__init__(tokenizer, vision_tower, split, **kwargs)
        # Many mirrors exist; the common one is "natural_questions".
        # If split not available, fallback to "train".

        ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")
        if split in ["validation", "val"]:
            ds = ds["dev"]
        elif split in ["test"]:
            ds = ds["test"]
        else:  # train
            ds = ds["train"]

        self.rows = ds
        self._dataset_name = "nq"
        self._inference = True if split == "validation" else False

        print(f"Number of samples in NQ: {len(self.rows)}")

    def _extract_qa(self, item: Dict[str, Any]) -> Tuple[str, List[str]]:
        q = item.get("question", "")
        a = item.get("golden_answers", "")
        if isinstance(a, str) and a.strip():
            answers = [a]
        else:
            answers = a
        return q, answers if answers else [""]


class HotpotQADataset(_QABase):
    '''
    Hugging Face: "hotpot_qa"
    Typical fields:
      - 'question': str
      - 'answer': str
      - 'split': str ('fullwiki' or 'distractor')
    '''
    def __init__(self, tokenizer, vision_tower, split: str = "validation", **kwargs):
        super().__init__(tokenizer, vision_tower, split, **kwargs)
        if split in ["val", "validation"]:
            split = "validation"
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        self.rows = ds
        self._dataset_name = "hotpotqa"
        self._inference = True if split == "validation" else False

    def _extract_qa(self, item: Dict[str, Any]) -> Tuple[str, List[str]]:
        q = item.get("question", "")
        a = item.get("answer", "")
        supfacts = item.get("supporting_facts", {})
        contexts = item.get("context", {})
        answers = [a] if isinstance(a, str) and a.strip() else []
        return q, answers if answers else [""]


class MuSiQueDataset(_QABase):
    '''
    Hugging Face: "musique"
    Common fields (vary by config):
      - 'question': str
      - answers may be under:
        * 'final_answer' (str)
        * 'answers' (list[str])
        * 'answer'  (str)
    '''
    def __init__(self, tokenizer, vision_tower, split: str = "validation", **kwargs):
        super().__init__(tokenizer, vision_tower, split, **kwargs)
        ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "musique")
        if split in ["validation", "val"]:
            ds = ds["dev"]
        else:  # train
            ds = ds["train"]

        self.rows = ds
        self._dataset_name = "musique"
        self._inference = True if split == "validation" else False

        print(f"Number of samples in MuSiQue: {len(self.rows)}")

    def _extract_qa(self, item: Dict[str, Any]) -> Tuple[str, List[str]]:
        q = item.get("question", "")
        a = item.get("golden_answers", "")
        if isinstance(a, str) and a.strip():
            answers = [a]
        else:
            answers = a
        return q, answers if answers else [""]


class TriviaQADataset(_QABase):
    '''
    Hugging Face: "trivia_qa"
    Typical fields:
      - 'question': str
      - 'answer' can be:
        * dict with 'value' (str) and possibly 'aliases' (list[str])
        * sometimes directly str
    '''
    def __init__(self, tokenizer, vision_tower, split: str = "validation", **kwargs):
        super().__init__(tokenizer, vision_tower, split, **kwargs)
        # Many mirrors exist; the common one is "natural_questions".
        # If split not available, fallback to "train".

        ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "triviaqa")
        if split in ["validation", "val"]:
            ds = ds["dev"]
        elif split in ["test"]:
            ds = ds["test"]
        else:  # train
            ds = ds["train"]

        self.rows = ds
        self._dataset_name = "triviaqa"
        self._inference = True if split == "validation" else False

        print(f"Number of samples in TriviaQA: {len(self.rows)}")

    def _extract_qa(self, item: Dict[str, Any]) -> Tuple[str, List[str]]:
        q = item.get("question", "")
        a = item.get("golden_answers", "")
        if isinstance(a, str) and a.strip():
            answers = [a]
        else:
            answers = a
        return q, answers if answers else [""]


class PopQADataset(_QABase):
    '''
    Hugging Face: "popqa"
    Typical fields:
      - 'question': str
      - 'answer' can be:
        * dict with 'value' (str) and possibly 'aliases' (list[str])
        * sometimes directly str
    '''
    def __init__(self, tokenizer, vision_tower, split: str = "validation", **kwargs):
        super().__init__(tokenizer, vision_tower, split, **kwargs)
        # Many mirrors exist; the common one is "natural_questions".
        # If split not available, fallback to "train".

        ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "popqa")
        if split in ["validation", "val", "test"]:
            ds = ds["test"]
        else:  # train
            ds = ds["train"]

        self.rows = ds
        self._dataset_name = "popqa"
        self._inference = True if split == "validation" else False

        print(f"Number of samples in PopQA: {len(self.rows)}")

    def _extract_qa(self, item: Dict[str, Any]) -> Tuple[str, List[str]]:
        q = item.get("question", "")
        a = item.get("golden_answers", "")
        if isinstance(a, str) and a.strip():
            answers = [a]
        else:
            answers = a
        return q, answers if answers else [""]



DATASET_NAME_TO_CLASS = {
    "nq": NaturalQuestionsDataset,
    "natural_questions": NaturalQuestionsDataset,
    "hotpotqa": HotpotQADataset,
    "hotpot_qa": HotpotQADataset,
    "musique": MuSiQueDataset,
    "triviaqa": TriviaQADataset,
    "trivia_qa": TriviaQADataset,
    "popqa": PopQADataset,
}

def build_qa_dataset(
    name: str,
    tokenizer,
    vision_tower: str,
    split: str = "validation",
    image_size: int = 1024,
    inference: bool = False,
    limit: Optional[int] = None,
    **kwargs,
):
    '''
    name: one of {'nq','natural_questions','hotpotqa','hotpot_qa','musique','triviaqa','trivia_qa'}
    '''
    key = name.lower()
    if key not in DATASET_NAME_TO_CLASS:
        raise ValueError(f"[qa_dataset] Unknown dataset '{name}'. "
                         f"Valid: {list(DATASET_NAME_TO_CLASS.keys())}")
    cls = DATASET_NAME_TO_CLASS[key]
    return cls(
        tokenizer=tokenizer,
        vision_tower=vision_tower,
        split=split,
        image_size=image_size,
        inference=inference,
        limit=limit,
        **kwargs,
    )
