import json
import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import CLIPImageProcessor
import requests
from PIL import Image

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN


def preprocess_multimodal(source, mm_use_im_start_end, is_textvqa=False):
    if is_textvqa:
        source = DEFAULT_IMAGE_TOKEN + "\n" + source
        source = source.strip()
        if "mmtag" in conversation_lib.default_conversation.version:
            source = source.replace(
                DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
            )
    else:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                    )
    return source


class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="llava_v1_5_mix665k"
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.vqa_data = vqa_data

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        if vqa_data == "llava_instruct_150k":
            self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")  # for 'llava_instruct_150k.json'
        elif vqa_data == "llava_v1_5_mix665k":
            self.vqa_image_root = base_image_dir  # for 'llava_v1_5_mix665k.json'
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_dataset = json.load(f)
        self.vqa_dataset = vqa_dataset

        print("vqa_dataset: ", len(self.vqa_dataset))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_dataset) - 1)
        item = self.vqa_dataset[idx]
        image_path = item.get("image", "")
        _text_only = False  # _text_only == True, when item['image'] is not there
        if image_path:
            if self.vqa_data == "llava_instruct_150k":
                image_path = os.path.join(self.vqa_image_root, os.path.basename(item["image"]))
            elif self.vqa_data == "llava_v1_5_mix665k":
                image_path = os.path.join(self.vqa_image_root, item["image"])
            image = cv2.imread(image_path)
        else:
            _text_only = True
            image = None

        if image is None:
            # skip this sample or load a default image
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]  # preprocess image for clip

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
            is_textvqa=False
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

        per_token_label_dict = [{}]  # VQADataset does not have masks - ignore during training
        search_calls_norm = []  # VQADataset doees not have search_calls_norm
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            per_token_label_dict,
            search_calls_norm
        )


class TextVQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="TextVQA_0.5.1_val",
        use_mm_start_end=False,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        DATA_DIR = os.path.join(base_image_dir, "textvqa")
        self.textvqa_image_root = os.path.join(base_image_dir, "textvqa/images")
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            textvqa_data = json.load(f)
        self.textvqa_data = textvqa_data["data"]
        self.dataset_type = textvqa_data["dataset_type"]
        self.dataset_name = textvqa_data["dataset_name"]
        self.dataset_version = textvqa_data["dataset_version"]
        self.use_mm_start_end = use_mm_start_end

        print("TextVQA_data: ", len(self.textvqa_data))

    def __len__(self):
        return len(self.textvqa_data)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        item = self.textvqa_data[idx]
        image_path = os.path.join(self.textvqa_image_root, f"{item['image_id']}.jpg")
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros((1024, 1024, 3), dtype=np.uint8) # handle the error, e.g., skip this sample or load a default image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # Process image for CLIP
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # Process image for SAM
        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        # Process conversation
        conv = conversation_lib.default_conversation.copy()
        source = item["question"]
        if self.use_mm_start_end:
            mm_use_im_start_end = conv.sep_style == conversation_lib.SeparatorStyle.TWO
        else:
            mm_use_im_start_end = False  # for LLaVA only evaluation
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=mm_use_im_start_end,
            is_textvqa=True
        )

        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []

        conv.messages = []
        role = conv.roles[0]

        text_instructions = 'Answer the question using a single word or phrase.'
        conv.append_message(role, source + ' ' + text_instructions)  # https://github.com/haotian-liu/LLaVA/issues/515
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_answer_choices = item["answers"]

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # TextVQA dataset doesn't use masks during training
        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
        per_token_label_dict = [{}]
        search_calls_norm = []

        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_answer_choices,
            per_token_label_dict,
            search_calls_norm,
            inference,
            'textvqa'
        )


class GQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(self, base_image_dir, tokenizer, vision_tower, split="val", use_mm_start_end=False):
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.split = split   # e.g., "test", "testdev"
        self.use_mm_start_end = use_mm_start_end
        DATA_DIR = os.path.join(base_image_dir, "gqa")
        with open(os.path.join(DATA_DIR, "questions", f"{split}_balanced_questions.json")) as f:
            self.gqa_data = json.load(f)

        self.limit_num_samples = 5000
        self.gqa_data_key_indices = {idx: instance_id for idx, (instance_id, instance_dict) in enumerate(self.gqa_data.items()) if idx < self.limit_num_samples}

        print(f"GQA_{split}_data: ", len(self.gqa_data_key_indices))

    def __len__(self):
        return len(self.gqa_data_key_indices)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        item = self.gqa_data[self.gqa_data_key_indices[idx]]
        image_path = os.path.join(self.base_image_dir, "gqa/images", f"{item['imageId']}.jpg")
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # Process image for CLIP
        clip_image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower)
        image_clip = clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # Process image for SAM
        transform = ResizeLongestSide(224)
        image = transform.apply_image(image)
        resize = image.shape[:2]

        # Process conversation
        conv = conversation_lib.default_conversation.copy()
        source = item["question"]
        if self.use_mm_start_end:
            mm_use_im_start_end = conv.sep_style == conversation_lib.SeparatorStyle.TWO
        else:
            mm_use_im_start_end = False  # for LLaVA only evaluation
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=mm_use_im_start_end,
            is_textvqa=True
        )

        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []

        conv.messages = []
        role = conv.roles[0]

        text_instructions = 'Answer the question using a single word or phrase.'
        conv.append_message(role, source + ' ' + text_instructions)
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_answer_choices = [item["answer"], item["fullAnswer"]]

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # GQA has no masks
        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
        per_token_label_dict = [{}]
        search_calls_norm = []
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_answer_choices,
            per_token_label_dict,
            search_calls_norm,
            inference,
            'gqa'
        )


class POPEDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(self, base_image_dir, tokenizer, vision_tower, split="val", use_mm_start_end=False):
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.split = split
        self.use_mm_start_end = use_mm_start_end

        # Setup paths
        DATA_DIR = os.path.join(base_image_dir, "pope/coco")
        self.image_root = os.path.join(DATA_DIR, "val2014")  # POPE uses COCO val2014 images

        # Load POPE questions
        pope_types = ["random", "popular", "adversarial"]
        self.pope_data = []

        for pope_type in pope_types:
            data = []
            with open(os.path.join(DATA_DIR, f"coco_pope_{pope_type}.json")) as f:
                print(f"Loading POPE {pope_type} data from {os.path.join(DATA_DIR, f'coco_pope_{pope_type}.json')}")
                for line in f:
                    line = line.strip()
                    data.append(json.loads(line))
                for item in data:
                    item["pope_type"] = pope_type
                self.pope_data.extend(data)

        print(f"POPE_{split}_data: ", len(self.pope_data))

    def __len__(self):
        return len(self.pope_data)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        item = self.pope_data[idx]
        image_path = os.path.join(self.image_root, item['image'])  # e.g., "COCO_val2014_000000000001.jpg"
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # Process image for CLIP
        clip_image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower)
        image_clip = clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # Process image for SAM
        transform = ResizeLongestSide(224)
        image = transform.apply_image(image)
        resize = image.shape[:2]

        # Process conversation
        conv = conversation_lib.default_conversation.copy()
        source = item["text"]
        if self.use_mm_start_end:
            mm_use_im_start_end = conv.sep_style == conversation_lib.SeparatorStyle.TWO
        else:
            mm_use_im_start_end = False  # for LLaVA only evaluation
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=mm_use_im_start_end,
            is_textvqa=True
        )

        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []

        conv.messages = []
        role = conv.roles[0]

        text_instructions = 'Please answer the question with "Yes" or "No".'
        conv.append_message(role, source + ' ' + text_instructions)
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_answer_choices = item["label"]  # POPE is a binary yes/no VQA task ['yes', 'no']

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # POPE has no masks
        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
        per_token_label_dict = [{}]

        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_answer_choices,
            per_token_label_dict,
            inference,
            'pope',
        )


class CragMMDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 1024,
        exclude_val=False,
        use_mm_start_end=False,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.crag_mm_data = load_dataset("crag-mm-2025/crag-mm-single-turn-public", revision="v0.1.2")  # TODO: cache the dataset
        self.split = "validation"
        self.dataset = self.crag_mm_data[self.split]
        self.use_mm_start_end = use_mm_start_end

    def __len__(self):
        return len(self.dataset)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle PIL Image directly since item['image'] is a PIL Image object
        if isinstance(item['image'], Image.Image):
            image = cv2.cvtColor(np.array(item['image']), cv2.COLOR_RGB2BGR)
            image_path = f"crag_mm_image_{item['session_id']}"
        else:
            # Fallback to file path handling if it's a string
            image_path = os.path.join(self.base_image_dir, "crag_mm", "images", f"{item['image']}")
            image = cv2.imread(image_path)
            if image is None:  # if the image is not available, use the image_url to download the image
                image_url = item.get('image_url', None)
                if image_url is None:
                    # Create a default black image if no URL is available
                    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
                else:
                    try:
                        image = Image.open(requests.get(image_url, stream=True).raw)
                        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"Error downloading image from {image_url}: {e}")
                        image = np.zeros((1024, 1024, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        conv = conversation_lib.default_conversation.copy()
        num_turns = len(item['turns']['query'])
        is_single_turn = num_turns == 1
        print(f"Type: {'Single-turn' if is_single_turn else 'Multi-turn'} ({num_turns} turns)")

        if is_single_turn:
            source = item["turns"]["query"][0]
        else:
            raise NotImplementedError("Multi-turn is not supported yet")

        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
            is_textvqa=True
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        conv.messages = []
        role = conv.roles[0]
        conv.append_message(role, source)
        conversations.append(conv.get_prompt())

        questions = conversations

        answer_lookup = {}
        if 'answers' in item and item['answers'] is not None:
            answer_lookup = {
                interaction_id: ans_full
                for interaction_id, ans_full in zip(
                    item['answers']['interaction_id'],
                    item['answers']['ans_full']
                )
            }

        sampled_answer_choices = []
        if is_single_turn:
            interaction_id = item['turns']['interaction_id'][0]
            if interaction_id in answer_lookup:
                sampled_answer_choices.append(answer_lookup[interaction_id])
            else:
                sampled_answer_choices.append("")  # Default empty answer if not found
        else:
            raise NotImplementedError("Multi-turn is not supported yet")

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # TODO: CragMM dataset doesn't use masks during evaluation
        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
        per_token_label_dict = [{}]
        search_calls_norm = []
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_answer_choices,
            per_token_label_dict,
            search_calls_norm,
            inference,
            'cragmm'
        )


class InfoseekDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        split_file: str = "infoseek_val.jsonl",
        image_size: int = 1024,
        use_mm_start_end: bool = False,
    ):
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.use_mm_start_end = use_mm_start_end

        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.root = os.path.join(base_image_dir, "infoseek")
        # self.image_root = os.path.join(self.root, "images")  # "/home/jeonghkim/projects/pixar/data/infoseek/images"
        self.image_root = os.path.join("/home/jeonghkim/projects/pixar/data/infoseek", "images")
        split_path = os.path.join(self.root, split_file)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"[InfoseekDataset] split file not found: {split_path}")

        # load jsonl
        self.rows = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        print(f"InfoSeek ({os.path.basename(split_path)}): {len(self.rows)}")

    def __len__(self):
        return len(self.rows)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @staticmethod
    def _find_image(image_root: str, image_id: str):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = os.path.join(image_root, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
        candidates = []
        stem_glob = os.path.join(image_root, f"{image_id}.*")
        candidates.extend(sorted(glob.glob(stem_glob)))
        return candidates[0] if candidates else None

    def __getitem__(self, idx: int):
        item = self.rows[idx]

        image_id = item.get("image_id", "")
        question = item.get("question", "")

        # Normalize answer field to list[str]
        raw_ans = item.get("answer", [])
        if isinstance(raw_ans, str):
            sampled_answer_choices = [raw_ans]
        elif isinstance(raw_ans, list):
            sampled_answer_choices = raw_ans
        else:
            sampled_answer_choices = [str(raw_ans)]

        # load image
        image_path = self._find_image(self.image_root, image_id)
        if image_path is None:
            image_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            image_path = ""
        else:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                image_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                image_path = ""
            else:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        ori_size = image_rgb.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image_rgb, return_tensors="pt")["pixel_values"][0]

        image_resized = self.transform.apply_image(image_rgb)
        resize = image_resized.shape[:2]
        image = torch.from_numpy(image_resized).permute(2, 0, 1).contiguous().float()
        image = self.preprocess(image)

        conv = conversation_lib.default_conversation.copy()
        mm_use = conv.sep_style == conversation_lib.SeparatorStyle.TWO if self.use_mm_start_end else False
        source = preprocess_multimodal(
            question,
            mm_use_im_start_end=mm_use,
            is_textvqa=True
        )
        conv.messages = []
        role = conv.roles[0]
        # instruction = 'Answer the question using a single word or phrase.'
        # conv.append_message(role, source + ' ' + instruction)
        conv.append_message(role, source)
        conversations = [conv.get_prompt()]

        questions = conversations

        # No masks for VQA eval
        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
        per_token_label_dict = [{}]
        search_calls_norm = []
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_answer_choices,
            per_token_label_dict,
            search_calls_norm,
            inference,
            "infoseek",
        )


class OvenDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        mode: str = "entity",  # "entity", "query"
        split: str = "val",
        image_size: int = 1024,
        use_mm_start_end: bool = False,
    ):
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.mode = mode.lower()
        assert self.mode in ("entity", "query"), "mode must be 'entity' or 'query'"
        self.split = split
        self.use_mm_start_end = use_mm_start_end

        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.root = os.path.join(base_image_dir, "oven")
        # self.image_root = os.path.join(self.root, "images")  # "/home/jeonghkim/projects/pixar/data/infoseek/images"
        self.image_root = os.path.join("/home/jeonghkim/projects/pixar/data/infoseek", "images")

        fname = f"oven_{'entity' if self.mode=='entity' else 'query'}_{split}.jsonl"
        split_path = os.path.join(self.root, fname)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"[OvenDataset] split file not found: {split_path}")

        self.rows = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        print(f"OVEN ({self.mode}, {split}): {len(self.rows)}")

    def __len__(self):
        return len(self.rows)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @staticmethod
    def _find_image(image_root: str, image_id: str):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = os.path.join(image_root, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
        candidates = sorted(glob.glob(os.path.join(image_root, f"{image_id}.*")))
        return candidates[0] if candidates else None

    def __getitem__(self, idx: int):
        item = self.rows[idx]

        image_id = item.get("image_id", "")
        question = item.get("question", "")

        if self.mode == "entity":
            gt = item.get("entity_text", "")
            sampled_answer_choices = [gt] if isinstance(gt, str) else [str(gt)]
        else:
            raw_ans = item.get("answer", [])
            if isinstance(raw_ans, str):
                sampled_answer_choices = [raw_ans]
            elif isinstance(raw_ans, list):
                sampled_answer_choices = raw_ans
            else:
                sampled_answer_choices = [str(raw_ans)]

        # load images
        image_path = self._find_image(self.image_root, image_id)
        if image_path is None:
            image_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            image_path = ""
        else:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                image_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                image_path = ""
            else:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        ori_size = image_rgb.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image_rgb, return_tensors="pt")["pixel_values"][0]

        image_resized = self.transform.apply_image(image_rgb)
        resize = image_resized.shape[:2]
        image = torch.from_numpy(image_resized).permute(2, 0, 1).contiguous().float()
        image = self.preprocess(image)

        conv = conversation_lib.default_conversation.copy()
        mm_use = conv.sep_style == conversation_lib.SeparatorStyle.TWO if self.use_mm_start_end else False
        source = preprocess_multimodal(
            question,
            mm_use_im_start_end=mm_use,
            is_textvqa=True
        )
        conv.messages = []
        role = conv.roles[0]
        instruction = 'Answer the question using a single word or phrase.'
        conv.append_message(role, source + ' ' + instruction)
        conversations = [conv.get_prompt()]

        questions = conversations

        # no masks for oven
        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
        per_token_label_dict = [{}]
        search_calls_norm = []
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_answer_choices,
            per_token_label_dict,
            search_calls_norm,
            inference,
            f"oven_{self.mode}",
        )
