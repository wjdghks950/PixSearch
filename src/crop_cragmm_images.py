import base64
import io
import json

import logging
import os
import sys
from functools import partial
from itertools import chain, islice
from pathlib import Path
from typing import List, Tuple, Optional, Union, Any

import coloredlogs

import cv2
import groundingdino.datasets.transforms as T
import jsonargparse as argparse
import numpy as np

import openai

import torch
import tqdm
import transformers

from evaluation.llm_as_judge_eval import evaluate_answer
from evaluation.rle_dict import get_mask_rle_dicts
from groundingdino.util.inference import load_model, predict
from PIL import Image

# Segment Anything & GroundingDINO imports
from segment_anything import sam_model_registry, SamPredictor
from torchvision.ops import box_convert
from transformers import BitsAndBytesConfig

# from utils.dataset import collate_fn
from utils.m4c_evaluator import TextVQAAccuracyEvaluator
from utils.vqa_dataset import CragMMDataset, GQADataset, POPEDataset, TextVQADataset

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def clamp(val, low, high):
    """
    Clamp a value between a low and high value.
    """
    return max(low, min(int(round(val)), high))


def _encode_image_base64(image_input) -> str:
    """
    Encode image to base64.
    Args:
        image_input: Can be either:
            - str: path to image file
            - PIL.Image: PIL Image object
            - np.ndarray: numpy array representing image
    """
    if isinstance(image_input, str):
        # Handle file path
        with open(image_input, "rb") as f:
            return base64.b64encode(f.read()).decode()
    elif isinstance(image_input, Image.Image):
        # Handle PIL Image
        buffer = io.BytesIO()
        # Convert to RGB if necessary
        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")
        image_input.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()
    elif isinstance(image_input, np.ndarray):
        # Handle numpy array
        # Assume it's in RGB format
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            # Convert numpy array to PIL Image then to base64
            pil_image = Image.fromarray(image_input.astype("uint8"))
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode()
        else:
            raise ValueError(f"Unsupported numpy array shape: {image_input.shape}")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def get_objects_from_llm(image_input, api_key: str, model: str = "gpt-4o", question=None) -> List[str]:
    """
    Ask GPT-4o to list visible objects as noun phrases and return them as a list.
    Args:
        image_input: Can be either:
            - str: path to image file
            - PIL.Image: PIL Image object
            - np.ndarray: numpy array representing image
        api_key: OpenAI API key
        model: Model name to use
        question: Used to condition the object extraction. If None, then the model will be asked to identify all objects in the image.
    """
    openai.api_key = api_key
    img_b64 = _encode_image_base64(image_input)
    if question is None:
        user_content = [
            {
                "type": "text",
                "text": (
                    "Identify all distinct, visible objects in noun-phrase form. "
                    'Return ***only*** a JSON object of the form {"objects": [<string>, ...]} without any extra text.'
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            },
        ]
    else:
        user_content = [
            {
                "type": "text",
                "text": (
                    f"Identify the most salient object of interest in generic noun-phrase form to the following question: {question}. "
                    'Return ***only*** a JSON object of the form {"object": <string>} without any extra text.'
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            },
        ]

    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=256,
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content.strip()
    print("LLM response: ", content)
    try:
        return json.loads(content)["object"]
    except (json.JSONDecodeError, KeyError):
        breakpoint()
        raise ValueError(f"LLM response not valid JSON: {content}")


# ---------------- GroundingDINO helpers ----------------
def load_grounding_dino(config_path: str, ckpt_path: str, device: str = "cuda"):
    model = load_model(config_path, ckpt_path)
    model.to(device).eval()
    return model


def load_image_for_groundingdino(image_input) -> Tuple[np.array, torch.Tensor]:
    if isinstance(image_input, str):
        image_source = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        if image_input.mode != "RGB":  # ensure 3-channel RGB mode
            image_source = image_input.convert("RGB")
        else:
            image_source = image_input
    else:
        breakpoint()
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


# ---------------- GroundingDINO helpers ----------------
def detect_boxes(
    model,
    image: torch.Tensor,
    orig_dim: Tuple[int, int],  # (H_orig, W_orig)
    phrases: List[str],
    box_threshold: float = 0.35,
    text_threshold: float = 0.35,
    device: str = "cuda",
):
    H_orig, W_orig = orig_dim
    boxes_xyxy, out_phrases = [], []

    for phrase in phrases:
        bboxes_cxcywh, _, _ = predict(
            model, image, phrase, box_threshold, text_threshold, device=device
        )
        if bboxes_cxcywh.size(0) == 0:
            continue

        # 1) convert cx-cy-w-h  →  x1-y1-x2-y2   (still normalised 0-1)
        bboxes_xyxy = box_convert(bboxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")

        # 2) clamp & scale to original-image pixels
        bboxes_xyxy.clamp_(0.0, 1.0)
        bboxes_xyxy[:, 0::2] *= W_orig  # x1, x2
        bboxes_xyxy[:, 1::2] *= H_orig  # y1, y2

        boxes_xyxy.extend(bboxes_xyxy.tolist())
        out_phrases.extend([phrase] * bboxes_xyxy.shape[0])

    return boxes_xyxy, out_phrases


def crop_image_regions_bbox(
    output_dir: str, img_rgb: np.ndarray, boxes: List[List[float]], phrases: List[str]
):
    # Crop the image regions within the bounding boxes and save them to the output directory
    H, W = img_rgb.shape[:2]

    # For each phrase, assign a integer count dict per phrase
    phrase_count = {}
    for phrase in phrases:
        if phrase not in phrase_count:
            phrase_count[phrase] = 0
        phrase_count[phrase] += 1

    for box, phrase in zip(boxes, phrases):
        x1, y1, x2, y2 = box
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid box: {box} for {phrase}")
        x1, y1, x2, y2 = (
            clamp(x1, 0, W - 1),
            clamp(y1, 0, H - 1),
            clamp(x2, 0, W + 1),
            clamp(y2, 0, H + 1),
        )
        cropped_img = img_rgb[y1:y2, x1:x2].copy()
        if phrase_count[phrase] > 0:
            out_fpath = f"cropped_{phrase}_{phrase_count[phrase]}.png"
            phrase_count[phrase] -= 1
        else:  # if phrase_count[phrase] == 0, then we have already saved all the images for this phrase
            continue
        cv2.imwrite(
            os.path.join(output_dir, "cropped_bboxes", out_fpath),
            cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR),
        )
        print(
            f"Saved cropped bbox image regions of {phrase} in {output_dir}/cropped_bboxes"
        )


# ---------------- SAM helpers ----------------
def load_sam(checkpoint: str, model_type: str = "vit_h", device: str = "cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return SamPredictor(sam)


def segment_masks(
    predictor: SamPredictor,
    image: np.ndarray,
    boxes: List[List[float]],
    device: str = "cuda",
) -> List[np.ndarray]:
    predictor.set_image(image)
    masks_outputs = []
    for box in boxes:
        np_box = np.array(box)
        masks, scores, logits = predictor.predict(box=np_box, multimask_output=False)
        masks = masks[0]
        masks_outputs.append(masks)

    return masks_outputs


def crop_image_regions_mask(
    output_dir: str,
    img_rgb: np.ndarray,
    masks: List[np.ndarray],
    phrases: List[str],
    alpha: float = 0.4,
    zoom_crop: bool = False,
):
    H, W = img_rgb.shape[:2]

    # For each phrase, assign a integer count dict per phrase
    phrase_count = {}
    for phrase in phrases:
        if phrase not in phrase_count:
            phrase_count[phrase] = 0
        phrase_count[phrase] += 1

    for mask, phrase in zip(masks, phrases):
        cropped_img = img_rgb.copy()
        cropped_img[mask == 0] = 0
        if zoom_crop:
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            x1, x2 = clamp(x1 - 1, 0, W), clamp(x2 + 1, 0, W + 1)
            y1, y2 = clamp(y1 - 1, 0, H), clamp(y2 + 1, 0, H + 1)
            cropped_img = cropped_img[y1:y2, x1:x2].copy()

        if phrase_count[phrase] > 0:
            out_fpath = (
                f"cropped_{phrase}_{phrase_count[phrase]}_zoom.png"
                if zoom_crop
                else f"cropped_{phrase}_{phrase_count[phrase]}.png"
            )
            phrase_count[phrase] -= 1
        else:  # if phrase_count[phrase] == 0, then we have already saved all the images for this phrase
            continue
        cv2.imwrite(
            os.path.join(output_dir, "cropped_masks", out_fpath),
            cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR),
        )
        print(
            f"Saved cropped mask image region of {phrase} in {output_dir}/cropped_masks"
        )


# ---------------- Utility ----------------
def overlay_masks(
    image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5
) -> np.ndarray:
    output = image.copy()
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 128, 0),
        (128, 0, 255),
        (128, 255, 0),
        (0, 128, 255),
    ]
    for idx, mask in enumerate(masks):
        color = palette[idx % len(palette)]
        colored = np.stack([mask * c for c in color], axis=-1).astype(np.uint8)
        output = np.where(
            mask[..., None],
            cv2.addWeighted(output, 1 - alpha, colored, alpha, 0),
            output,
        )
    return output


def overlay_bboxes(
    image: np.ndarray, boxes: List[List[float]], phrases: List[str]
) -> np.ndarray:
    output = image.copy()
    for box, phrase in zip(boxes, phrases):
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(
            output,
            phrase,
            (int(x1), int(y1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            3,
        )
    return output


def generate_gold_mask(
    args,
    images: List[Union[str, np.ndarray, Image.Image]],
    question: str,
    query_texts: List[str] = None,
    out_dir: str = "",
):
    """
    Generate a gold mask for the most salient object(s) for the given image and question
        Args:
            - args: command line arguments
            - images: image(s) to be processed
            - question: question from a VQA instance

    """
    assert isinstance(images, list), "images must be a list"

    # Load GroundingDINO and SAM models
    dino = load_grounding_dino(args.dino_config, args.dino_checkpoint, device=args.device)
    predictor = load_sam(args.sam_checkpoint, args.sam_model_type, device=args.device)

    rle_dicts_list = []
    phrases_list = []
    paths_list = []
    for img_idx, image in enumerate(images):
        if query_texts is None:
            try:
                objects = get_objects_from_llm(
                    img, args.openai_api_key, model=args.llm_model, question=question,
                )
            except Exception as e:
                print(f"Error getting objects from LLM for {img}: {e}")
                breakpoint()
                objects = []
        else:
            objects = query_texts

        if not objects:
            print(f"No objects detected for {images}")
            return [], ""

        # GroundingDINO box detection
        try:
            img_rgb, img_transformed = load_image_for_groundingdino(image)
        except Exception as e:
            print(f"Error loading image for GroundingDINO: {e}")
            breakpoint()

        boxes, phrases = detect_boxes(
            dino,
            img_transformed,
            orig_dim=(img_rgb.shape[0], img_rgb.shape[1]),
            phrases=objects,
            device=args.device,
        )
        if not boxes:  # if there are no boxes, return to get_objects_from_llm and get objects list
            objects = get_objects_from_llm(image, args.openai_api_key, model=args.llm_model, question=question)
            img_rgb, img_transformed = load_image_for_groundingdino(image)
            boxes, phrases = detect_boxes(dino, img_transformed, orig_dim=(img_rgb.shape[0], img_rgb.shape[1]), phrases=objects, device=args.device)
            if not boxes:
                print(f"* No boxes detected for {image}")
                return [], [], []

        assert len(boxes) == len(phrases), f"Boxes and phrases have different lengths: {len(boxes)} != {len(phrases)}"

        # save bboxes
        Path(os.path.join(out_dir, "cropped_bboxes")).mkdir(parents=True, exist_ok=True)
        overlay = overlay_bboxes(img_rgb, boxes, phrases)
        cv2.imwrite(
            os.path.join(out_dir, "cropped_bboxes", f"bbox_overlay_{img_idx}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        print(
            f"Saved {len(boxes)} bounding boxes + overlay in {out_dir}/cropped_bboxes/bbox_overlay_{img_idx}.png"
        )
        # crop the image regions within the bounding boxes and save them to the output directory
        crop_image_regions_bbox(
            out_dir, img_rgb, boxes, phrases
        )

        # need to return mask rle_dicts and phrases
        masks = segment_masks(predictor, img_rgb, boxes, device=args.device)
        try:
            assert len(masks) == len(phrases)
        except AssertionError:
            print(
                f"Masks and phrases have different lengths: {len(masks)} != {len(phrases)}"
            )
            breakpoint()

        Path(os.path.join(out_dir, "cropped_masks")).mkdir(parents=True, exist_ok=True)
        overlay = overlay_masks(img_rgb, masks, alpha=args.mask_alpha)
        cv2.imwrite(
            os.path.join(out_dir, "cropped_masks", f"mask_overlay_{img_idx}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        f"Saved {len(masks)} masks + overlay in {out_dir}/cropped_masks/mask_overlay_{img_idx}.png"
        crop_image_regions_mask(
            out_dir, img_rgb, masks, phrases, zoom_crop=True,
        )
        if len(boxes) > 0 and len(masks) > 0:
            paths = {
                'bbox': os.path.join(out_dir, "cropped_bboxes", f"bbox_overlay_{img_idx}.png"),
                'mask': os.path.join(out_dir, "cropped_masks", f"mask_overlay_{img_idx}.png"),
            }
        else:
            breakpoint()

        rle_dicts_list.extend(get_mask_rle_dicts(np.stack(masks), phrases))
        phrases_list.extend(phrases)
        paths_list.append(paths)

    return rle_dicts_list, paths_list, phrases_list


def parse_args(args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser(
        description="Object Extraction + SAM masking pipeline with LLM extracting noun phrases"
    )
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--workers", default=0, type=int, help="number of workers")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument(
        "--backbone",
        default="liuhaotian/llava-llama-2-13b-chat-lightning-preview",
        type=str,
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision_tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--val_dataset", default="cragmm|validation", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--limit_batches", default=0, type=int)
    parser.add_argument("--dataset_dir", default="../data", type=str)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)

    parser.add_argument("--llm_model", default="gpt-4o", type=str)
    parser.add_argument("--use_llm_as_judge", action="store_true")

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="../results/",
    )

    parser.add_argument("--exp_name", default="llm-seg", type=str)

    parser.add_argument(
        "--image", required=False, help="Input image path (for single image processing)"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="../../results/cragmm/llmseg",
    )
    parser.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--sam_checkpoint",
        default="~/weights/sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--sam_model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"]
    )
    parser.add_argument(
        "--dino_config",
        default="~/weights/groundingdino/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--dino_checkpoint",
        default="~/weights/groundingdino/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--mask_alpha", default=0.4, type=float)

    args = parser.parse_string(config_str) if config_str else parser.parse_args(args)
    return args, parser


def main(args: list[str] = None, config_str: str = None):
    args, parser = parse_args(args, config_str)

    if not args.openai_api_key:
        raise RuntimeError(
            "OpenAI API key is required. Please set the OPENAI_API_KEY environment variable."
        )

    # Check if we're doing single image processing or dataset validation
    if args.image and os.path.exists(args.image):
        # Single image processing mode
        process_single_image(args)
    else:
        # Dataset validation mode
        process_dataset(args, parser)


def process_single_image(args):

    # TODO: Needs to be updated - Refer to process_dataset for the up-to-date implementation

    """Process a single image with LLM + GroundingDINO + SAM pipeline"""
    # output directory creation
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. LLM object extraction
    objects = get_objects_from_llm(args.image, args.openai_api_key)

    with open(os.path.join(args.output_dir, "objects.json"), "w") as f:
        json.dump({"objects": objects}, f, indent=2)
    print(f"LLM detected {len(objects)} objects: {objects}")

    # 2. GroundingDINO box detection
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    dino = load_grounding_dino(
        args.dino_config, args.dino_checkpoint, device=args.device
    )
    boxes, phrases = detect_boxes(
        dino,
        img_rgb,
        orig_dim=(img_rgb.shape[0], img_rgb.shape[1]),
        phrases=objects,
        device=args.device,
    )
    if not boxes:
        print("No boxes found - exiting.")
        return

    # 3. SAM mask refinement
    predictor = load_sam(args.sam_checkpoint, args.sam_model_type, device=args.device)
    masks = segment_masks(predictor, img_rgb, boxes, device=args.device)

    # 4. Save masks
    for i, (phrase, mask) in enumerate(zip(phrases, masks)):
        out_path = os.path.join(
            args.output_dir,
            f"mask_{i:02d}_{phrase.replace(' ', '_')}.png",
        )
        cv2.imwrite(out_path, (mask[0] * 255).astype(np.uint8))

    # 5. Save overlay
    overlay = overlay_bboxes(img_rgb, [m[0] for m in masks])
    cv2.imwrite(
        os.path.join(args.output_dir, "overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    print(f"Saved {len(masks)} masks + overlay in {args.output_dir}")


def process_dataset(args, parser):
    """Process dataset with integrated LLM segmentation"""
    # build dataset for validation
    if "textvqa" in args.val_dataset:
        val_dataset = TextVQADataset(
            base_image_dir=args.dataset_dir,
            tokenizer=None,
            vision_tower=args.vision_tower,
        )
    elif "gqa" in args.val_dataset:
        val_dataset = GQADataset(
            base_image_dir=args.dataset_dir,
            tokenizer=None,
            vision_tower=args.vision_tower,
        )
    elif "pope" in args.val_dataset:
        val_dataset = POPEDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=None,
            vision_tower=args.vision_tower,
        )
    elif "cragmm" in args.val_dataset:
        val_dataset = CragMMDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=None,
            vision_tower=args.vision_tower,
        )
    else:
        raise ValueError("No validation dataset provided. Exiting.")

    # Load GroundingDINO and SAM models
    dino = load_grounding_dino(
        args.dino_config, args.dino_checkpoint, device=args.device
    )
    predictor = load_sam(args.sam_checkpoint, args.sam_model_type, device=args.device)

    # Create output directory
    Path(args.predictions_path).mkdir(parents=True, exist_ok=True)
    pred_file_path = os.path.join(
        args.predictions_path,
        f'predictions_{args.val_dataset.split("|")[0]}_llmseg_{args.llm_model}.jsonl',
    )

    if args.limit_batches > 0:
        n_samples = min(args.limit_batches, len(val_dataset))
    else:
        n_samples = len(val_dataset)

    pbar = tqdm.tqdm(range(n_samples), desc="Processing dataset with LLM segmentation")

    for idx in pbar:
        data_item = val_dataset[idx]
        session_id = val_dataset.dataset[idx]["session_id"]

        if isinstance(val_dataset, CragMMDataset):
            image_path = data_item[0]  # image_path
            dataset_item = val_dataset.dataset[idx]
            if isinstance(dataset_item["image"], Image.Image):
                image_input = dataset_item["image"]
            else:
                # Fallback to loading from path if available
                image_input = image_path
        else:
            # For other datasets, use the image path
            image_path = data_item[0]
            image_input = image_path

        # LLM object extraction
        print("image_input: ", image_input)
        try:
            objects = get_objects_from_llm(
                image_input, args.openai_api_key, model=args.llm_model
            )
        except Exception as e:
            print(f"Error getting objects from LLM for {image_path}: {e}")
            objects = []

        if not objects:
            print(f"No objects detected for {image_path}")
            continue

        # Load image for GroundingDINO processing
        if isinstance(image_input, str):
            print("image_input is a string")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                print(f"Could not load image: {image_input}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            print("image_input is a PIL Image")
            img_rgb = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        else:
            print("image_input is a numpy array")
            print(f"Unsupported image type for {image_path}")
            continue

        # GroundingDINO box detection
        img_rgb, img_transformed = load_image_for_groundingdino(image_input)
        boxes, phrases = detect_boxes(
            dino,
            img_transformed,
            orig_dim=(img_rgb.shape[0], img_rgb.shape[1]),
            phrases=objects,
            device=args.device,
        )
        if not boxes:
            print(f"No boxes found for {image_path}")
            continue
        try:
            assert len(boxes) == len(phrases)
        except AssertionError:
            print(
                f"Boxes and phrases have different lengths: {len(boxes)} != {len(phrases)}"
            )
            breakpoint()

        # Overlay boxes on image and save image to output directory
        Path(os.path.join(args.output_dir, session_id)).mkdir(
            parents=True, exist_ok=True
        )

        overlay = overlay_bboxes(img_rgb, boxes, phrases)
        cv2.imwrite(
            os.path.join(args.output_dir, session_id, "bbox_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        print(
            f"Saved {len(boxes)} bounding boxes + overlay in {args.output_dir}/{session_id}"
        )

        # Crop the image regions within the bounding boxes and save them to the output directory
        Path(os.path.join(args.output_dir, session_id, "cropped_bboxes")).mkdir(
            parents=True, exist_ok=True
        )
        crop_image_regions_bbox(
            os.path.join(args.output_dir, session_id), img_rgb, boxes, phrases
        )

        # 4. SAM mask refinement
        masks = segment_masks(predictor, img_rgb, boxes, device=args.device)

        try:
            assert len(masks) == len(phrases)
        except AssertionError:
            print(
                f"Masks and phrases have different lengths: {len(masks)} != {len(phrases)}"
            )
            breakpoint()

        # Overlay masks on image and save image to output directory
        overlay = overlay_masks(img_rgb, masks, alpha=args.mask_alpha)
        cv2.imwrite(
            os.path.join(args.output_dir, session_id, "mask_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        print(f"Saved {len(masks)} masks + overlay in {args.output_dir}/{session_id}")

        # Crop the image regions within the masks and save them to the output directory
        Path(os.path.join(args.output_dir, session_id, "cropped_masks")).mkdir(
            parents=True, exist_ok=True
        )
        crop_image_regions_mask(
            os.path.join(args.output_dir, session_id),
            img_rgb,
            masks,
            phrases,
            zoom_crop=True,
        )

        # 5. Save results
        output_dict = {
            "image_path": image_path,
            "detected_objects": objects,
            "boxes": boxes,
            "masks": get_mask_rle_dicts(np.stack(masks), phrases),
            "phrases": phrases,
            "num_masks": len(masks),
        }

        with open(pred_file_path, "a") as f:
            line = json.dumps(output_dict, ensure_ascii=False)
            line += "\n" if idx < n_samples - 1 else ""
            f.write(line)

        print(
            f"Processed {image_path}: {len(objects)} objects -> {len(boxes)} boxes -> {len(masks)} masks"
        )

    print(f"Processing complete. Results saved to {pred_file_path}")


def to_plum_inputs(
    args, batch: dict, tokenizer: transformers.AutoTokenizer, set_mc_choices=False
):
    input_dict = {
        "images": batch["images"].to(args.device),
        "images_clip": batch["images_clip"].to(args.device),
        "input_ids": batch["input_ids"].to(args.device),
        "resize_list": batch["resize_list"],
        "label_list": batch["label_list"],
        "attention_masks": batch["attention_masks"].to(args.device),
        "max_new_tokens": args.model_max_length,
        "tokenizer": tokenizer,
        # 'prompt_user_input': True
    }
    return input_dict


if __name__ == "__main__":
    coloredlogs.install(level="DEBUG")

    main(sys.argv[1:])
