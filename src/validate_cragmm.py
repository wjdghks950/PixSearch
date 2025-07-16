import os
import re
import sys
import json
import httpx
import torch
import transformers
import pprint
import logging
import openai
import coloredlogs
import jsonargparse as argparse

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from functools import partial
from collections import defaultdict
from itertools import chain, islice
from typing import List, Tuple, Optional
from transformers import AutoConfig, AutoProcessor, Llama4ForConditionalGeneration

from llama_api_client import LlamaAPIClient, APIStatusError, InternalServerError, BadRequestError

from evaluation.llm_as_judge_eval import evaluate_answer
from evaluation.rle_dict import get_mask_rle_dicts
from image_recall_calculate import load_cached_results

from utils.dataset import collate_fn_llama4
from utils.vqa_dataset import CragMMDataset
from utils.utils import image_to_base64, image_to_pil, choose_gold_object, pil_to_base64, resize_keep_aspect, sanitize_pil, TIMEOUT_SECONDS

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args(args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser(description="Validate LLama-4 on CRAG-MM datasets")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--workers", default=0, type=int, help="number of workers")
    parser.add_argument(
        "--version",
        default="/home/jeonghkim/weights/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model", default="Llama-4-Scout-17B-16E-Instruct", type=str)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--val_dataset", default="cragmm", type=str)
    parser.add_argument("--split", default="validation", type=str)
    parser.add_argument(
        "--turn_type", default="single", type=str
    )  # 'single' or 'multi' in cragmm
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--limit_batches", default=0, type=int)
    parser.add_argument(
        "--cragmm_master_data_path",
        default="cragmm_qa_data_for_benchmark_eval.json",
        type=str,
    )
    parser.add_argument("--resize_img", action="store_true", default=False)
    parser.add_argument("--use_crop", action="store_true", default=False)
    parser.add_argument("--crop_type", default="bbox", type=str, help="bbox or mask")
    parser.add_argument("--use_gold_crop", action="store_true", default=False)
    parser.add_argument("--topk", default=5, type=int, help="k for top-k retrieval")
    parser.add_argument("--max_gold_objects", default=30, type=int, help="Max number of gold objects to use during retrieval")
    parser.add_argument("--use_retrieved", action="store_true", default=False)
    parser.add_argument("--use_summarized", action="store_true", default=False)
    parser.add_argument("--use_reranked", action="store_true", default=False)

    parser.add_argument("--image_only", action="store_true", default=False)
    parser.add_argument("--text_only", action="store_true", default=False)

    parser.add_argument(
        "--dataset_dir",
        default="/home/jeonghkim/projects/region-mm-retrieve/data",
        type=str,
    )
    parser.add_argument(
        "--cached_results_dir",
        type=str,
        default="/home/jeonghkim/projects/region-mm-retrieve/results/cragmm",
        help="Directory to load cached results from.",
    )
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
    parser.add_argument("--use_llama_api", action="store_true", default=False, help="Use llama api instead of llama4")

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="/home/jeonghkim/projects/region-mm-retrieve/results/cragmm",
    )

    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--exp_name", default="plum-13b-vqa-eval", type=str)

    args = parser.parse_string(config_str) if config_str else parser.parse_args(args)
    return args, parser


def construct_session_id_to_cragmm_dataset_mapping(
    dataset, image_cache_dir, cache_path=None
):
    """
    Construct the session_id to cragmm dataset mapping
    Args:
        - dataset: the cragmm dataset (validation, public_test, private_test)
    Returns:

    """
    Path(image_cache_dir).mkdir(parents=True, exist_ok=True)

    cragmm_dataset = {}
    if cache_path is None or not os.path.exists(cache_path):
        # Create a dictionary with session_id as key and rest as values
        for example in tqdm(dataset, desc="Constructing session_id to cragmm dataset"):
            sid = example["session_id"]
            if example["image"] is not None:
                image_cache_fpath = os.path.join(image_cache_dir, f"{sid}.jpg")
                example["image"].save(image_cache_fpath)
                image_cache_fpath = str(os.path.basename(image_cache_fpath))
            else:
                image_cache_fpath = ""

            cragmm_dataset[sid] = {
                "image": image_cache_fpath,
                "image_url": example["image_url"],
                "turns": example["turns"],
                "answers": example["answers"],
            }
        with open(cache_path, "w") as f:
            json.dump(cragmm_dataset, f)
    else:
        with open(cache_path, "r") as f:
            cragmm_dataset = json.load(f)
        # Convert cached image paths to PIL images
        for sid, example in tqdm(
            cragmm_dataset.items(), desc=f"Loading cached images from {cache_path}"
        ):
            image_cache_fpath = os.path.join(image_cache_dir, f"{sid}.jpg")
            if os.path.exists(image_cache_fpath):
                example["image"] = Image.open(image_cache_fpath)

    return cragmm_dataset


def main(args: list[str] = None, config_str: str = None):
    args, parser = parse_args(args, config_str)

    processor = AutoProcessor.from_pretrained(args.version)
    args.model = os.path.basename(args.version)

    if not args.debug and not args.use_llama_api:
        model = Llama4ForConditionalGeneration.from_pretrained(
            args.version,
            attn_implementation="sdpa",  # # NOTE: attn_implementation="flex_attention" causes error
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    elif args.use_llama_api:
        assert os.environ.get("LLAMA_API_KEY", None) is not None, "LLAMA_API_KEY is not set"
        model = None
        if "Llama-4-Scout-17B-16E-Instruct" in args.version:
            args.model = "Llama-4-Scout-17B-16E-Instruct-FP8"
        elif "Llama-4-Maverick-17B-128E-Instruct" in args.version:
            args.model = "Llama-4-Maverick-17B-128E-Instruct-FP8"

        model = LlamaAPIClient()
        response = model.chat.completions.create(
            model=args.model,
            messages=[  # Test message with images
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do these two images have in common?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"https://upload.wikimedia.org/wikipedia/commons/2/2e/Lama_glama_Laguna_Colorada_2.jpg",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"https://upload.wikimedia.org/wikipedia/commons/1/12/Llamas%2C_Laguna_Milluni_y_Nevado_Huayna_Potos%C3%AD_%28La_Paz_-_Bolivia%29.jpg",
                            },
                        },
                    ],
                },
            ],
        )
        print("(Test) << Llama4 Response >>\n", response.completion_message.content.text)
    else:
        model = None

    # build dataset for validation
    if "cragmm" in args.val_dataset:
        val_dataset = CragMMDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=processor.tokenizer,
            vision_tower=args.vision_tower,
            split=args.split,
            return_raw=True,
        )

    # map session_id to cragmm data instances
    cragmm_dataset_dict = construct_session_id_to_cragmm_dataset_mapping(
        val_dataset.crag_mm_data,
        image_cache_dir=os.path.join(args.dataset_dir, "cragmm_image_cache"),
        cache_path=os.path.join(args.dataset_dir, "cragmm_sid_to_data.json"),
    )

    # build DataLoader
    if val_dataset is not None:
        assert (
            args.val_batch_size == 1
        ), "Currently supports batch_size=1 for segmentation eval"
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            collate_fn=partial(
                collate_fn_llama4,
                processor=processor,
                local_rank=args.local_rank,
                turn_type=args.turn_type,
            ),
        )
    else:
        raise ValueError("No validation dataset provided. Exiting.")

    validate_cragmm(val_loader, model, processor, args, cragmm_dataset_dict)


@torch.no_grad()
def validate_cragmm(val_loader, model, processor, args, cragmm_dataset_dict):
    """
    Validation script for CRAG-MM
    """
    if not args.debug and not args.use_llama_api:
        model.eval()

    if args.use_gold_crop and args.use_crop:  # Need LLM 'client' to extract question + image-conditioned gold objects
        if "gpt" in args.llm_model:
            client = openai.Client(timeout=TIMEOUT_SECONDS)

        elif "Llama-4-Scout-17B-16E-Instruct" in args.version:
            args.llm_model = "Llama-4-Scout-17B-16E-Instruct-FP8"
            assert os.environ.get("LLAMA_API_KEY", None) is not None, "LLAMA_API_KEY is not set"
            client = LlamaAPIClient()

    skipped_instance_num = 0
    if args.limit_batches > 0:
        n_batches = min(args.limit_batches, len(val_loader))
    else:
        n_batches = len(val_loader)

    Path(args.predictions_path).mkdir(parents=True, exist_ok=True)
    preds_fpath = f"predictions_{args.val_dataset}_{args.model.lower()}_{args.llm_model}"
    if args.use_crop:
        preds_fpath += f"_crop_{args.crop_type}"
        if args.use_gold_crop:
            preds_fpath += "_gold"
    if args.use_retrieved:
        if args.image_only and not args.text_only:
            preds_fpath += "_image_only"
        elif args.text_only and not args.image_only:
            preds_fpath += "_text_only"
        elif args.image_only and args.text_only:
            preds_fpath += "_image_text"
        else:
            raise ValueError("Either image_only or text_only, or both must be True")
        preds_fpath += f"_retrieved_k_{args.topk}"
    if args.use_summarized:
        preds_fpath += "_summarized"
    if args.use_reranked:
        preds_fpath += "_reranked"

    preds_fpath += ".jsonl"

    pred_file_path = os.path.join(args.predictions_path, preds_fpath)

    # Load the cached retrieval results
    cached_jsonl_list = []
    if os.path.exists(pred_file_path):
        with open(pred_file_path, "r") as f:
            for line in f:
                if line.strip():
                    cached_jsonl_list.append(json.loads(line))
        print(f"Loaded {len(cached_jsonl_list)} cached results from {pred_file_path}")

    # Load the cached retrieval results
    if args.use_crop:
        if args.crop_type == "bbox":
            bbox_cached_results = load_cached_results(
                os.path.join(
                    args.cached_results_dir,
                    "bbox_image_retrieval",
                    "bbox_image_retrieval_cache.json",
                )
            )
        elif args.crop_type == "mask":
            mask_cached_results = load_cached_results(
                os.path.join(
                    args.cached_results_dir,
                    "mask_image_retrieval",
                    "mask_image_retrieval_cache.json",
                )
            )
    elif not args.use_crop and (args.use_retrieved and args.topk > 0):
        whole_image_cached_results = load_cached_results(
            os.path.join(
                args.cached_results_dir,
                "whole_image_retrieval",
                "whole_image_retrieval_cache.json",
            )
        )

    val_iter = iter(val_loader)
    pbar = tqdm(range(n_batches), desc=f"Validating {args.model} with CRAG-MM")

    total_accuracy = 0
    output_dicts = []
    for idx in pbar:
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        # Check for cached results
        if idx < len(cached_jsonl_list):
            print(f"batch[sid]: {batch['session_ids'][0]} | cached_jsonl_list[sid]: {cached_jsonl_list[idx]['session_id']}")
            continue

        messages = [  # This is a single batch; (len(messages) means number of turns in a single batch)
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant that answers the given question. You will be provided images and a question that asks about the image. Optionally, you will also be provided with retrieved images or texts that help you understand the image and answer the provided question. Make sure to answer the question based on the images and respond in a concise manner."
                    }
                ]
            },
            {"role": "user", "content": []},
        ]
        if args.turn_type == "single":
            # Load the image and question and format according to llama 4 chat style
            session_id = batch["session_ids"][0]
            question = batch["turns_list"][0]["query"][0]
            image = batch["images"][0] if batch["images"][0] is not None else batch["image_urls"][0]

            messages[-1]["content"].append({"type": "text", "text": "Here is the main image to the given question: "})

            if args.use_llama_api:
                messages[-1]["content"].append({"type": "image_url", "image_url": {"url": image_to_base64(image, with_data_uri=True)}})
            else:
                image = image_to_pil(image) if isinstance(image, str) else image
                if image == -1:
                    print("Image not found")
                    raise ValueError(f"Image not found. image: {image}")
                image = sanitize_pil(image)
                messages[-1]["content"].append({"type": "image", "image": image})

            if args.use_crop:
                cropped_dir_path = "cropped_masks" if args.crop_type == "mask" else "cropped_bboxes"
                if args.crop_type == "bbox":
                    if session_id not in bbox_cached_results:
                        print(f"Session id {session_id} not found in bbox_cached_results. Skipping.")
                        skipped_instance_num += 1
                        output_dict = {
                            "session_id": session_id,
                            "question": question,
                            "output_text": "",
                            "ground_truth_text": batch["answers_list"][0]["ans_full"][0],
                            "explanation": "",
                            "accuracy": None,
                        }
                        output_dicts.append(output_dict)
                        with open(pred_file_path, "a") as f:
                            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
                        continue
                    topk_retrieved_dict = bbox_cached_results[session_id]
                    pattern = re.compile(r'^cropped_(.*?_\d+)\.png$')  # e.g., '../../results/cragmm/fd93d7e6-9cdf-4334-a5be-c2ad13880d0b/cropped_masks/cropped_blue car_1.png'
                    cropped_region_paths = os.listdir(os.path.join(args.cached_results_dir, "fixed", session_id, cropped_dir_path))

                elif args.crop_type == "mask":
                    if session_id not in mask_cached_results:
                        print(f"Session id {session_id} not found in mask_cached_results. Skipping.")
                        skipped_instance_num += 1
                        output_dict = {
                            "session_id": session_id,
                            "question": question,
                            "output_text": "",
                            "ground_truth_text": batch["answers_list"][0]["ans_full"][0],
                            "explanation": "",
                            "accuracy": None,
                        }
                        output_dicts.append(output_dict)
                        with open(pred_file_path, "a") as f:
                            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
                        continue
                    topk_retrieved_dict = mask_cached_results[session_id]
                    pattern = re.compile(r'^cropped_(.*?_\d+)_zoom\.png$')  # e.g., '../../results/cragmm/fd93d7e6-9cdf-4334-a5be-c2ad13880d0b/cropped_masks/cropped_blue car_1_zoom.png'
                    cropped_region_paths = os.listdir(os.path.join(args.cached_results_dir, "fixed", session_id, cropped_dir_path))

                # construct 'obj_lbl' to list of region image paths - could be multiple 'region_path's for a single 'obj_lbl'
                object_to_region_path = {}
                for region_path in cropped_region_paths:
                    region_path = os.path.join(args.cached_results_dir, "fixed", session_id, cropped_dir_path, region_path)
                    obj_lbl = pattern.match(os.path.basename(region_path)).group(1)
                    if obj_lbl not in object_to_region_path:
                        object_to_region_path[obj_lbl] = [region_path]  # e.g., ['blue car_1': [path1, ...], 'blue car_2': [path2, ...]
                    else:
                        object_to_region_path[obj_lbl].append(region_path)

                identified_objects = list(object_to_region_path.keys())

                image_crops = []
                if args.use_gold_crop:  # (1) only the gold object images
                    gold_obj_lbls, llm_output_text = choose_gold_object(args, client, question, image, identified_objects)  # choose gold object using LLM (e.g., gpt-4o)
                    if gold_obj_lbls is None:
                        breakpoint()
                        raise ValueError("No gold object found. Exiting.")
                    elif args.max_gold_objects > 0 and len(gold_obj_lbls) > args.max_gold_objects:
                        gold_obj_lbls = gold_obj_lbls[:args.max_gold_objects]
                # else (2) all the cropped objects images

                if args.use_retrieved:  # using retrieved image / text snippets
                    topk_retrieved_list = defaultdict(list)
                    # store the retrieved results under the corresponding object label
                    for region_path, retrieved_list in topk_retrieved_dict.items():
                        obj_lbl = pattern.match(os.path.basename(region_path)).group(1)
                        if obj_lbl not in topk_retrieved_list:
                            topk_retrieved_list[obj_lbl].extend(retrieved_list)

                cropped_imgs = []
                cropped_topk_retrieved_list = []
                # get the cropped images and their retrieved items (if args.use_retrieved) to the messages
                for obj_lbl, region_paths in object_to_region_path.items():
                    if args.use_gold_crop:
                        if obj_lbl not in gold_obj_lbls:  # skip the non-gold crop objects
                            continue
                    region_path = region_paths[0]
                    image = image_to_pil(region_path)
                    if image == -1:
                        print(f"Image not found in: {region_path}")
                        continue
                    else:
                        pil_image = sanitize_pil(image)
                        if args.resize_img:
                            pil_image = resize_keep_aspect(pil_image)
                        cropped_imgs.append(pil_image)
                    if args.use_retrieved:
                        cropped_topk_retrieved_list.append(topk_retrieved_list[obj_lbl])

                # add cropped images to the message
                if not args.use_retrieved:
                    messages[-1]["content"].append({"type": "text", "text": "Here are the cropped images from the whole image that you can also use to help answer the question: "})
                    if args.use_llama_api:
                        messages[-1]["content"].extend([{"type": "image_url", "image_url": {"url": pil_to_base64(pil_image)}} for pil_image in cropped_imgs])
                    else:
                        messages[-1]["content"].extend([{"type": "image", "image": pil_image} for pil_image in cropped_imgs])
            else:
                topk_retrieved_list = whole_image_cached_results[session_id]

            if args.use_retrieved and args.topk > 0:
                # Choose image or text
                if args.use_crop: # Retrieved items for the cropped image (interleaved with cropped image (e.g., cropped_img1 + retrieved_item1...K + cropped_img2 + retrieved_item2...K + ...))

                    assert len(cropped_imgs) == len(cropped_topk_retrieved_list), f"len(cropped_imgs): {len(cropped_imgs)} | len(cropped_topk_retrieved_list): {len(cropped_topk_retrieved_list)}"

                    messages[-1]["content"].append({"type": "text", "text": f"Here are the cropped images and their top-{args.topk} retrieved images/texts that you can also use to help answer the question: "})
                    for cropped_img, topk_retrieved in zip(cropped_imgs, cropped_topk_retrieved_list):
                        retrieved_images = []
                        retrieved_texts = []
                        for retrieved in topk_retrieved[:args.topk]:
                            if args.image_only:
                                image = image_to_pil(retrieved["url"])
                                if image == -1:
                                    print("Image not found")
                                    continue
                                else:
                                    pil_image = sanitize_pil(image)
                                    if args.resize_img:  # NOTE: resize image to not exceed input_seq length - llama4 gives error otherwise
                                        pil_image = resize_keep_aspect(pil_image)
                                    retrieved_images.append(pil_image)
                            if args.text_only:
                                retrieved_texts.append(retrieved["entities"][0]["entity_name"])  # NOTE: Not only entities, but there can also be paragraphs/docs

                        # append cropped images
                        cropped_img_retrieved_msg = []
                        if args.use_llama_api:
                            cropped_img_msg = [{"type": "image_url", "image_url": {"url": pil_to_base64(cropped_img)}}]
                            if args.image_only:  # append retrieved images of the cropped image
                                cropped_img_retrieved_msg += [{"type": "image_url", "image_url": {"url": pil_to_base64(pil_image)}} for pil_image in retrieved_images]
                        else:
                            cropped_img_msg = [{"type": "image", "image": cropped_img}]
                            if args.image_only:  # append retrieved images of the cropped image
                                cropped_img_retrieved_msg += [{"type": "image", "image": pil_image} for pil_image in retrieved_images]

                        if args.text_only:  # append retrieved texts of the cropped image
                            cropped_img_retrieved_msg += [{"type": "text", "text": txt} for txt in retrieved_texts]
                        messages[-1]["content"].extend(cropped_img_msg)

                        # append retrieved items (e.g., text or image) of cropped images
                        messages[-1]["content"].extend(cropped_img_retrieved_msg)

                else:  # Retrieved items for whole image
                    retrieved_images = []
                    retrieved_texts = []
                    for retrieved in topk_retrieved_list[:args.topk]:
                        if args.image_only:
                            image = image_to_pil(retrieved["url"])
                            if image == -1:
                                print("Image not found")
                                continue
                            else:
                                pil_image = sanitize_pil(image)
                                if args.resize_img:  # NOTE: resize image to not exceed input_seq length - llama4 gives error otherwise
                                    pil_image = resize_keep_aspect(pil_image)
                                retrieved_images.append(pil_image)
                        if args.text_only:
                            retrieved_texts.append(retrieved["entities"][0]["entity_name"])  # NOTE: Not only entities, but there can also be paragraphs/docs
                    whole_img_retrieved_msg = []
                    if args.use_llama_api:
                        if args.image_only:
                            whole_img_retrieved_msg += [{"type": "image_url", "image_url": {"url": pil_to_base64(pil_image)}} for pil_image in retrieved_images]
                    else:
                        if args.image_only:
                            whole_img_retrieved_msg += [{"type": "image", "image": pil_image} for pil_image in retrieved_images]

                    if args.text_only:
                        whole_img_retrieved_msg += [{"type": "text", "text": txt} for txt in retrieved_texts]

                    retrieved_prompt = f"And here are the top {args.topk} retrieved images/texts from the first image that you can use as a reference: "
                    messages[-1]["content"].append({"type": "text", "text": retrieved_prompt})
                    messages[-1]["content"].extend(whole_img_retrieved_msg)

            # TODO: Implement the summarized, reranked retrieval settings

            elif args.use_summarized and args.topk > 0:
                pass
            elif args.use_reranked and args.topk > 0:
                pass  # TODO: For the reranker, what reranker to use? BGE? CLIP? Sentence Transformer?

            messages[-1]["content"].append({"type": "text", "text": question})


        elif args.turn_type == "multi":
            pass

        if args.use_llama_api:
            try:
                response = model.chat.completions.create(
                    model=args.model,
                    messages=messages,
                )
                output_text = response.completion_message.content.text

            except InternalServerError as err:  # 500 error handle
                print("\n── Internal-Server-Error ──────────────────────────")
                print("status :", err.response.status_code)
                print("headers:", err.response.headers)
                # body may be JSON or plain text; try both
                try:
                    body = err.response.json()
                except json.JSONDecodeError:
                    body = err.response.text
                skipped_instance_num += 1
                output_text = ""

            except BadRequestError as err: # Add 400 error handling
                resp = err.response          # httpx.Response
                body = resp.text
                # Example: malformed image URL → just skip
                if "No 'uri' scheme" in body:
                    print("!! bad image URL - skipping")
                    return None

                # Otherwise we treat it as transient and retry
                print("400 Bad Request")
                skipped_instance_num += 1
                output_text = ""

            # ───── generic API or transport erroårs ────────────────────
            except APIStatusError as err:     # 4xx / 5xx that aren't mapped
                print("HTTP %s - %s", err.response.status_code, err.response.text)
                skipped_instance_num += 1
                output_text = ""

            except httpx.HTTPError as err:     # network timeouts, DNS, etc.
                print("Network error - %s", err)
                skipped_instance_num += 1
                output_text = ""

        else:  # use local ckpt
            while True:  # TODO: Remove the tentative check on len(inputs["input_ids"]) > 8000 after debugging llama4
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                if inputs["input_ids"].shape[1] > 8000:
                    del(messages[-1]["content"][-2])
                    print(f"Input length: {inputs['input_ids'].shape}")
                    print("Input length exceeds 8,000. Drop appended images.")
                    print(f"len(messages[-1]['content']): {len(messages[-1]['content'])}")
                    continue
                else:
                    break
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.model_max_length,
                    # use_cache=False,  # TODO: This works but gives OOM and takes too much time - unusable
                )
            except RuntimeError as err:
                print("RuntimeError:", err)
                breakpoint()

            output_text = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]

        # TODO: Use llama4 or gpt-4o as llm-as-judge
        ground_truth_text = batch["answers_list"][0]["ans_full"][0]
        score, accuracy = None, None
        explanation = ""
        if args.use_llm_as_judge and output_text != "":
            score, explanation = evaluate_answer(
                output_text, ground_truth_text, model=args.llm_model
            )
            accuracy = score
        print(f">> score: {score}, explanation: {explanation}")

        logger.info(
            f"==> Output Text: {output_text} | Ground Truth Text: {ground_truth_text} | Accuracy: {accuracy}"
        )
        total_accuracy += accuracy if accuracy is not None else 0
        output_dict = {
            "session_id": batch["session_ids"][0],
            "question": question,
            "output_text": output_text,
            "ground_truth_text": ground_truth_text,
            "explanation": explanation,
            "accuracy": accuracy,
        }
        output_dicts.append(output_dict)

        # save the output_dict (line by line) to a jsonl files
        with open(pred_file_path, "a") as f:
            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")

    # calculate the total_accuracy
    total_accuracy /= len(output_dicts)
    logger.info(f"==> Total Accuracy: {total_accuracy}")
    logger.info(f"==> Output Dicts: {output_dicts}")
    logger.info(f"==> Skipped Instance Num: {skipped_instance_num}")


if __name__ == "__main__":
    # coloredlogs.install(level="DEBUG")

    main(sys.argv[1:])
