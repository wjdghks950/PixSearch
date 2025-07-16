import argparse
import json

import logging
import os
import re
import math
import subprocess
from io import BytesIO
from typing import Any, Dict, List
from statistics import mean, stdev

import coloredlogs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from cragmm_search.search import UnifiedSearchPipeline
from PIL import Image
from tqdm import tqdm

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process images and calculate recall@k."
    )
    parser.add_argument(
        "--entity_mapping",
        type=str,
        default="/home/jeonghkim/projects/region-mm-retrieve/data/st_entity_to_kg_entity.json",
        help="Path to entity name mapping JSON file.",
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="crag-mm-2025/[private] CRAG-MM QA master data - ST_snapshot_0415.csv",
        help="Path to the CSV data file.",
    )
    parser.add_argument("--max_k", type=int, default=30, help="Maximum k for recall@k.")
    parser.add_argument(
        "--center_crop",
        type=float,
        default=0,
        help="Center crop the image before processing. 0 for no cropping, >0-1 for ratio of the crop.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="Maximum number of successful queries to process.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="error_examples.txt",
        help="File to write error examples.",
    )
    parser.add_argument(
        "--cached_results_dir",
        type=str,
        default="/home/jeonghkim/projects/region-mm-retrieve/results/cragmm",
        help="Directory to load cached results from.",
    )
    parser.add_argument(
        "--cragmm_eval_master_data_json",
        type=str,
        default="/home/jeonghkim/projects/region-mm-retrieve/data/cragmm_qa_data_for_benchmark_eval.json",
        help="Path to the CRAG-MM Eval Master Data JSON file with gold entities per session_id.",
    )
    parser.add_argument(
        "--use_cached_results",
        action="store_true",
        help="Use cached results instead of processing images.",
    )
    parser.add_argument(
        "--use_whole_image",
        action="store_true",
        help="Use whole image vs region-level (e.g., bbox, mask) images",
    )
    parser.add_argument(
        "--use_split_k",
        action="store_true",
        help="Use a distributed context budge per cropped objects"
    )
    parser.add_argument(
        "--whole_image_retrieval_dir",
        type=str,
        default="/shared/nas2/jk100/partonomy_private/results/cragmm/whole_image_retrieval",
        help="Directory to load whole image retrieval results from.",
    )
    parser.add_argument(
        "--bbox_image_retrieval_dir",
        type=str,
        default="/shared/nas2/jk100/partonomy_private/results/cragmm/bbox_image_retrieval",
        help="Directory to load whole image retrieval results from.",
    )
    parser.add_argument(
        "--mask_image_retrieval_dir",
        type=str,
        default="/shared/nas2/jk100/partonomy_private/results/cragmm/mask_image_retrieval",
        help="Directory to load mask image retrieval results from.",
    )
    parser.add_argument(
        "--results_output_path",
        type=str,
        default="/shared/nas2/jk100/partonomy_private/results/cragmm/recall_results.json",
        help="Path to the output file for the recall results.",
    )
    return parser.parse_args()


def load_entity_name_mappings(entity_mapping_path):
    with open(entity_mapping_path, "r") as f:
        return json.load(f)


def create_search_pipeline():
    return UnifiedSearchPipeline(
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        image_model_name="openai/clip-vit-large-patch14-336",
        web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
        image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
    )


def center_crop(img, crop_ratio):
    img_width, img_height = img.size
    crop_width = img_width * crop_ratio
    crop_height = img_height * crop_ratio
    return img.crop(
        (
            (img_width - crop_width) // 2,
            (img_height - crop_height) // 2,
            (img_width + crop_width) // 2,
            (img_height + crop_height) // 2,
        )
    )


def get_object_stats(object_counts: List[int]) -> Dict[str, float]:
    """Return {mean, max, min, std} for a list of counts."""
    if not object_counts:
        return {"mean": 0, "max": 0, "min": 0, "std": 0}
    return {
        "mean": mean(object_counts),
        "max": max(object_counts),
        "min": min(object_counts),
        "std": stdev(object_counts) if len(object_counts) > 1 else 0,
    }


def allocate_budget_evenly(n_objects: int, k_budget: int) -> List[int]:
    """
    Split the total budget k_budget across n_objects as evenly as possible.

    Returns a list 'quota' of length n_objects, where quota[i] is the
    number of retrievals allowed for object i.  The first 'remainder'
    objects get one extra item so that sum(quota)==k_budget.
    """
    if n_objects == 0:
        return []
    base = k_budget // n_objects
    remainder = k_budget % n_objects
    # e.g. k=30, n=11  -> [3,3,3,3,3,3,3,3,3,3,3]  (base=2, remainder=8)
    return [base + 1 if i < remainder else base for i in range(n_objects)]


def build_fair_share_list(
    object_to_retrievals: Dict[str, List[Dict[str, Any]]],
    k_budget: int,
) -> List[Dict[str, Any]]:
    """
    Aggregate results from every object while respecting the per-object quota.
    The objects are traversed in a deterministic order (sorted by key).
    """
    keys = sorted(object_to_retrievals.keys())
    quota = allocate_budget_evenly(len(keys), k_budget)

    aggregated: List[Dict[str, Any]] = []
    for key, q in zip(keys, quota):
        aggregated.extend(object_to_retrievals[key][: q])
        if len(aggregated) >= k_budget:
            break
    return aggregated[: k_budget]


def load_cached_results(cached_results_dir):
    with open(cached_results_dir, "r") as f:
        return json.load(f)


def load_cragmm_eval_master_data_json(cragmm_eval_master_data_json):
    with open(cragmm_eval_master_data_json, "r") as f:
        return json.load(f)


def process_cached_retrieval_results(
    args,
    results: Dict[str, List[Dict[str, Any]]],
    entity_name_mapping: Dict[str, str],
    cragmm_eval_master_data_dict: Dict[str, Dict[str, Dict[str, Any]]],
    max_k: int = 30,
    output_file: str = "",
    use_whole_image: bool = True,
):
    """
    results:
    entity_name_mapping: Dict[str, str]
    cragmm_eval_master_data_dict: Dict[str, Dict[str, Dict[str, Any]]]
    max_k: int
    output_file: str
    use_whole_image: bool
    """
    recall_at_k = np.zeros(max_k)
    successful_queries = 0
    error_queries = 0
    error_examples = []
    entities_retrieved_at_k = {}

    if use_whole_image:
        for sid, retrieved_dicts in tqdm(
            results.items(), desc="Processing whole image retrieval results"
        ):
            # Get the ground-truth entity name and its mapped entity name
            try:
                gt_entity_name = cragmm_eval_master_data_dict[sid]["entity_name"]
            except KeyError:
                logger.warning(
                    f"No ground-truth entity name found for session_id: {sid}"
                )
                error_queries += 1
                continue

            mapped_gt_entity_name = entity_name_mapping.get(
                gt_entity_name, gt_entity_name
            )
            print(f"Processed {successful_queries+1}/{len(results)} images: {sid}")

            # Calculate the recall@k
            found_at_k = track_recall(
                retrieved_dicts, gt_entity_name, mapped_gt_entity_name, max_k
            )
            entities_retrieved_at_k[sid] = {
                "gt_entity_name": gt_entity_name,
                "mapped_gt_entity_name": mapped_gt_entity_name,
                "retrieved_entities": [
                    retrieved_dict["entities"][0]["entity_name"]
                    for retrieved_dict in retrieved_dicts
                ],
                "found_at_k": found_at_k,
            }
            if not found_at_k[
                max_k - 1
            ]:  # NOTE: This is just logging the @K sample - not error examples
                error_examples.append(
                    f"{sid}, Gold: {gt_entity_name}, Predicted: {retrieved_dicts[0]['entities'][0]['entity_name']}"
                )
            update_recall_counts(found_at_k, recall_at_k, max_k)
            successful_queries += 1

            # if max_queries != 0 and successful_queries > max_queries:
            #     break

            logger.info(
                f"Recall@{max_k}: {recall_at_k[max_k-1]:.4f} | Error Queries: {error_queries} | Successful Queries: {successful_queries}"
            )

        results_log_dict = {
            "recall_at_k": recall_at_k.tolist(),
            "recall_at_k_score": calculate_recall(recall_at_k, successful_queries).tolist(),
            "successful_queries": successful_queries,
            "error_queries": error_queries,
            "error_examples": error_examples,
            "entities_retrieved_at_k": entities_retrieved_at_k,
        }

    else:  # Image retrieval evaluation for bbox and mask-level image retrieval
        bbox_object_counts, mask_object_counts = [], []
        bbox_recall_at_k = np.zeros(max_k)
        bbox_entities_retrieved_at_k = {}
        pattern = re.compile(r"^cropped_(.+?)\.png$")
        bbox_image_retrieval_results = results[0]
        for sid, bbox_retrieved_dicts in tqdm(bbox_image_retrieval_results.items(), desc="Processing bbox image retrieval results"):
            bbox_object_counts.append(len(bbox_retrieved_dicts))
            # Get the ground-truth entity name and its mapped entity name
            try:
                gt_entity_name = cragmm_eval_master_data_dict[sid]["entity_name"]
            except KeyError:
                logger.warning(
                    f"No ground-truth entity name found for session_id: {sid}"
                )
                error_queries += 1
                continue

            mapped_gt_entity_name = entity_name_mapping.get(
                gt_entity_name, gt_entity_name
            )
            print(f"Processed {successful_queries+1}/{len(results)} images: {sid}")

            # Calculate the recall@k - note, per sid, there are "multiple" image regions - i.e., bboxes
            if args.use_split_k:
                # 1) build one aggregated result list using the quota rule
                aggregated = build_fair_share_list(bbox_retrieved_dicts, max_k)
                found_at_k = track_recall(aggregated, gt_entity_name, mapped_gt_entity_name, max_k)
                retrieved_entities = [r["entities"][0]["entity_name"] for r in aggregated]
                most_likely_object = "(split-k)"
                max_found_at_k = found_at_k
            else:
                most_likely_object = ""
                max_object_recall = 0
                for bbox_path, bbox_retrieved_list in bbox_retrieved_dicts.items():  # For every bbox, take the max-recall object
                    found_at_k = track_recall(bbox_retrieved_list, gt_entity_name, mapped_gt_entity_name, max_k)
                    if sum(found_at_k) > max_object_recall:
                        max_found_at_k = found_at_k.copy()
                        max_object_recall = sum(found_at_k)
                        most_likely_object = pattern.match(os.path.basename(bbox_path)).group(1)  # e.g., # '/shared/nas2/jk100/partonomy_private/results/cragmm/df084975-1781-4be6-9177-d5434e1ce28d/cropped_bboxes/cropped_advertisement.png'
                        retrieved_entities = [retrieved_dict["entities"][0]["entity_name"] for retrieved_dict in bbox_retrieved_list]

            # Calculate the recall@k
            bbox_entities_retrieved_at_k[sid] = {
                "gt_entity_name": gt_entity_name,
                "mapped_gt_entity_name": mapped_gt_entity_name,
                "most_likely_object": most_likely_object,
                "retrieved_entities": retrieved_entities,
                "found_at_k": max_found_at_k,
            }
            if not max_found_at_k[
                max_k - 1
            ]:  # NOTE: This is just logging the @K sample - not error examples
                error_examples.append(
                    f"{sid}, Gold: {gt_entity_name}, Predicted: {most_likely_object}"
                )
            update_recall_counts(max_found_at_k, bbox_recall_at_k, max_k)
            successful_queries += 1

            # if max_queries != 0 and successful_queries > max_queries:
            #     break

            logger.info(
                f"Bbox - Recall@{max_k}: {bbox_recall_at_k[max_k-1]:.4f} | Error Queries: {error_queries} | Successful Queries: {successful_queries}"
            )

        # Mask-level image retrieval
        mask_recall_at_k = np.zeros(max_k)
        mask_entities_retrieved_at_k = {}
        pattern = re.compile(r"^cropped_(.+?)_zoom\.png$")
        mask_image_retrieval_results = results[1]
        for sid, mask_retrieved_dicts in tqdm(mask_image_retrieval_results.items(), desc="Processing mask image retrieval results"):
            mask_object_counts.append(len(mask_retrieved_dicts))
            # Get the ground-truth entity name and its mapped entity name
            try:
                gt_entity_name = cragmm_eval_master_data_dict[sid]["entity_name"]
            except KeyError:
                logger.warning(
                    f"No ground-truth entity name found for session_id: {sid}"
                )
                error_queries += 1
                continue

            mapped_gt_entity_name = entity_name_mapping.get(
                gt_entity_name, gt_entity_name
            )
            print(f"Processed {successful_queries+1}/{len(results)} images: {sid}")

            # Calculate the recall@k - note, per sid, there are "multiple" image regions - i.e., masks
            if args.use_split_k:
                # 1) build one aggregated result list using the quota rule
                aggregated = build_fair_share_list(mask_retrieved_dicts, max_k)
                found_at_k = track_recall(aggregated, gt_entity_name, mapped_gt_entity_name, max_k)
                retrieved_entities = [r["entities"][0]["entity_name"] for r in aggregated]
                most_likely_object = "(split-k)"
                max_found_at_k = found_at_k
            else:
                most_likely_object = ""
                max_object_recall = 0
                for mask_path, mask_retrieved_list in mask_retrieved_dicts.items():  # For every mask, take the max-recall object
                    found_at_k = track_recall(mask_retrieved_list, gt_entity_name, mapped_gt_entity_name, max_k)
                    if sum(found_at_k) > max_object_recall:
                        max_found_at_k = found_at_k.copy()
                        max_object_recall = sum(found_at_k)
                        most_likely_object = pattern.match(os.path.basename(mask_path)).group(1)  # e.g., # '/shared/nas2/jk100/partonomy_private/results/cragmm/df084975-1781-4be6-9177-d5434e1ce28d/cropped_maskes/cropped_advertisement.png'
                        retrieved_entities = [retrieved_dict["entities"][0]["entity_name"] for retrieved_dict in mask_retrieved_list]
            # Calculate the recall@k
            mask_entities_retrieved_at_k[sid] = {
                "gt_entity_name": gt_entity_name,
                "mapped_gt_entity_name": mapped_gt_entity_name,
                "most_likely_object": most_likely_object,
                "retrieved_entities": retrieved_entities,
                "found_at_k": max_found_at_k,
            }
            if not max_found_at_k[
                max_k - 1
            ]:  # NOTE: This is just logging the @K sample - not error examples
                error_examples.append(
                    f"{sid}, Gold: {gt_entity_name}, Predicted: {most_likely_object}"
                )
            update_recall_counts(max_found_at_k, mask_recall_at_k, max_k)
            successful_queries += 1

            # if max_queries != 0 and successful_queries > max_queries:
            #     break

            logger.info(
                f"Mask - Recall@{max_k}: {mask_recall_at_k[max_k-1]:.4f} | Error Queries: {error_queries} | Successful Queries: {successful_queries}"
            )

        results_log_dict = {
            "bbox_recall_at_k": bbox_recall_at_k.tolist(),
            "bbox_recall_at_k_score": calculate_recall(bbox_recall_at_k, successful_queries).tolist(),
            "mask_recall_at_k": mask_recall_at_k.tolist(),
            "mask_recall_at_k_score": calculate_recall(mask_recall_at_k, successful_queries).tolist(),
            "successful_queries": successful_queries,
            "error_queries": error_queries,
            "error_examples": error_examples,
            "bbox_entities_retrieved_at_k": bbox_entities_retrieved_at_k,
            "mask_entities_retrieved_at_k": mask_entities_retrieved_at_k,
            "bbox_object_stats": get_object_stats(bbox_object_counts),
            "mask_object_stats": get_object_stats(mask_object_counts)
        }

    # save and log the evaluation results
    with open(output_file, "w") as f:
        json.dump(results_log_dict, f)

    return recall_at_k, successful_queries, error_queries


def process_images(
    df,
    entity_name_mapping,
    search_pipeline,
    max_k,
    max_queries,
    center_crop_size,
    output_file: str = "",
):
    recall_at_k = np.zeros(max_k)
    successful_queries = 0
    error_queries = 0
    error_examples = []

    for _, row in df.iterrows():
        entity_name = row["entity_name"]
        mapped_entity_name = entity_name_mapping.get(entity_name, entity_name)
        print(f"Processed {successful_queries+1}/{len(df)} images: {row['image_link']}")

        image = get_image_path(row)
        if image is None:
            error_queries += 1
            continue
        if center_crop_size > 0:
            if isinstance(image, str):
                image = Image.open(image)
            image = center_crop(image, center_crop_size)
        results = search_pipeline(image, k=max_k)
        found_at_k = track_recall(results, entity_name, mapped_entity_name, max_k)

        if not found_at_k[max_k - 1]:
            error_examples.append(
                f"{row['image_link']}, Gold: {entity_name}, Predicted: {results[0]['entities'][0]['entity_name']}"
            )

        update_recall_counts(found_at_k, recall_at_k, max_k)
        successful_queries += 1
        if max_queries != 0 and successful_queries > max_queries:
            break

    with open(output_file, "w") as f:
        for example in error_examples:
            f.write(f"{example}\n")

    return recall_at_k, successful_queries, error_queries


def get_image_path(row):
    if "manifold" in row["image_link"]:
        return download_manifold_image(row["image_link"])
    else:
        return download_image(row["image_link"])


def download_manifold_image(image_link):
    manifold_path = image_link.split("://")[1].split("/")
    bucket = manifold_path[3]
    path = "/".join(manifold_path[4:])
    image_path = f"crag-mm-2025/images/{bucket}/{path}"

    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        command = f"manifold get {bucket}/{path} {image_path}"
        try:
            subprocess.run(command.split(), check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
            print(e.stderr)
            return None
    return image_path


def download_image(image_link):
    headers = {
        "User-Agent": "CragMM/0.1 (https://example.org/cragmm/; cragmm@example.org)"
    }
    response = requests.get(image_link, headers=headers)
    if response.status_code == 200:
        try:
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error: Failed to open image from {image_link}")
            return None
    else:
        print(f"Error: Failed to download image from {image_link}")
        return None


def track_recall(results, entity_name, mapped_entity_name, max_k):
    found_at_k = [False] * max_k
    for k, result in enumerate(results):
        if k >= max_k:
            break

        retrieved_entity_name = result["entities"][0]["entity_name"]
        if (
            retrieved_entity_name.lower() == entity_name.lower()
            or retrieved_entity_name.lower() == mapped_entity_name.lower()
        ):
            for j in range(k, max_k):
                found_at_k[j] = True
            break
    return found_at_k


def update_recall_counts(found_at_k, recall_at_k, max_k):
    for k in range(max_k):
        if found_at_k[k]:
            for j in range(k, max_k):
                recall_at_k[j] += 1
            break


def calculate_recall(recall_at_k, successful_queries):
    if successful_queries > 0:
        return recall_at_k / successful_queries
    else:
        print("No successful queries were processed")
        return recall_at_k


def main():
    args = parse_arguments()
    entity_name_mapping = load_entity_name_mappings(args.entity_mapping)
    cragmm_eval_master_data_dict = load_cragmm_eval_master_data_json(
        args.cragmm_eval_master_data_json
    )["validation"]
    if args.use_cached_results:
        if args.use_whole_image:
            cached_results = load_cached_results(
                os.path.join(
                    args.cached_results_dir,
                    "whole_image_retrieval",
                    "whole_image_retrieval_cache.json",
                )
            )
        else:
            bbox_cached_results = load_cached_results(
                os.path.join(
                    args.cached_results_dir,
                    "bbox_image_retrieval",
                    "bbox_image_retrieval_cache.json",
                )
            )
            mask_cached_results = load_cached_results(
                os.path.join(
                    args.cached_results_dir,
                    "mask_image_retrieval",
                    "mask_image_retrieval_cache.json",
                )
            )
            cached_results = (bbox_cached_results, mask_cached_results)

        cragmm_eval_master_data_dict = load_cragmm_eval_master_data_json(
            args.cragmm_eval_master_data_json
        )["validation"]

        # TODO: Clean this up - this is a hack to get the results to be cached
        results_output_path = "/shared/nas2/jk100/partonomy_private/results/cragmm/recall_results.json"
        if args.use_split_k:
            results_output_path = "/shared/nas2/jk100/partonomy_private/results/cragmm/recall_results_split_k.json"

        process_cached_retrieval_results(
            args,
            cached_results,
            entity_name_mapping,
            cragmm_eval_master_data_dict,
            max_k=args.max_k,
            output_file=args.results_output_path,
            use_whole_image=args.use_whole_image,
        )
    else:
        search_pipeline = create_search_pipeline()

        df = pd.read_csv(args.data_csv)
        df = df[df["entity_name_source"] == "kg"]
        df = df[df["split"] == "validation"]
        max_k = args.max_k
        output_plot_name = f"image_recall_k{max_k}_q{args.max_queries}.png"
        plt.figure(figsize=(10, 6))
        for cc in [1]:  # [0.5, 0.75, 0.85, 0.9, 0.95, 1]:
            recall_at_k, successful_queries, error_queries = process_images(
                df,
                entity_name_mapping,
                search_pipeline,
                args.max_k,
                args.max_queries,
                cc,
                args.output_file,
            )

            recall_at_k = calculate_recall(recall_at_k, successful_queries)

            print(f"Successful queries: {successful_queries}")
            print(f"Error queries: {error_queries}")
            print("\nRecall@k Results:")
            for k in range(max_k):
                print(f"Recall@{k+1}: {recall_at_k[k]:.4f}")
            plt.plot(
                range(1, max_k + 1), recall_at_k, marker="o", label=f"crop ratio: {cc}"
            )

        plt.xlabel("k")
        plt.ylabel("Recall@k")
        plt.title("Image Search Recall@k")
        plt.grid(True)
        plt.xticks(range(1, max_k + 1, 2))
        plt.legend()
        plt.savefig(output_plot_name)


if __name__ == "__main__":
    main()
