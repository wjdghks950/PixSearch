# multimodal_rag_dataset_builder

import os, io, json, uuid, time, random, sys
import logging, argparse, re, glob, requests
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from openai import OpenAI
from datasets import load_dataset, Dataset
from PIL import Image
from tqdm import tqdm
from itertools import islice

from evaluation.rle_dict import get_mask_rle_dicts
from utils.utils import (
                            START_INFO_TOKEN,
                            END_INFO_TOKEN,
                            START_SEARCH_TOKEN,
                            END_SEARCH_TOKEN,
                            image_to_base64,
                            DATASETS,
                            RETRY_BACKOFF_SECONDS,
                            TEMPERATURE,
                            resize_keep_aspect,
                            image_to_pil,
                            load_image_by_id,
                            extract_dict_like_str,
                            load_webqa_tsv_images,
                            cache_image,
                            extract_search_queries,
                            append_retrieved_items,
                            call_local_search
                        )
from crop_cragmm_images import generate_gold_mask
from local_search_api import LocalSearchPipeline
from cragmm_search.search import UnifiedSearchPipeline


# determine whether a question needs external knowledge or not
SYS_NEEDS_RETRIEVAL = (
    "You are an expert at analyzing whether a visual question needs external knowledge "
    "that is not contained in the image. First, you should reason step-by-step and provide a detailed explanation for whether a visual question needs retrieval or not. "
    "There may be no input image for the question, but it may still require retrieval of external image or text knowledge to be answered. "
    "Return exactly JSON with keys `{reasoning: str, needs_retrieval: bool}`."
)
USER_NEEDS_RETRIEVAL = (
    "If the question can be answered with the following criteria, it does not need retrieval so set `needs_retrieval = false`:\n"
    "* question can be answered by image alone.\n"
    "* needs translation.\n"
    "* can rely on common sense reasoning or basic knowledge by a typical human.\n"
    "* requires situation awareness or context that does not necessitate additional knowledge.\n"
    "However, if the question requires the following conditions to answer correctly, it needs retrieval so set `needs_retrieval = true`:\n"
    "* external / encyclopedic / domain-specific / product-related / historical / wikipedia knowledge.\n"
    "* identifying object categories and related knowledge.\n"
    "* anything that requires searching on the internet and knowledge beyond the pixels to answer correctly.\n"
    "Here are some examples:\n"
)

# Question decomposition system and user prompts
SYS_QUESTION_DECOMPOSE = (
    "You are an expert at decomposing a question into multiple atomic subquestions. "
    "Return exactly JSON with keys `{reasoning: str, is_decomposed: bool, subquestions: List[str], needs_retrieval: List[str]}`."
)
USER_QUESTION_DECOMPOSE = (
    "Given a question, you should determine if it should be decomposed into a series of atomic subquestions (set 'is_decomposed' to true or false). "
    "If `is_decomposed: true`, then decompose the question into multiple atomic subquestions in `subquestions`. "
    "For each subquestion, decide whether it needs to be answered with external knowledge (append true or false to `needs_search`list). "
    "To determine whether a subquestion needs to be answered with external knowledge, use the following criteria:\n"
    "* If the question can be answered with the following criteria, it does not need retrieval so set `needs_retrieval = false`:\n"
    "1. question can be answered by image alone.\n"
    "2. needs translation.\n"
    "3. can rely on common sense reasoning or basic knowledge by a typical human.\n"
    "4. requires situation awareness or context that does not necessitate additional knowledge.\n"
    "* However, if the question requires the following conditions to answer correctly, it needs retrieval so set `needs_retrieval = true`:\n"
    "1. external / encyclopedic / domain-specific / product-related / historical / wikipedia knowledge.\n"
    "2. identifying object categories and related knowledge.\n"
    "3. anything that requires searching on the internet and knowledge beyond the pixels to answer correctly.\n"
    "If `is_decomposed: false`, then set `subquestions` to an empty list. "
    "Here are some examples:\n"
)

# generate answer draft for the questions (with/without <search> token)
SYS_DRAFT_ANSWER = (
    "You are a visual reasoning assistant answering questions about an image as a smart-glasses assistant. "
    "Your answer must be concise and returned as JSON with exactly one key: {answer: str}. "
    "When external knowledge is required, you will mark the exact place in your text where retrieval should happen "
    "by inserting the token <search>. "
    "CRITICAL: Place <search> immediately BEFORE the first token of any fact fragment that requires external knowledge "
    "(e.g., a proper noun, date, model name, statistic, title). Do NOT state that fact before the <search> token. "
    "Use exactly one space after <search> and continue the sentence. "
    "Never add code fences, additional keys, or explanations—only JSON with {answer: ...}."
)

USER_DRAFT_ANSWER = (
    "Write a draft answer for the question about the image. "
    "If you do not need external knowledge, do NOT use <search>. "
    "Return exactly JSON with {\"answer\": \"...\"}."
)

USER_DRAFT_ANSWER_FOR_RETRIEVAL = (
    "Write a draft answer for the question about the image. You are given: the question, ground-truth answers, "
    "and the decomposition (reasoning, subquestions, and for each subquestion whether it needs retrieval). "
    "Rules for <search>:\n"
    "1) Insert one <search> for EACH subquestion with needs_retrieval == True.\n"
    "2) Place <search> IMMEDIATELY BEFORE the answer fragment for that subquestion (the very first token of the grounded fact), "
    "so retrieval happens BEFORE you state the fact. Do not place <search> after the fact or at the end of a sentence.\n"
    "3) Use exactly one space after <search> and then continue the fact text.\n"
    "4) For subquestions with needs_retrieval == False, do NOT use <search>.\n"
    "5) The total number of <search> tokens MUST equal the number of subquestions where needs_retrieval == True.\n"
    "Return exactly JSON with {\"answer\": \"...\"}.\n"
    "Good examples:\n"
    "- Q: From what brand can you buy the vehicle the person is driving?\n"
    "  A: {\"answer\": \"I see a blue car <search> Subaru WRX 2021.\"}\n"
    "- Q: Which car has a better gas mileage?\n"
    "  A: {\"answer\": \"The car on the left <search> Subaru WRX averages 19 city / 26 highway, "
    "and the car on the right <search> Honda CR-V 2WD averages 28 city / 34 highway.\"}\n"
    "- Q: What is the theater and when did the namesake die?\n"
    "  A: {\"answer\": \"The theater shown in the image <search> is named after Ricardo Montalban, "
    "and he died in <search> 2009.\"}\n"
    "Here are some additional examples:\n"
)

# determine the modality for each <search> token
SYS_QUERY_ROUTER = (
    "You are deciding which *single* retrieval modality the agent should use for each <search> placeholder *in order of preference*:"
    "1. REGION - when the answer relies on a particular object, logo, region, "
    "or clearly bounded area in the image. Note, region-level image-to-image retrieval also returns additional metadata "
    "(e.g., entity names, wikipedia documents, product details) related to the object, logo, text, or area.\n"
    "2. IMAGE - when understanding the whole scene (landscape, crowd, weather) is required and no single object suffices.\n"
    "3. TEXT - only when neither REGION nor IMAGE would return useful hits."
    "TEXT queries cost 5 times more latency; avoid them unless absolutely necessary.\n"
    "Return exactly JSON with keys `{reasoning: str, query_types: List[str], queries: List[str]}`.\n"
    "Here are some examples:\n"
)
USER_QUERY_ROUTER = (
    "You should decide what retrieval modality (REGION, IMAGE, TEXT) the multimodal agent should use for each `<search>` request. "
    "You will be provide a draft answer that contains `<search>` token(s) and a question asking about the image. "
    "For each `<search>` token, you should decide which modality (REGION, IMAGE, TEXT) to use as a query for effecitve search. "
    "For example, when asked about what is the object seen in the image, you should use REGION since we can segment it and use it as a query."
    "Rules:\n"
    "* If the answer needs additional external information/knowledge about specific visible objects/entities appearing in the image, choose REGION.\n"
    "* If the question requires text-based search or recognizing certain texts identified in the image, choose TEXT.\n"
    "* If recognizing the entire scene for context is crucial, choose IMAGE.\n"

    "Make sure to reason step-by-step and provide a detailed explanation of your decision in the reasoning field. "

    "For `queries` field, return the query for each `<search>` token and according to their assigned query_type in the `query_types`: "
    "Rules:\n"
    "* If the query type is REGION, then the `queries` field should contain a list of noun phrases that we can use for open-vocabulary segmentation or referring segmentation models."
    "* If the query type is IMAGE, just append 'image' to the `queries` list. "
    "* If the query type is TEXT, generate a question that can be used as a textual query to the search API and append the generated text to the `queries` list. "

    "Query type must be from one of the following: REGION, IMAGE, or TEXT. "
    "Note, you need to mind this constraint as well: `len(query_types) == len(queries) == answer.count('<search>')`.\n"
    "Here are some examples:\n"
)


def build_in_context_cache(
    k: int,
    dataset: Dataset,
    dataset_name: str,
    cache_path: str | Path,
    prompt_type: str,
    resize_img: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return K in-context A-OKVQA samples (load cache if present; else build+cache).

    Each example is already a dict that can be spliced into `messages`:
        { "role": "user",
          "content": [
                {"type":"text","text": question},
                {"type":"image_url","image_url":{"url": data_uri}}
            ]
        }

    Args:
    - k : int
        Number of examples to sample.
    - cache_path : str | Path
        Where to save / reload the cached JSON lines file.

    Returns: List[dict]
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[In-Context] Loading {k} examples from cache: {cache_path}")
        with cache_path.open() as f:
            incontext_samples = [json.loads(line) for _, line in zip(range(k), f)]
            if len(incontext_samples) == k:
                print(f"[In-Context] Cache found → loading {k} examples")
                for sample in incontext_samples:
                    if not sample["content"][-1]["image_url"]["url"].startswith("data:"):
                        img_path = sample["content"][-1]["image_url"]["url"]
                        pil_img = image_to_pil(img_path) if not resize_img else resize_keep_aspect(image_to_pil(img_path))
                        img = image_to_base64(pil_img, with_data_uri=True)
                        sample["content"][-1]["image_url"]["url"] = img
                return incontext_samples
            else:
                k -= len(incontext_samples)
                print(f"[In-Context] Cache found but incomplete → rebuilding")

    # build cache of in-context samples
    rng = random.Random(42)          # reproducible seed
    indices = rng.sample(range(len(dataset)), k)
    records: List[Dict[str, Any]] = []

    img_dir = cache_path.parent / "images"
    os.makedirs(img_dir, exist_ok=True)

    if prompt_type == "modality":  # for per <search> token modality routing
        # load from `answer_incontext_sample.jsonl` and use it for modality routing few-shot sample construction
        answer_incontext_sample_path = os.path.join(cache_path.parent, "answer_incontext_sample.jsonl")
        answer_incontext_samples = []
        if os.path.exists(answer_incontext_sample_path):
            with open(answer_incontext_sample_path, "r") as f:
                answer_incontext_samples = [json.loads(line) for line in f]
        else:
            raise FileNotFoundError(f"answer_incontext_sample.jsonl not found in {cache_path.parent}")

    if prompt_type == "answer":
        # load from `qdecomp_incontext_sample.jsonl` and use it for answer draft generation few-shot sample construction
        qdecomp_incontext_sample_path = os.path.join(cache_path.parent, "qdecomp_incontext_sample.jsonl")
        qdecomp_incontext_samples = []
        if os.path.exists(qdecomp_incontext_sample_path):
            with open(qdecomp_incontext_sample_path, "r") as f:
                qdecomp_incontext_samples = [json.loads(line) for line in f]
        else:
            raise FileNotFoundError(f"qdecomp_incontext_sample.jsonl not found in {cache_path.parent}")

    for i, sample_idx in enumerate(indices):
        if dataset_name == "aokvqa":
            sample = dataset[sample_idx]
            q = sample["question"]
            qid = sample["question_id"]
            img = sample["image"]
        elif dataset_name == "cragmm":
            sample = dataset[sample_idx]
            q = sample["turns"]["query"][0].strip()
            qid = sample['session_id']
            img = sample["image"]

        img_path = img
        if isinstance(img, Image.Image):
            img_path = os.path.join(img_dir, f"{qid}.png")
            img.save(img_path)
        else:
            img_path = img

        # prompt input from user for response construction
        if prompt_type == "needs_retrieval":
            print(f"--------Sample - qid: {qid}--------\n\n"
                "Question:\nFrom what brand can you buy the vehicle the person is driving?\n{'reasoning': 'the type of vehicle requires external knowledge to identify.', 'needs_retrieval': true} \n"
                "Question:\nWhat animal will most likely eat this meal?\n{'reasoning': 'you can see that this meal is sandwich, so humans will most likely eat this meal.', 'needs_retrieval': false} \n"
                "Question:\nIn which country is this bus located?\n{'reasoning': 'the type of the bus require image search to identify.', 'needs_retrieval': true} \n\n")
            print(f"Question:\n{q}\n")
            reasoning = input("reasoning (explain your reasoning) : ").lower().strip()
            needs_retrieval = input("needs_retrieval (true/false) : ").lower().strip()
            if needs_retrieval not in ["true", "false"]:
                print("Invalid response. Please enter 'true' or 'false'.")
                continue
            json_response = {
                "reasoning": reasoning,
                "needs_retrieval": needs_retrieval,
            }
            incontext_sample = f"Question:\n{q}\n{json_response}\n"

        elif prompt_type == "qdecomp":
            print(f"--------Sample - qid: {qid}--------\n\n")
            print("Question:\nFrom what brand can you buy the vehicle the person is driving?\n{'reasoning': 'the type of vehicle requires external knowledge to identify.', 'is_decomposed': false, 'subquestions': ['what brand is this car?'], 'needs_retrieval': [true]} \n")
            print("Question:\nWhat animal will most likely eat this meal?\n{'reasoning': 'you can see that this meal is sandwich, so humans will most likely eat this meal.', 'is_decomposed': true, 'subquestions': ['what is this meal?', 'what animal most likely eats this meal?'], 'needs_retrieval': [false, false]}\n")
            print("\n>> Return exactly JSON with keys `{reasoning: str, is_decomposed: bool, subquestions: List[str], needs_retrieval: List[str]}`.<<\n")
            print(f"Question:\n{q}\n")
            reasoning = input("reasoning (explain your reasoning for question decomposition) : ").lower().strip()
            is_decomposed = input("is_decomposed (true/false) : ").lower().strip()
            if is_decomposed == "true":
                questions = input("subquestions (delineate subquestions with '||') : ").lower().strip()
                subquestions = [question.lower().strip() for question in questions.split("||")]
            else:
                subquestions = [q.lower().strip()]
            k = 0
            needs_retrieval = []
            while k < len(subquestions):  # for each subquestion determine whether it needs external knowledge or not
                is_retrieval = input(f"needs_retrieval for [ '{subquestions[k].strip()}' ] (true/false) : ").lower().strip()
                if is_retrieval not in ["true", "false"]:
                    print("Invalid response. Please enter 'true' or 'false'.")
                    continue
                needs_retrieval.append(is_retrieval)
                k += 1

            assert len(needs_retrieval) == len(subquestions)

            json_response = {
                "reasoning": reasoning,
                "is_decomposed": is_decomposed,
                "subquestions": subquestions,
                "needs_retrieval": needs_retrieval
            }
            incontext_sample = f"Question:\n{q}\n{json_response}\n"

        elif prompt_type == "answer":  # for answer drafting
            print(f"--------Sample - qid: {qid}--------\n\n")
            print("Question:\nFrom what brand can you buy the vehicle the person is driving?\n{'answer': 'I see a blue car <search> which I have identified as Subaru WRX 2021.'} \n")
            print("Question:\nWhich car has a better gas mileage?\n{'answer': 'The car on the left <search> is a Subaru WRX, which has an average of 19 mpg city and 26 mpg highway, and the car on the right <search> is a Honda CR-V 2WD, which has 28 mpg city and 34 mpg highway.'} \n")
            print(f"Question:\n{q}\n")
            print(">>Question Decomposition sample: ", qdecomp_incontext_samples[i]["content"][0]["text"], "\n\n")
            answer_draft = input("answer_draft (response interleaved with <search>) : ").lower().strip()
            json_response = {
                "answer": answer_draft
            }
            incontext_sample = f"Question:\n{q}\n{json_response}\n"

        elif prompt_type == "modality":  # TODO: for per <search> token modality routing <- this needs to be fixed to accommodate qdecomp
            print(f"--------Sample - qid: {qid}--------\n")
            print("Question:\nFrom what brand can you buy the vehicle the person is driving?\nAnswer: I see a blue car <search> which I have identified as Subaru WRX 2021.\n{'num_search_toks': 1, 'query_type': [REGION]}\n")
            print("Question:\nWhich car has a better gas mileage?\nAnswer: The car on the left <search> is a Subaru WRX, which has an average of 19 mpg city and 26 mpg highway, and the car on the right <search> is a Honda CR-V 2WD, which has 28 mpg city and 34 mpg highway.\n{'num_search_toks': 2, 'query_type': [REGION, REGION]}\n")
            print(">> In-context Sample:\n")
            print(answer_incontext_samples[i]["content"][0]["text"])
            print(f"\nQuestion:\n{q}\n")
            num_search_toks = int(input("num_search_toks : ").strip())
            reasoning = input("reasoning (explain your reasoning) : ").lower().strip()
            query_types = []
            queries = []
            for t in range(num_search_toks):
                query_type = input(f"query_type (REGION, IMAGE, TEXT) for the {t}-th <search> token: ").strip()
                query = input(f"query for the {t}-th <search> token: ").strip()
                query_types.append(query_type)
                queries.append(query)

            json_response = {
                "context": answer_incontext_samples[i]["content"][0]["text"],
                "reasoning": reasoning,
                "num_search_toks": num_search_toks,
                "query_types": query_types,
                "queries": queries
            }
            incontext_sample = f"{json_response}\n"

            print(f">> INCONTEXT_SAMPLE\n: {incontext_sample}\n")
            print(f"i: {i} | context: {answer_incontext_samples[i]['content'][0]['text']}")

        records.append({
            "role": "user",
            "content": [
                {"type": "text", "text": incontext_sample},
                {"type": "image_url",
                 "image_url": {"url": img_path}}
            ]
        })

    # write one JSON object per line
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[In-Context] Cached examples → {cache_path}")

    return records

def call_llama(args,
               client,
               system_prompt: str,
               user_prompt: str,
               question: str,
               image_path: Optional[Union[Image.Image, str, os.PathLike]] = None,
               incontext_samples: Optional[List[Any]] = None,
               max_tokens: int = 512,
               json_mode=False,
               json_schema=None) -> str:
    """
    Blocking helper with simple exponential back-off retries.
    """
    image_base64 = image_to_base64(image_path, with_data_uri=True) if image_path else None

    # TODO: Generate JSON with llama4, log it and re-generate the answer with another LLM
    # TODO: Python dictionary version of the output
    pass


# openai helper
def call_openai(args,
                client,
                system_prompt: str,
                user_prompt: str,
                question: str,
                image_path: Optional[List[Union[Image.Image, str, os.PathLike]]] = None,
                incontext_samples: Optional[List[Any]] = None,
                max_tokens: int = 512,
                json_mode=False,
                json_schema=None) -> str:
    """
    Blocking helper with simple exponential back-off retries.
    """
    image_base64 = [image_to_base64(img, with_data_uri=True) for img in image_path] if image_path else None

    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
        ],
    }
    if incontext_samples is not None:
        for sample in incontext_samples:
            user_msg["content"].extend(sample["content"])

    user_msg["content"].append({"type": "text", "text": f"Question:\n{question}\n"})

    if image_base64 is not None:
        for im_base64 in image_base64:
            user_msg["content"].append({"type": "image_url", "image_url": {"url": im_base64}})

    params = dict(
        model=args.llm_model,
        temperature=TEMPERATURE,
        messages=[{"role": "system", "content": system_prompt}, user_msg],
        max_tokens=max_tokens
    )
    if json_schema is not None:
        params['tools'] = [json_schema]
        params['tool_choice'] = {"type": "function", "function": {"name": json_schema["function"]["name"]}}
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    for backoff in RETRY_BACKOFF_SECONDS + [None]:
        try:
            response = client.chat.completions.create(**params)
            if json_schema is not None:
                router = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                query_types = router["query_types"]
                queries = router["queries"]
            return response.choices[0].message.content.strip() if json_schema is None else router
        except Exception as e:
            if backoff is None:        # last attempt – raise
                raise e


def process_sample(args,
                   client,
                   sample: Dict[str, Any],
                   dataset_name: str,
                   out_dir: Path = "",
                   incontext_samples_dict: Optional[Dict[str, Any]] = None,
                   ) -> Optional[Dict[str, Any]]:
    """
    Process a single sample from the dataset.

    Args:
        args: Namespace with arguments from `argparse`.
        client: OpenAI / Llama API client.
        sample: Sample from the dataset.
        dataset_name: Name of the dataset.
        out_dir: Output directory for saving the results.
        incontext_samples_dict: Dictionary with in-context samples for each prompt type.
    """

    # for every dataset_name, cache image_path or load from image_path once qid is already there.
    image_cache_dir = os.path.join(out_dir, "images")
    os.makedirs(image_cache_dir, exist_ok=True)

    if dataset_name == "aokvqa":
        qid = sample.get("question_id") or str(uuid.uuid4())
        raw_img = sample.get("image") or sample.get("image_path")
        image_path = [raw_img] if raw_img is not None else []
        cached_paths = cache_image(image_path, image_cache_dir, qid)
        question = sample.get("question")
        mc_choices = sample.get("choices")
        answers = sample.get("direct_answers")
        rationales = sample.get("rationales")

    elif dataset_name == "okvqa":
        qid = sample.get("question_id") or str(uuid.uuid4())
        raw_img = sample.get("image") or sample.get("image_path")
        image_path = [raw_img] if raw_img is not None else []
        cached_paths = cache_image(image_path, image_cache_dir, qid)
        question = sample.get("question")
        mc_choices = []
        answers = [ans["answer"] for ans in sample["answers"]]
        rationales = []

    elif dataset_name == "webqa":
        qid = sample.get("guid") or str(uuid.uuid4())
        # webqa has no question-paired image (but requires retrieved images/texts to answer)
        image_gold_retrieved = sample.get("img_posFacts")
        text_gold_retrieved = sample.get("txt_posFacts")
        image_ids = {
                item["image_id"]: item["caption"] for item in image_gold_retrieved
            }
        id2image, cached_paths = load_webqa_tsv_images(args.webqa_tsv_path, args.webqa_lineidx_path, list(image_ids.keys()), args.webqa_img_cache_dir)
        image_path = [image for id, image in id2image.items()]
        cached_paths = cache_image(image_path, image_cache_dir, qid)
        question = sample.get("question")  # 'question_category' is in webqa
        answers = [sample.get("answer")]
        mc_choices = []
        rationales = []

    elif dataset_name == "vizwiz":
        qid = sample.get('id')
        raw_img = sample.get("image") or sample.get("image_path")
        image_path = [raw_img] if raw_img is not None else []
        cached_paths = cache_image(image_path, image_cache_dir, qid)
        question = sample.get("question")
        answers = sample.get("answers")
        answerable = sample.get("answerable")
        mc_choices = []
        rationales = []

    elif dataset_name == "textvqa":
        qid = sample.get("question_id") or str(uuid.uuid4())
        image_id = sample.get("image_id")
        raw_img = sample.get("image") or sample.get("image_path")
        image_path = [raw_img] if raw_img is not None else []
        cached_paths = cache_image(image_path, image_cache_dir, f"{qid}_{image_id}")
        question = sample.get("question")
        answers = sample.get("answers")
        ocr_tokens = sample.get("ocr_tokens")
        mc_choices = []
        rationales = []

    elif dataset_name == "infoseek":
        qid = sample.get("data_id") or str(uuid.uuid4())
        image_id = sample.get("image_id")
        question = sample.get("question")
        answers = sample.get("answer") + sample.get("answer_eval")
        image_path = [sample.get("image")] or [sample.get("image_path")]
        cached_paths = []
        for ext in [".jpg", ".JPEG", ".jpeg"]:
            if os.path.exists(os.path.join(args.output_dir, args.infoseek_path, "images", f"{sample.get('image_id')}{ext}")):
                cached_paths.append(os.path.join(args.output_dir, args.infoseek_path, "images", f"{sample.get('image_id')}{ext}"))
        mc_choices = []
        rationales = []

    elif dataset_name == "oven":
        qid = sample.get("data_id") or str(uuid.uuid4())
        image_id = sample.get("image_id")
        question = sample.get("question")
        answers = [sample.get("entity_text")]
        image_path = [sample.get("image")] or [sample.get("image_path")]
        cached_paths = [os.path.join(args.output_dir, args.infoseek_path, "images", sample.get("image_id"))]
        mc_choices = [sample.get("entity_id"), sample.get("entity_text")]
        rationales = []

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported yet.")

    # Determine if the sample needs retrieval or not
    needs_resp = call_openai(
        args,
        client,
        SYS_NEEDS_RETRIEVAL,
        USER_NEEDS_RETRIEVAL,
        question,
        image_path=image_path,
        incontext_samples=incontext_samples_dict["needs_incontext_samples"],
        max_tokens=256,
        json_mode=True,
    )
    try:
        needs_json = json.loads(
            needs_resp.lower().replace("```json", "").replace("```", "")
        )
        needs_retrieval = bool(needs_json["needs_retrieval"])
        needs_retrieval_val = needs_json.get("needs_retrieval", None)
        needs_retrieval = needs_retrieval_val if isinstance(needs_retrieval_val, bool) else str(needs_retrieval_val).lower().strip() == "true"
        needs_rationale = needs_json["reasoning"]
    except Exception:
        needs_retrieval = "true" in needs_resp.lower()
        needs_rationale = ""
        print("DEBUG: needs_json - ", needs_retrieval)
        breakpoint()

    # If needs_retrieval == True, qdecomp and answer_draft should be considered
    if needs_retrieval:
        # decompose the questions and determine whether each subquestion needs retrieval or not
        qdecomp_resp = call_openai(
            args,
            client,
            SYS_QUESTION_DECOMPOSE,
            USER_QUESTION_DECOMPOSE,
            question,
            image_path=image_path,
            incontext_samples=incontext_samples_dict["qdecomp_incontext_samples"],
            max_tokens=256,
            json_mode=True,
        )
        try:
            qdecomp_json = json.loads(
                qdecomp_resp.lower().replace("```json", "").replace("```", "")
            )
            qdecomp_isdecomposed = qdecomp_json["is_decomposed"]
            qdecomp_retrieval = [True if need == 'true' or need == True else False for need in qdecomp_json["needs_retrieval"]]
            qdecomp_rationale = qdecomp_json["reasoning"]
            qdecomp_subquestions = qdecomp_json["subquestions"]
        except Exception:
            print("DEBUG: qdecomp_json - ", qdecomp_retrieval)
            breakpoint()

        # Construct incontext_samples for answer_draft since we need to include the qdecomp_json outputs as well
        user_draft_answer = USER_DRAFT_ANSWER_FOR_RETRIEVAL
        answer_incontext_samples = []
        for i, (qdecomp_sample, answer_sample) in enumerate(zip(incontext_samples_dict["qdecomp_incontext_samples"], incontext_samples_dict["answer_incontext_samples"])):
            qdecomp_dict = extract_dict_like_str(qdecomp_sample["content"][0]["text"].strip())
            answer_user_msg_question = " ".join(answer_sample["content"][0]["text"].strip().split("\n")[:2]) + "\nGround-truth Answer: " + answer_sample["content"][0]["text"].strip().split("\n")[2]
            answer_user_msg_answer = " ".join(answer_sample["content"][0]["text"].strip().split("\n")[3:])
            answer_user_msg = answer_user_msg_question + \
                                "\nReasoning: " + qdecomp_dict['reasoning'] + \
                                "\nSubquestions: " + str(qdecomp_dict['subquestions']) + \
                                "\nneeds_retrieval: " + str(qdecomp_dict['needs_retrieval']) + "\n" + \
                                answer_user_msg_answer
            answer_incontext_samples.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": answer_user_msg},
                            answer_sample["content"][-1]  # image
                        ]
                    }
                )
        answer_draft_question = question + "\nGround-truth Answer: " + str(answers)
        answer_draft_question += f"\nReasoning: {qdecomp_rationale}\n" + f"Subquestions: {qdecomp_subquestions}\n" + f"needs_retrieval: {str(qdecomp_retrieval)}"

    else:
        qdecomp_retrieval = []
        qdecomp_subquestions = []
        qdecomp_isdecomposed = False
        qdecomp_rationale = ""
        answer_draft_question = question
        user_draft_answer = USER_DRAFT_ANSWER
        # remove <search> from the incontext_samples
        answer_incontext_samples = []
        for sample in incontext_samples_dict["answer_incontext_samples"]:
            sample["content"][0]["text"] = sample["content"][0]["text"].replace("<search> ", "")
            answer_incontext_samples.append(sample)

    # construct answer drafts from the LLM (single-shot; drop the sample if it violates constraints)
    answer_resp = call_openai(
        args,
        client,
        SYS_DRAFT_ANSWER,
        user_draft_answer,
        answer_draft_question,
        image_path=image_path,
        incontext_samples=answer_incontext_samples,
        max_tokens=256,
        json_mode=True,
    )

    # Parse JSON safely without lowercasing the entire payload (preserves case in the answer)
    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL)
        return s

    try:
        cleaned = _strip_code_fences(answer_resp)
        answer_json = json.loads(cleaned)
        answer_draft = str(answer_json.get("answer", "")).strip()
    except Exception as e:
        logging.warning("Failed to parse answer JSON for id=%s: %s. Dropping sample.", qid, e)
        return None

    expected_search = sum(qdecomp_retrieval)
    found_search = answer_draft.count("<search>")
    if expected_search != found_search:
        logging.info(
            "Dropping sample id=%s due to <search> token mismatch (expected=%d, found=%d).",
            qid, expected_search, found_search
        )
        return None

    # we only get here when counts match; compute num_search_toks for downstream logic
    num_search_toks = found_search

    # assign qtype to each <search> token
    query_types, queries = [], []
    qtype_reasoning = ""
    if needs_retrieval and num_search_toks > 0:  # both needs_retrieval and the num_search_toks should be unanimously true
         # Build a function schema whose array lengths are fixed
        router_schema = {
            "name": "route_queries",
            "description": "Choose retrieval modality and query for each <search> token",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "items": {"type": "string"},
                    },
                    "query_types": {
                        "type": "array",
                        "minItems": num_search_toks,   # fixed length
                        "maxItems": num_search_toks,
                        "items": {"type": "string",
                                "enum": ["REGION","IMAGE","TEXT"]}
                    },
                    "queries": {
                        "type": "array",
                        "minItems": num_search_toks,
                        "maxItems": num_search_toks,
                        "items": {"type": "string"}
                    }
                },
                "required": ["reasoning", "query_types","queries"]
            }
        }
        router_tool = {
            "type": "function",
            "function": router_schema   # the dict that has name/description/parameters
        }

        # assign qtype (modality type) to each <search> token
        modality_resp = call_openai(
            args,
            client,
            SYS_QUERY_ROUTER,
            USER_QUERY_ROUTER,
            question + "\n" + str({'answer': answer_draft, 'num_search_toks': num_search_toks}),
            image_path=image_path,
            incontext_samples=incontext_samples_dict["modality_incontext_samples"],
            max_tokens=256,
            json_mode=True,
            json_schema=router_tool,
        )
        try:
            if not isinstance(modality_resp, dict) and isinstance(modality_resp, str):
                modality_json = json.loads(modality_resp.lower().replace("```json", "").replace("```", ""))
            elif isinstance(modality_resp, dict):
                modality_json = modality_resp

        except Exception:
            print("DEBUG: modality_resp - ", modality_resp)
            breakpoint()
        qtype_reasoning = modality_json["reasoning"]
        for qtype, query_txt in zip(modality_json['query_types'], modality_json['queries']):
            if qtype.upper().strip() not in ["TEXT", "REGION", "IMAGE"]:
                try:
                    raise ValueError(f"Unknown query type: {qtype}")
                except ValueError as e:
                    breakpoint()
            query_types.append(qtype.upper().strip())
            queries.append(query_txt)

    record = {
        "id"             : f"{dataset_name}-{qid}",
        "dataset"        : dataset_name,
        "image_path"     : cached_paths,
        "question"       : question,
        "needs_retrieval": needs_retrieval if needs_retrieval and sum(qdecomp_retrieval) > 0 else False,
        "needs_rationale": needs_rationale,
        "answer_draft"   : answer_draft,
        "num_search_toks": num_search_toks,
        "is_decomposed" : qdecomp_isdecomposed,
        "decomposed_questions": qdecomp_subquestions if len(qdecomp_subquestions) > 0 else [],
        "decomposed_rationale": qdecomp_rationale if len(qdecomp_rationale) > 0 else "",
        "decomposed_needs_retrieval": qdecomp_retrieval if len(qdecomp_retrieval) > 0 else [],
        "mc_choices" : mc_choices if len(mc_choices) > 0 else [],
        "ground_truth" : answers if len(answers) > 0 else [],
        "rationales" : rationales if rationales else "",
        "search_calls"   : []
    }

    if dataset_name == "webqa":
        record["image_gold_retrieved"] = image_gold_retrieved
        record["text_gold_retrieved"] = text_gold_retrieved

    # the MASK extraction pipeline logic
    # save the MASKs in the record and the visualized mask paths + the bounding box
    qtype = ""
    decomp_mismatch_cnt = 0
    if needs_retrieval and num_search_toks > 0:
        try:      # NOTE: Currently, keeping track of `decomp_mismatch_cnt` to debug the issue of mismatch between sum(qdecomp_retrieval) and num_search_toks
            assert sum(qdecomp_retrieval) == len(query_types) == len(queries), "Number of <search> tokens in the answer draft does not match the number of decomposed questions labeled as need_retrieval == True"
        except AssertionError as e:
            decomp_mismatch_cnt += 1
            print("**** ERROR **** there are no <search> tokens in the answer draft but the qdecomp_retrieval is true - regenerating answer...")
            print("DEBUG: needs_retrieval - ", needs_retrieval)
            print("DEBUG: qdecomp_retrieval - ", qdecomp_retrieval)
            print("DEBUG: answer_draft - ", answer_draft)

        pattern = re.compile(r"<search>")
        start_pos = [m.start() for m in re.finditer(pattern, answer_draft.lower())]
        # generate gold masks in case the <search> token's qtype == "REGION"
        for search_tok_idx, (qtype, query_txt) in enumerate(zip(query_types, queries)):
            call_entry = {}
            rle_dict = []
            masks_path = []
            if qtype in ["REGION", "IMAGE"]:
                region_cache_dir = os.path.join(out_dir, "images", f"{dataset_name}-{qid}")
                os.makedirs(region_cache_dir, exist_ok=True)
                if qtype == "REGION":
                    rle_dict, paths, phrases = generate_gold_mask(args, image_path, question, [query_txt], region_cache_dir)
                    if len(rle_dict) == 0 and len(paths) == 0 and len(phrases) == 0:
                        qtype = "IMAGE"  # If no bbox or mask is found, fallback to IMAGE search
                    masks_path.extend(paths)
            elif qtype == "IMAGE":
                pass
            elif qtype == "TEXT":
                pass
            else:
                try:
                    raise ValueError(f"Unknown query type: {qtype}")
                except ValueError as e:
                    breakpoint()

            try:
                call_entry = {
                    "position" : start_pos[search_tok_idx] if len(start_pos) >= search_tok_idx + 1 else -1,  # trigger search at the end of the generated answer if needed
                    "qtype" : qtype,
                    "qtype_reasoning": qtype_reasoning,
                    "query_text": query_txt,
                    "mask": rle_dict,
                    "mask_path": masks_path,
                }
            except IndexError as e:
                print(e)
                print("DEBUG: start_pos - ", start_pos)
                print("DEBUG: search_tok_idx - ", search_tok_idx)
                breakpoint()

            record["search_calls"].append(call_entry)
            record["decomp_mismatch_cnt"] = decomp_mismatch_cnt

    if args.debug:
        print(f"------------[ qid: {qid} ]------------")
        print(f"qtype: {qtype}")
        if "REGION" in query_types:
            print(f">> paths: {paths}")
            print(f">> phrases: {phrases}")
        print(f"question: {question}")
        print(f"needs_retrieval: {needs_retrieval}")
        print(f"decomposed questions: {qdecomp_subquestions}")
        print(f"answer_draft: {answer_draft}")
        print(f"answers: {answers}")

    return record


def interleave_search_and_query(
    output_path: str,
    processed_instances: List[Dict[str, Any]],
    instances: List[Dict[str, Any]],
    question_type_stat: List[Dict[str, Any]],
    use_llm_for_span_extraction: bool = False):
    """
    Interleave search and query instances in a list of instances, and (NEW)
    extract a REGION query text span that precedes each <search> token whose qtype == "REGION".

    - If any mask.label (from the corresponding search_call) appears before the <search> token,
      record that exact substring as the region_query_span.
    - Otherwise, record the entire preceding sentence fragment as region_query_preceding_span,
      and optionally call a small LLM to extract the best short span to highlight.

    The extracted metadata is added to each REGION search_call:
      * region_query_span (optional)
      * region_query_span_source in {"label","llm"} (optional)
      * region_query_span_offsets = [start_rtxt, end_rtxt] (optional)
      * region_query_label, region_query_label_mask_index (when label matched)
      * region_query_preceding_span, region_query_preceding_span_offsets (always when no label match)

    Args:
        output_path: JSONL file path for logging the 'processed_instances'
        processed_instances: <search> payload </search> interleaved instances.
        instance (List[Dict[str, Any]]): A list of <search> token-only instances.
        question_type_stat (List[Dict[str, Any]]): A list of dictionaries containing the question type per question.
    """

    # NOTE: Run `calculate_dataset_stats.py` to get the question type statistics first from the `{name}_search_annotated_v1.jsonl` files
    assert len(instances) == len(question_type_stat), (
        f"The length of instance and question_type_stat must be equal. "
        f"{len(instances)} != {len(question_type_stat)}"
    )

    # helper functions
    def _find_case_insensitive(haystack: str, needle: str) -> Optional[tuple]:
        """Return (start,end) of case-insensitive match of `needle` in `haystack`, else None."""
        if not needle:
            return None
        m = re.search(re.escape(needle), haystack, flags=re.IGNORECASE)
        if not m:
            return None
        return (m.start(), m.end())

    def _call_llm_extract_span(preceding_span: str, label_hint: Optional[str]) -> Optional[str]:
        """
        Optionally call an LLM (if enabled) to pick best substring of `preceding_span`.
        Returns the chosen substring (must be exact substring of preceding_span) or None.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not (api_key and preceding_span.strip()):
            return None

        try:
            from openai import OpenAI as _OpenAI
            client = _OpenAI(api_key=api_key)
            model_name = os.environ.get("REGION_SPAN_LLM_MODEL", "gpt-4.1-mini")

            system_prompt = (
                "You extract a short text span from a provided paragraph. "
                "Return strictly JSON: {\"span\": \"...\"}. "
                "The span MUST be an exact substring of the paragraph and no more than 8 words. "
                "Prefer a concrete noun phrase that best refers to the region described by the hint."
            )
            user_prompt = (
                "Paragraph (context before <search>):\n"
                f"{preceding_span}\n\n"
                f"Hint about the region (may be empty): {label_hint or ''}\n\n"
                "Return JSON with the best exact substring to highlight."
            )
            params = dict(
                model=model_name,
                temperature=0.0,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
                max_tokens=64
            )
            resp = client.chat.completions.create(**params)
            content = resp.choices[0].message.content.strip()
            # basic robust parse
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL)
            data = json.loads(content)
            span = (data.get("span") or "").strip()
            if span and span in preceding_span:
                return span
        except Exception as _e:
            pass  # fallback on any LLM errors
        return None

    # interleave <search> </search> with query text + region query span for qtype == "REGION"
    print(f"Starting from : {len(processed_instances)}")
    pattern = re.compile(r"<search>")

    with open(output_path, "a") as jsonl_writer:
        pbar_desc = f"[{instances[0]['dataset']}] Interleaving search and query for {len(instances)} instances"
        for i, (instance, qtype_dict) in tqdm(
                enumerate(zip(instances[len(processed_instances):], question_type_stat[len(processed_instances):])),
                desc=pbar_desc
        ):
            question_type = qtype_dict["label"].lower().strip()
            answer_draft = instance["answer_draft"]
            if any(k in question_type for k in ("factoid", "kb", "external", "entity")):
                # interleave the <search> </search> with query text
                question = instance["question"]
                answer_draft = instance["answer_draft"]
                start_pos_list = sorted([m.start() for m in re.finditer(pattern, answer_draft.lower())])
                search_calls = instance["search_calls"]
                search_calls_start_pos_list = sorted([int(pos['position']) for pos in search_calls])

                try:
                    assert all(p == q for p, q in zip(start_pos_list, search_calls_start_pos_list)), f"List Mismatch - start_pos_list: {start_pos_list} | search_calls_start_pos_list: {search_calls_start_pos_list}"
                except AssertionError as e:
                    print(e)
                    breakpoint()

                # perform region span extraction before mutating the answer_draft
                pos2sc = {}
                for sc in search_calls:
                    try:
                        pos2sc[int(sc["position"])] = sc
                    except Exception:
                        continue

                for pos in start_pos_list:
                    sc = pos2sc.get(pos)
                    if not sc or sc.get("qtype", "").upper() != "REGION": # we need region query text span only for REGION qtypes
                        continue

                    # 1. full preceding text up to <search> position
                    preceding_span = answer_draft[:pos].strip(" \t")

                    # 2. exact label match first
                    label_matched = False
                    label_match_offsets = None
                    label_text_used = None
                    label_mask_idx_used = None
                    masks = sc.get("mask") or []
                    for mi, m in enumerate(masks):
                        label = (m.get("label") or "").strip()
                        if not label:
                            continue
                        # try locating within preceding span
                        region_txt = _find_case_insensitive(preceding_span, label)
                        if region_txt:
                            start_rtxt, end_rtxt = region_txt
                            sc["region_query_span"] = answer_draft[start_rtxt:end_rtxt]
                            sc["region_query_span_source"] = "label"
                            sc["region_query_span_offsets"] = [start_rtxt, end_rtxt]
                            sc["region_query_label"] = label
                            sc["region_query_label_mask_index"] = mi
                            label_matched = True
                            label_match_offsets = (start_rtxt, end_rtxt)
                            label_text_used = label
                            label_mask_idx_used = mi
                            break

                    # 3. if no label match, store preceding span; optionally LLM to pick a short span
                    if not label_matched:
                        # always store the full preceding span + offsets
                        sc["region_query_preceding_span"] = preceding_span
                        sc["region_query_preceding_span_offsets"] = [0, pos]

                        # optional LLM best-substring extraction
                        if use_llm_for_span_extraction:
                            span_llm = _call_llm_extract_span(preceding_span, label_hint=(masks[0].get("label") if masks else None))
                            if span_llm:
                                region_txt = _find_case_insensitive(preceding_span, span_llm)
                                if region_txt:
                                    start_rtxt, end_rtxt = region_txt
                                    sc["region_query_span"] = answer_draft[start_rtxt:end_rtxt]  # exact substring
                                    sc["region_query_span_source"] = "llm"
                                    sc["region_query_span_offsets"] = [start_rtxt, end_rtxt]

                # alignment check and insert <search> payload </search> & interactively fix erroneous instances in case there are errors
                try:
                    assert len(search_calls) == len(start_pos_list), f"The length of search_calls and number of <search> tokens must be equal. {len(search_calls)} != {len(start_pos_list)}"
                except AssertionError as e:
                    # 1. Rewrite
                    # 2. Get the new start_pos_list
                    # 3. assign / drop each search call to its new position according to start_pos_list
                    print(f"\n\n----MANUAL Filtering (instance number: {i} | dataset_name: {instance['dataset']})----\n")
                    print(f"(Original) Question: {question}")
                    print(f"(Original) Answer Draft: {answer_draft}")
                    print(f"(Search Pos.) <search> tokens starting indices : ", start_pos_list)
                    print("==============================")
                    for i, sc in enumerate(search_calls):
                        print(f"> ({i}) Search Call: {sc}")
                    print("==============================")

                    if instance['dataset'] == 'oven':
                        user_prompt = input("Press 'Enter' to continue or type 'fix' to revise...")

                    if user_prompt == "fix" or instance['dataset'] != 'oven':
                        answer_draft = input("* New Answer Draft >> ")
                        start_pos_list = sorted([m.start() for m in re.finditer(pattern, answer_draft.lower())])
                        print("* (new) <search> tokens starting indices : ", start_pos_list)
                        new_search_calls = []
                        for i, sc in enumerate(search_calls):
                            sc['position'] = int(input(f"New pos index for {i}-th <search> token (qtype: {sc['qtype']} | query_text: {sc['query_text']}) (input '-1' to drop the search call): "))
                            if sc['position'] != -1:
                                new_search_calls.append(sc)
                        search_calls = new_search_calls
                        print(f"** New Answer Draft: {answer_draft}\n")
                        print(f"** New Search Calls: {search_calls}\n")
                        input("To Proceed, Press 'Enter'...")
                    else: # just drop the last search call
                        search_calls = search_calls[:-1]

                for sc in sorted(search_calls, key=lambda c: c["position"], reverse=True):  # reverse order to prevent pos index from being off
                    if sc["qtype"] == "REGION":
                        payload = "<region>"
                    elif sc["qtype"] == "IMAGE":
                        payload = "<img>"
                    elif sc["qtype"] == "TEXT":
                        payload = sc["query_text"].lower().strip()
                    resp = f"<search> {payload} </search>"
                    start_pos = sc["position"]
                    end_pos = sc["position"] + len("<search>")
                    answer_draft = answer_draft[:start_pos] + resp + answer_draft[end_pos:]
                instance["search_calls"] = search_calls
            else:
                # remove <search> </search> from the answer draft
                answer_draft = " ".join(answer_draft.replace(" <search>", "").replace("<search> ", "").split())
                instance["needs_retrieval"] = False
                instance["num_search_toks"] = 0
                instance["decomposed_needs_retrieval"] = [False for _ in range(len(instance["decomposed_questions"]))]
                instance["search_calls"] = []

            instance["answer"] = answer_draft
            processed_instances.append(instance)

            # append the fixed instance line by line in here
            jsonl_writer.write(json.dumps(instance, ensure_ascii=False, default=str) + "\n")

    return processed_instances


def interleave_retrieved(
    output_path: str,
    processed_instances: List[Dict[str, Any]],
    instances: List[Dict[str, Any]],
    search_pipeline: Union[LocalSearchPipeline, UnifiedSearchPipeline],
    ktype: str = "title",  # "title", "entity", "all" (include all the retrieved attributes)
    region_type: str = "mask", # "mask", "bbox"
    k: int = 5):
    """
    Interleave the top-'k' retrieved instances into the model output (between the <information> and </information> tokens)

    Args:
        ktype - knowledge type; "title", "content", "entity", "all" (include all the retrieved attributes)
    """

    print(f"Starting from : {len(processed_instances)}")
    pattern = re.compile(r"<search>")

    with open(output_path, "a") as jsonl_writer:
        for i, instance in tqdm(enumerate(instances[len(processed_instances):]), desc=f"[{instances[0]['dataset']}] Interleaving retrieved results for {len(instances)} instances"):
            ds_name = instance["dataset"]
            question = instance["question"]
            image = instance["image_path"]
            answer_draft = instance["answer_draft"]
            search_interleaved_answer = instance["answer"]
            start_pos_list = sorted([m.start() for m in re.finditer(pattern, search_interleaved_answer.lower())])
            search_calls = instance["search_calls"]
            if start_pos_list:
                search_queries = extract_search_queries(search_interleaved_answer)  # extract search queries in between <search> </search> tokens
                search_queries = [q.lower().strip() for q in search_queries]
                retrieved_items = []
                for qidx, q in enumerate(search_queries):  # construct different input_query per search query type
                    if q == "<img>":
                        input_query = image
                    elif q == "<region>":
                        region_dir_path = os.path.dirname(search_calls[qidx]["mask_path"][0][region_type])
                        input_query = glob.glob(os.path.join(region_dir_path, "cropped_*.png"))[0]  # NOTE: there could be multiple queries
                    else:  # text query
                        input_query = q

                    r = search_pipeline(input_query, k=k)  # send the search queries to the search pipeline
                    retrieved_items.append(r)  # 'r' is list of lists (inner list == # of search_queries)

                    # NOTE: For Crag-mm search api, some 'r' is list of list while some 'r' are list of list of list
                    # (which causes inconsistency in the number of retrieved items)

                instance["retrieved"] = {"k": k, "item": retrieved_items}
                # interleave the retrieved text (e.g., title, summary, plot) with the <information> </information> tokens and save
                filtered_retrieved_items = []
                for item in retrieved_items:
                    if ktype == "all":
                        filtered_r = [f"{str(r['title'])}: {str(r['content'])}" for r in item]
                    elif ktype == "entity":
                        try:
                            filtered_r = [str(r["entities"]) for r in item]
                        except Exception as e:
                            print(e)
                            breakpoint()
                    elif ktype == "title":
                        filtered_r = [str(r["title"]) for r in item]
                    elif ktype == "content":
                        filtered_r = [str(r["content"]) for r in item]
                    filtered_retrieved_items.append(filtered_r)

                # construct the <information> </information> tokens interleaved with retrieved items
                final_text, info_offsets = append_retrieved_items(search_interleaved_answer, filtered_retrieved_items)

                instance["answer"] = final_text
                instance["info_offsets"] = info_offsets
                processed_instances.append(instance)

            # append the fixed instance line by line in here
            jsonl_writer.write(json.dumps(instance, ensure_ascii=False, default=str) + "\n")


def parse_args(args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser(
        description="VQA Dataset Construction"
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

    parser.add_argument("--llm_model", default="gpt-4o-mini", type=str)
    parser.add_argument("--use_llm_as_judge", action="store_true")

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="../results/",
    )
    parser.add_argument("--save_every_n", default=100, type=int)

    parser.add_argument("--exp_name", default="llm-seg", type=str)

    parser.add_argument(
        "--image", required=False, help="Input image path (for single image processing)"
    )
    parser.add_argument(
        "--output_dir",
        default="/home/jeonghkim/projects/region-mm-retrieve/data/",
        help="Dataset output directory",
    )
    parser.add_argument("--dataset_name", default="region_rag_vqa", type=str)
    parser.add_argument("--question_type_stat_path", default="question_types.json", type=str)

    parser.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--sam_checkpoint",
        default="/home/jeonghkim/weights/sam/sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--sam_model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"]
    )
    parser.add_argument(
        "--dino_config",
        default="/home/jeonghkim/weights/groundingdino/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--dino_checkpoint",
        default="/home/jeonghkim/weights/groundingdino/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--add_incontext_samples", action="store_true")
    parser.add_argument("--num_incontext_samples", default=8, type=int)
    parser.add_argument("--needs_incontext_cache_path", default="needs_incontext_sample.jsonl", type=str)
    parser.add_argument("--qdecomp_incontext_cache_path", default="qdecomp_incontext_sample.jsonl", type=str)
    parser.add_argument("--answer_incontext_cache_path", default="answer_incontext_sample.jsonl", type=str)
    parser.add_argument("--modality_incontext_cache_path", default="modality_incontext_sample.jsonl", type=str)
    parser.add_argument("--infoseek_path", default="infoseek", type=str)
    parser.add_argument("--infoseek_limit_num", default=20000, type=int)
    parser.add_argument("--mask_alpha", default=0.4, type=float)
    parser.add_argument("--limit_num", default=-1, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--version_num", default=0, type=int)

    parser.add_argument("--topk", default=5, type=int)

    # webqa args
    parser.add_argument("--webqa_tsv_path", default="/home/jeonghkim/projects/region-mm-retrieve/data/webqa/WebQA_imgs/WebQA_imgs_7z_chunks/imgs.tsv", type=str)
    parser.add_argument("--webqa_lineidx_path", default="/home/jeonghkim/projects/region-mm-retrieve/data/webqa/imgs.lineidx", type=str)
    parser.add_argument("--webqa_img_cache_dir", default="/home/jeonghkim/projects/region-mm-retrieve/data/webqa/WebQA_imgs", type=str)

    # local search api flags
    parser.add_argument("--retriever", default="local", type=str, help="cragmm search api or local faiss index (i.e., local search api)")
    parser.add_argument("--search_api_url", default="http://127.0.0.1:7000", type=str)
    parser.add_argument("--search_api_collection", default="wiki10k_index", type=str)
    parser.add_argument("--search_api_index_name", default=None, type=str)  # optional override
    parser.add_argument("--search_api_pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--search_api_text_model", default="sentence-transformers/all-mpnet-base-v2", type=str)
    parser.add_argument("--search_api_image_model", default="facebook/dinov3-vitb16-pretrain-lvd1689m", type=str)

    parser.add_argument("--use_llm_for_span_extraction", action="store_true")

    args = parser.parse_string(config_str) if config_str else parser.parse_args(args)
    return args, parser


def main(args: list[str] = None, config_str: str = None):
    args, parser = parse_args(args, config_str)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # construct in-context samples
    incontext_samples_dict = {}
    incontext_ds = load_dataset("crag-mm-2025/crag-mm-single-turn-public", split="validation")
    incontext_out_dir = os.path.join(args.output_dir, args.dataset_name, "cragmm")
    print(">> Constructing in-context samples using cragmm dataset")
    needs_incontext_samples = build_in_context_cache(
            args.num_incontext_samples,
            incontext_ds,
            "cragmm",
            os.path.join(incontext_out_dir, args.needs_incontext_cache_path),
            prompt_type="needs_retrieval",
            resize_img=True,
        )
    qdecomp_incontext_samples = build_in_context_cache(
            args.num_incontext_samples,
            incontext_ds,
            "cragmm",
            os.path.join(incontext_out_dir, args.qdecomp_incontext_cache_path),
            prompt_type="qdecomp",
            resize_img=True,
        )
    answer_incontext_samples = build_in_context_cache(
            args.num_incontext_samples,
            incontext_ds,
            "cragmm",
            os.path.join(incontext_out_dir, args.answer_incontext_cache_path),
            prompt_type="answer",
            resize_img=True,
        )
    modality_incontext_samples = build_in_context_cache(
            args.num_incontext_samples,
            incontext_ds,
            "cragmm",
            os.path.join(incontext_out_dir, args.modality_incontext_cache_path),
            prompt_type="modality",
            resize_img=True,
        )
    incontext_samples_dict["needs_incontext_samples"] = needs_incontext_samples
    incontext_samples_dict["qdecomp_incontext_samples"] = qdecomp_incontext_samples
    incontext_samples_dict["answer_incontext_samples"] = answer_incontext_samples
    incontext_samples_dict["modality_incontext_samples"] = modality_incontext_samples

    if args.question_type_stat_path is not None:
        with open(os.path.join(args.output_dir, args.dataset_name, args.question_type_stat_path), "r") as fp:
            question_type_stat = json.load(fp)
            print(f">> Loaded question type stats from {args.question_type_stat_path}")

    if args.retriever is None or args.retriever == "local":
        search_pipeline = LocalSearchPipeline(
            base_url=args.search_api_url,
            collection=args.search_api_collection,
            index_name=args.search_api_index_name,
            pooling=args.search_api_pooling,
            text_model=args.search_api_text_model,
            image_model=args.search_api_image_model,
        )
    elif retriever == "cragmm":
        os.environ["HTTPS_PROXY"] = "http://fwdproxy:8080"
        os.environ["NO_PROXY"] = ",".join([
            ".fbcdn.net", ".facebook.com", ".thefacebook.com", ".tfbnw.net",
            ".fb.com", ".fburl.com", ".facebook.net", ".sb.fbsbx.com", "localhost"
        ])
        search_pipeline = UnifiedSearchPipeline(
            image_model_name="openai/clip-vit-large-patch14-336",
            image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
            text_model_name="BAAI/bge-large-en-v1.5",
            web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
        )

    # generate data samples for the datasets
    for ds_name, cfg in DATASETS.items():

        print(f"\n### Processing {ds_name} ###")
        ds = []
        if ds_name == "infoseek":  # infoseek_train.jsonl causes error if loaded with `datasets.load_dataset`
            with open(os.path.join(args.output_dir, args.infoseek_path, "infoseek_train.jsonl"), "r") as fp_reader:
                for idx, line in enumerate(tqdm(fp_reader, desc="Loading infoseek_train.jsonl")):
                    if idx >= args.infoseek_limit_num:
                        break
                    line = line.strip()
                    rec = json.loads(line)
                    rec["image"] = load_image_by_id(Path(os.path.join(args.output_dir, args.infoseek_path, "images")), rec["image_id"])
                    ds.append(rec)
        elif ds_name == "oven":  # TODO: Later, oven_query_train.jsonl maybe used alongside `oven_entity_train.jsonl`
            with open(os.path.join(args.output_dir, args.infoseek_path, "oven_entity_train.jsonl"), "r") as fp_reader:
                for idx, line in enumerate(tqdm(fp_reader, desc="Loading oven_entity_train.jsonl")):
                    if idx >= args.infoseek_limit_num:
                        break
                    line = line.strip()
                    rec = json.loads(line)
                    if rec["image_id"].lower().strip() in ["oven_01830596"]: # NOTE: skip erroneous samples
                        continue
                    rec["image"] = load_image_by_id(Path(os.path.join(args.output_dir, args.infoseek_path, "images")), rec["image_id"])
                    ds.append(rec)
        else:
            ds = load_dataset(cfg["hf_name"], split=cfg["split"], trust_remote_code=True)

        out_dir = os.path.join(args.output_dir, args.dataset_name, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, f"{ds_name}_search_annotated_v{str(args.version_num)}.jsonl")
        buffer = []
        if os.path.exists(output_file):  # load from cache
            print(f">> Loading from cache: {output_file}")
            with open(output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    buffer.append(json.loads(line))
            print(f">> Loaded from cache: {output_file} | len: {len(buffer)}")

        # construct the dataset
        with open(output_file, "a") as f:
            for idx, sample in enumerate(tqdm(ds, total=len(ds))):
                if len(buffer) > idx:
                    continue
                if len(buffer) >= args.limit_num:
                    break
                rec = process_sample(args, client, sample, ds_name, out_dir, incontext_samples_dict)
                if rec is None:
                    # dropped due to invalid JSON or <search> token mismatch
                    print(f">> Skipping invalid JSON or <search> token mismatch: ds_name: {ds_name} | idx: {idx} | len: {len(buffer)}")
                    continue
                buffer.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        # based on the constructed data and question type stats, interleave `<search> search_query </search>`` tokens
        # if the question type is not in ["Factoid / KB Questions", "Multi-hop External Knowledge Reasoning", "Fine-grained Entity-Identification"]
        # then we simply remove the <search> tokens and revert back to the original response
        interleaved_buffer = []
        interleaved_output_file = os.path.join(out_dir, f"{ds_name}_search_interleaved_v{str(args.version_num)}.jsonl")
        if os.path.exists(interleaved_output_file):
            print(f">> Loading from cache: {interleaved_output_file}")
            with open(interleaved_output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    interleaved_buffer.append(json.loads(line))
            print(f">> Loaded from cache: {interleaved_output_file} | len: {len(interleaved_buffer)}")
        if len(interleaved_buffer) < len(buffer):
            interleaved_buffer = interleave_search_and_query(
                                    interleaved_output_file,
                                    interleaved_buffer,
                                    buffer,
                                    question_type_stat[ds_name],
                                    args.use_llm_for_span_extraction
                                )

        # after building the `interleaved_buffer`, interleave the search API-retrieved results between the <information> </information> tokens
        # <information> </information> tokens should immediately follow the <search> </search> tokens
        # based on each of the search calls, retrieve using cragmm search api and then interleave with <information> </information> tokens
        retrieved_buffer = []
        # retrieved_output_file = os.path.join(out_dir, f"{ds_name}_retrieve_k_{args.topk}_interleaved_v{str(args.version_num)}.jsonl")
        retrieved_output_file = os.path.join(out_dir, f"{ds_name}_retrieve_k_{args.topk}_interleaved_v{str(args.version_num + 1)}.jsonl")
        if os.path.exists(retrieved_output_file):
            print(f">> Loading from cache: {retrieved_output_file}")
            with open(retrieved_output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    retrieved_buffer.append(json.loads(line))
            print(f">> Loaded from cache: {retrieved_output_file} | len: {len(retrieved_buffer)}")
        if len(retrieved_buffer) < len(interleaved_buffer):
            retrieved_buffer = interleave_retrieved(
                                    retrieved_output_file,
                                    retrieved_buffer,
                                    interleaved_buffer,
                                    search_pipeline,
                                    ktype="title",
                                    k=args.topk,
                                    )


if __name__ == "__main__":
    main()
