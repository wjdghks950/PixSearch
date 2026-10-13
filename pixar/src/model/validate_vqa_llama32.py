# validate_vqa_llama32.py
# Evaluate meta-llama/Llama-3.2-11B-Vision on your VQA datasets in 3 modes:
#   1) raw     : plain VQA (no search)
#   2) inject  : prepend/append retrieved results (image/text/both)
#   3) search  : 0-shot <search> ... </search> prompting with on-the-fly retrieval injection
#
# Usage example (local wiki-like search, CRAG-MM val split):
#   python validate_vqa_llama32.py \
#     --dataset_dir /home/jeonghkim/projects/pixar/data/region_rag_vqa \
#     --val_dataset "cragmm|validation" \
#     --setting raw \
#     --predictions_path /home/jeonghkim/projects/pixar/pixar/src/results/llama32_raw
#
#   python validate_vqa_llama32.py \
#     --dataset_dir /home/jeonghkim/projects/pixar/data/region_rag_vqa \
#     --val_dataset "cragmm|validation" \
#     --setting inject --inject_where prepend --inject_sources img,text \
#     --search_api local --retrieve_type title \
#     --predictions_path /home/jeonghkim/projects/pixar/pixar/src/results/llama32_inject
#
#   python validate_vqa_llama32.py \
#     --dataset_dir /home/jeonghkim/projects/pixar/data/region_rag_vqa \
#     --val_dataset "cragmm|validation" \
#     --setting search --search_api local --retrieve_type title \
#     --predictions_path /home/jeonghkim/projects/pixar/pixar/src/results/llama32_search \
#     --limit_batches 200

import os
import re
import sys
import json
import time
import cv2
import numpy as np
import tqdm
import openai
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# ==== your utilities ====
# dataset + evaluators
from utils.vqa_dataset import TextVQADataset, GQADataset, POPEDataset, CragMMDataset, InfoseekDataset, OvenDataset
from utils.m4c_evaluator import TextVQAAccuracyEvaluator

# search helpers / pipelines (reused to match your validate_search)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.local_search_api import LocalSearchPipeline

sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))
from src.evaluation.rle_dict import get_mask_rle_dicts
from src.evaluation.llm_as_judge_eval import evaluate_answer


# ==== special token strings used in prompting ====
SEARCH_START_TOKEN = "<search>"
SEARCH_END_TOKEN   = "</search>"
INFO_START_TOKEN   = "<information>"
INFO_END_TOKEN     = "</information>"
IMG_PAYLOAD_TOKEN  = "<img>"
REGION_TOKEN       = "<region>"

# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser("Llama-3.2-Vision VQA validator (raw / inject / search)")
    p.add_argument("--model_id", default="meta-llama/Llama-3.2-11B-Vision-Instruct", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])

    p.add_argument("--dataset_dir", required=True, type=str)
    p.add_argument("--val_dataset", default="cragmm|validation", type=str)
    p.add_argument("--workers", default=0, type=int)
    p.add_argument("--val_batch_size", default=1, type=int)
    p.add_argument("--limit_batches", default=0, type=int)

    p.add_argument("--setting", required=True, choices=["raw", "inject", "search"], help="Evaluation mode")
    # inject-mode controls
    p.add_argument("--inject_where", choices=["prepend", "append"], default="prepend")
    p.add_argument("--inject_sources", default="img,text", help="Comma list of sources to query: img, text, both")
    # shared search controls (inject/search)
    p.add_argument("--search_api", default="local", choices=["local", "cragmm"])
    p.add_argument("--retrieve_type", default="title", choices=["title","first_sent","full"])
    p.add_argument("--search_k", default=5, type=int)

    p.add_argument("--max_new_tokens", default=128, type=int)
    p.add_argument("--predictions_path", required=True, type=str)
    p.add_argument("--llm_as_judge", action="store_true", help="(optional) not used by default; stick to TextVQAAccuracyEvaluator")
    p.add_argument("--format_with_llm", action="store_true", help="Format the results with an LLM")
    p.add_argument("--llm_model", default="gpt-4o", type=str)
    p.add_argument("--use_llm_as_judge", action="store_true")
    return p

# -----------------------------------------------------------------------------
# Data loader (reuse your collate_fn)
# -----------------------------------------------------------------------------
def build_val_dataset(args, vision_tower_id="openai/clip-vit-large-patch14"):
    ds = args.val_dataset
    if "textvqa" in ds:
        return TextVQADataset(args.dataset_dir, None, vision_tower_id, use_pil_image=True)
    if "gqa" in ds:
        return GQADataset(args.dataset_dir, None, vision_tower_id, use_pil_image=True)
    if "pope" in ds:
        return POPEDataset(args.dataset_dir, None, vision_tower_id, use_pil_image=True)
    if "cragmm" in ds:
        return CragMMDataset(args.dataset_dir, None, vision_tower_id, use_pil_image=True)
    if "infoseek" in ds:
        return InfoseekDataset(args.dataset_dir, None, vision_tower_id, use_pil_image=True)
    if "oven" in ds:
        return OvenDataset(args.dataset_dir, None, vision_tower_id, use_pil_image=True)
    raise ValueError(f"Unsupported val_dataset: {ds}")

def build_loader(args, tokenizer_stub=None):
    val_dataset = build_val_dataset(args)
    assert args.val_batch_size == 1, "Batch size must be 1 for this evaluator."
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=None  # No custom collation - use dataset items directly
    )

# -----------------------------------------------------------------------------
# Question extraction from your conversations (robust-ish)
# -----------------------------------------------------------------------------
def extract_question(conv_text: str) -> str:
    """
    Your conversation strings look like:
      "USER: <image>\n What is ... ? ASSISTANT: ..."
    We try to capture what's after the <image> marker and before ASSISTANT.
    """
    s = conv_text
    # remove any role labels for safety
    s = s.replace("USER:", "").replace("ASSISTANT:", "")
    # split around your image placeholder line
    parts = s.split("\n")
    # try to find the line containing the question text (skip image token line)
    question_lines = []
    for line in parts:
        if "<image>" in line.lower():
            continue
        if "<img_start>" in line.lower() or "<img_end>" in line.lower():
            continue
        question_lines.append(line.strip().replace("###", ""))
    q = " ".join([t for t in question_lines if t]).strip()
    if not q:
        # fallback: keep original
        q = conv_text
    return q

# -----------------------------------------------------------------------------
# Search pipeline adapter
# -----------------------------------------------------------------------------
def get_search_pipeline(args):
    if args.search_api == "local":
        return LocalSearchPipeline(
            base_url="http://127.0.0.1:7000",
            collection="wiki10k_index",
            index_name=None,
            pooling="mean",
        )
    elif args.search_api == "cragmm":
        from cragmm_search.search import UnifiedSearchPipeline
        os.environ["HTTPS_PROXY"] = "http://fwdproxy:8080"
        os.environ["NO_PROXY"] = ",".join([
            ".fbcdn.net", ".facebook.com", ".thefacebook.com", ".tfbnw.net",
            ".fb.com", ".fburl.com", ".facebook.net", ".sb.fbsbx.com", "localhost"
        ])
        return UnifiedSearchPipeline(
            image_model_name="openai/clip-vit-large-patch14-336",
            image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
            text_model_name="BAAI/bge-large-en-v1.5",
            web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
        )
    else:
        raise ValueError(f"Unknown search_api: {args.search_api}")


def format_results_with_llm(
    results: List[dict],
    retrieve_type: str,
    openai_client: openai.Client,
    model: str,
    temperature: float = 0.2,
    **chat_completion_kwargs: Dict[str, Any]) -> str:
    '''
    Uses an LLM (e.g., gpt-4o-mini, gpt-4.1-mini) to extract and flatten the
    essential retrieved entities or phrases from arbitrary retrieval results
    into a compact, '|' separated string.
    '''
    if results is None or len(results) == 0:
        return " "

    # Convert results to a compact string summary for the prompt
    try:
        serialized = json.dumps(results, ensure_ascii=False)
    except Exception:
        serialized = str(results)

    # Build prompt for LLM
    prompt = f"""
                You are an assistant that extracts the key entities or phrases from search results.

                Below is a JSON-like list of retrieved items. Each item may contain 'title', 'content', or nested lists.
                Your job is to extract the most informative and relevant entities or short phrases (no full sentences),
                separated by ' | ' (pipe character). Avoid redundancy and remove any special characters.

                Return ONLY the compact formatted string, nothing else.

                Retrieve type: {retrieve_type}
                Results:
                {serialized}
              """

    messages = [
        {"role": "system", "content": "You are a concise text formatter."},
        {"role": "user", "content": prompt},
    ]

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=256,
        )
        formatted = resp.choices[0].message.content.strip()
        # sanitize to remove accidental prefixes or line breaks
        formatted = re.sub(r"[\n\r]+", " ", formatted)
        formatted = re.sub(r"\s*\|\s*", " | ", formatted)

        # Ensure fallback if model gives empty output
        if not formatted:
            formatted = "_"
        return formatted
    except Exception as e:
        print(f"[format_results_with_llm] LLM formatting failed: {e}")

        # fallback: flatten manually
        def flatten(item):
            if isinstance(item, (list, tuple)):
                return " | ".join(flatten(x) for x in item)
            elif isinstance(item, dict):
                vals = []
                for k in ["title", "content", "name", "entity"]:
                    if k in item and isinstance(item[k], str):
                        vals.append(item[k])
                return " ".join(vals)
            elif isinstance(item, str):
                return item
            return ""
        fallback = [flatten(r) for r in results if r]
        fallback_str = " | ".join([x.strip() for x in fallback if x.strip()])
        return fallback_str if fallback_str else "_"


def format_results(results: List[dict], retrieve_type: str, format_with_llm: bool = False) -> str:
    if results is None or len(results) == 0:
        return "_"
    safe = []
    for r in results:
        if "error" in r:
            continue
        title = r.get("title","").strip()
        content = r.get("content","").strip()
        if retrieve_type == "title":
            if title:
                safe.append(title)
        elif retrieve_type == "first_sent":
            # super simple first sentence cut
            first = content.split(".")[0].strip()
            if title and first:
                safe.append(f"{title}: {first}.")
            elif first:
                safe.append(first + ".")
        else:
            if title and content:
                safe.append(f"{title}: {content}")
            elif content:
                safe.append(content)

    if not safe:
        return "_"
    # collapse to a compact payload
    return " | ".join(safe[:10])

# -----------------------------------------------------------------------------
# Llama-3.2 helpers
# -----------------------------------------------------------------------------
def load_llama32(model_id: str, device: str, precision: str):
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def load_rgb_image(image_path: str) -> Image.Image:
    # robust load fallback
    if (image_path is None) or (not os.path.exists(image_path)):
        return Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------
def gen_once(model, processor, image_pil: Image.Image, prompt: str, max_new_tokens: int):
    inputs = processor(image_pil, prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.decode(out[0], skip_special_tokens=False)
    return text

def evaluate_raw(args, model, processor, val_loader, out_path):
    evaluator = TextVQAAccuracyEvaluator()
    os.makedirs(out_path, exist_ok=True)
    dump_path = os.path.join(out_path, f"predictions_{args.val_dataset.replace('|','_')}_raw.jsonl")

    # load from the out_path and start from the last index
    if os.path.exists(dump_path):
        with open(dump_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                last_line_idx = line_idx
    else:
        last_line_idx = 0

    last_line_idx += 1

    n_batches = min(args.limit_batches, len(val_loader)) if args.limit_batches > 0 else len(val_loader)
    it = iter(val_loader)
    pbar = tqdm.tqdm(range(n_batches), desc="[raw] Llama-3.2-Vision eval")

    acc = 0.0
    f1_score = 0.0
    for line_idx in pbar:
        if line_idx < last_line_idx:
            continue
        try:
            batch = next(it)
        except Exception:
            break

        # Dataset returns: (image_path, image, image_clip, conversations, masks, label, resize, questions, sampled_classes, ...)
        img_path = batch[0][0]  # First element is image_path, batch is list of tuples
        conv = batch[3][0][0]  # conversations is at index 3, first conversation
        question = extract_question(conv)

        image_pil = load_rgb_image(img_path) if batch[1] is None else batch[1]

        prompt = f"<|image|><|begin_of_text|>{question}"
        output_text = gen_once(model, processor, image_pil, prompt, args.max_new_tokens)

        gt = batch[8][0][0]  # sampled_classes is at index 8
        if args.use_llm_as_judge and args.val_dataset == "cragmm":
            score, explanation = evaluate_answer(output_text, gt, model=args.llm_model)
            acc = score
            f1_score = 0.0
        else:
            evaluator = TextVQAAccuracyEvaluator()
            acc, f1_score = evaluator.eval_pred_list([{'pred_answer': output_text, 'gt_answers': gt}])

        rec = {"image_path": img_path, "question": question, "output_text": output_text, "ground_truth_text": gt, "accuracy": acc, "f1_score": f1_score}
        with open(dump_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # summarize
    accs = []
    f1_scores = []
    with open(dump_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                accs.append(json.loads(line)["accuracy"])
                f1_scores.append(json.loads(line)["f1_score"])
            except Exception:
                pass
    total_acc = float(sum(accs) / max(1, len(accs)))
    total_f1 = float(sum(f1_scores) / max(1, len(f1_scores)))
    print(f"==> [raw] Total Accuracy: {total_acc:.4f}")
    print(f"==> [raw] Total F1: {total_f1:.4f}")

def run_inject_query(search_pipe, args, img_path: str, question: str, use_img: bool, use_text: bool):
    queries = []
    if use_img:
        queries.append([os.path.abspath(img_path)] if img_path else [""])
    if use_text:
        queries.append(question)
    merged = []
    for q in queries:
        try:
            res = search_pipe(q, k=args.search_k)
            if res is not None:
                merged.extend(res)
        except Exception as e:
            merged.append({"error": str(e)})
    return merged

def evaluate_inject(args, model, processor, val_loader, out_path):
    evaluator = TextVQAAccuracyEvaluator()
    TIMEOUT_SECONDS = 120   # make sure OPENAI_API_KEY is set in your environment.
    openai_client = openai.Client(timeout=TIMEOUT_SECONDS)
    os.makedirs(out_path, exist_ok=True)
    dump_path = os.path.join(out_path, f"predictions_{args.val_dataset.replace('|','_')}_inject_{args.inject_where}.jsonl")

    # load and read from dump_path and start from the last line_idx
    if os.path.exists(dump_path):
        with open(dump_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                last_line_idx = line_idx
    else:
        last_line_idx = 0

    last_line_idx += 1

    n_batches = min(args.limit_batches, len(val_loader)) if args.limit_batches > 0 else len(val_loader)
    it = iter(val_loader)
    pbar = tqdm.tqdm(range(n_batches), desc=f"[inject:{args.inject_where}] Llama-3.2-Vision eval")

    search_pipe = get_search_pipeline(args)
    use_img = "img" in [s.strip().lower() for s in args.inject_sources.split(",")]
    use_text = "text" in [s.strip().lower() for s in args.inject_sources.split(",")]

    for line_idx in pbar:
        if line_idx < last_line_idx:
            continue

        breakpoint()

        try:
            batch = next(it)
        except StopIteration:
            break

        # Dataset returns: (image_path, image, image_clip, conversations, masks, label, resize, questions, sampled_classes, ...)
        img_path = batch[0][0]  # First element is image_path
        conv = batch[3][0][0]  # conversations is at index 3, first conversation
        question = extract_question(conv)
        image_pil = load_rgb_image(img_path) if batch[1] is None else batch[1]

        # retrieve
        results = run_inject_query(search_pipe, args, img_path, question, use_img, use_text)

        if args.format_with_llm:
            info = format_results_with_llm(results, args.retrieve_type, openai_client, args.llm_model, temperature=0.2, max_tokens=256)
        else:
            info = format_results(results, args.retrieve_type)
        info_block = f"{INFO_START_TOKEN} {info} {INFO_END_TOKEN}"

        if args.inject_where == "prepend":
            prompt = f"<|image|><|begin_of_text|>Retrieved Information: {info_block}\n{question}Answer:"
        else:  # append
            prompt = f"<|image|><|begin_of_text|>{question}\nRetrieved Information:{info_block}\nAnswer:"

        output_text = gen_once(model, processor, image_pil, prompt, args.max_new_tokens)

        gt = batch[8][0][0]  # sampled_classes is at index 8
        if args.use_llm_as_judge:
            score, explanation = evaluate_answer(output_text, gt, model=args.llm_model)
            acc = score
            f1_score = 0.0
        else:
            acc, f1_score = evaluator.eval_pred_list([{"pred_answer": output_text, "gt_answers": gt}])

        rec = {"image_path": img_path, "question": question, "info": info, "output_text": output_text, "ground_truth_text": gt, "accuracy": acc, "f1_score": f1_score}
        with open(dump_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    accs = []
    f1_scores = []
    with open(dump_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                accs.append(json.loads(line)["accuracy"])
                f1_scores.append(json.loads(line)["f1_score"])
            except Exception:
                pass
    total_acc = float(sum(accs) / max(1, len(accs)))
    total_f1 = float(sum(f1_scores) / max(1, len(f1_scores)))
    print(f"==> [inject] Total Accuracy: {total_acc:.4f}")
    print(f"==> [inject] Total F1: {total_f1:.4f}")

def find_closed_search_spans(text: str) -> List[Tuple[int,int,str]]:
    """
    Returns list of (start_idx, end_idx, payload_text) for fully closed <search>...</search> spans
    in the *tail* you just generated.
    """
    spans = []
    cur = 0
    while True:
        s = text.find(SEARCH_START_TOKEN, cur)
        if s == -1: break
        e = text.find(SEARCH_END_TOKEN, s + len(SEARCH_START_TOKEN))
        if e == -1: break
        payload = text[s+len(SEARCH_START_TOKEN): e].strip()
        spans.append((s, e+len(SEARCH_END_TOKEN), payload))
        cur = e + len(SEARCH_END_TOKEN)
    return spans

def evaluate_search(args, model, processor, val_loader, out_path):
    """
    0-shot <search> prompting loop. We generate in chunks; when a closed <search>...</search> appears
    in the newly generated tail, we execute the search and inject <information>...</information> back into
    the *prompt*, then continue generation.
    """
    evaluator = TextVQAAccuracyEvaluator()
    os.makedirs(out_path, exist_ok=True)
    dump_path = os.path.join(out_path, f"predictions_{args.val_dataset.replace('|','_')}_0shot_search.jsonl")

    search_pipe = get_search_pipeline(args)

    n_batches = min(args.limit_batches, len(val_loader)) if args.limit_batches > 0 else len(val_loader)
    it = iter(val_loader)
    pbar = tqdm.tqdm(range(n_batches), desc="[search] Llama-3.2-Vision eval")

    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            break

        # Dataset returns: (image_path, image, image_clip, conversations, masks, label, resize, questions, sampled_classes, ...)
        img_path = batch[0][0]  # First element is image_path
        image_pil = load_rgb_image(img_path) if batch[1] is None else batch[1]
        conv = batch[0][3][0]  # conversations is at index 3, first conversation
        question = extract_question(conv)

        breakpoint()

        # System instruction per your spec:
        sys_prompt = (
            "Answer the given question. You must conduct reasoning to answer the given question. "
            "While you reason, if you find you lack some knowledge, you can call a search engine by "
            "<search> query </search>, and it will return the top searched results between "
            "<information> and </information>. For the query type, there are three choices: "
            "<region>, <img>, and text. If you generate <search> <region> </search>, it means you "
            "need more specific information about a region/entity/object in the image and use it to query "
            "the search api. If you generate <search> <img> </search>, it means you need more contextual "
            "and holistic information and use the whole image to query the search api. If you generate a textual "
            "query/question inside <search> </search>, then it means the external knowledge needed cannot be queried "
            "directly by image and you need textual question. You can search as many times as you want. If you find no "
            "further external knowledge needed, you can directly provide the answer."
        )

        # running prompt that accumulates injected info
        base_prompt = f"<|image|><|begin_of_text|>{sys_prompt}\n\nQuestion: {question}\n\nAnswer:"
        final_output = ""
        max_rounds = 6         # how many search/inject rounds
        chunk_tokens = 96      # tokens per generation chunk
        rounds = 0

        while rounds < max_rounds:
            rounds += 1
            # generate chunk
            text_full = gen_once(model, processor, image_pil, base_prompt, chunk_tokens)
            # isolate new tail after base prompt
            if text_full.startswith(base_prompt):
                new_tail = text_full[len(base_prompt):]
            else:
                # fallback if tokenizer adds its own wrappers
                new_tail = text_full

            # see if a closed <search> span appears in the new tail
            spans = find_closed_search_spans(new_tail)
            if not spans:
                # if no search appeared, either keep extending or stop if eos-ish patterns
                # simple stopping heuristic
                if "</s>" in new_tail.lower() or "###" in new_tail or len(new_tail.strip()) == 0:
                    final_output = text_full
                    break
                # else, extend base prompt and continue (to allow multi-chunk)
                base_prompt = text_full
                continue

            # take the *last* completed search span in the latest tail
            _, _, payload = spans[-1]
            payload_norm = payload.strip().lower()

            # decide query mode
            is_img = (payload_norm == IMG_PAYLOAD_TOKEN.lower())
            is_region = (payload_norm == REGION_TOKEN.lower())
            # NOTE: region fallback: we cannot crop without SAM here, so we use text+img heuristic:
            # If <region>, we use the *preceding words* near the search tag from the current tail (very simple heuristic)
            region_text = None
            if is_region:
                # grab up to ~10 tokens before <search> in the new tail:
                ctx = new_tail
                s = ctx.rfind(SEARCH_START_TOKEN)
                window = ctx[max(0, s-128): s].strip()
                # take last sentence fragment as region text
                # (You can improve parsing as needed for your data)
                tokens = window.split()
                region_text = " ".join(tokens[-10:]) if tokens else ""

            # build input_query for the pipeline
            if is_img:
                input_query = [os.path.abspath(img_path)] if img_path else [""]
            elif is_region:
                # fallback: combine image + nearby text cue
                if region_text:
                    # pipeline will typically use the text; image is provided as context by passing a list
                    input_query = region_text
                else:
                    input_query = [os.path.abspath(img_path)] if img_path else [""]
            else:
                # textual query
                input_query = payload

            # execute
            try:
                results = search_pipe(input_query, k=args.search_k)
            except Exception as e:
                results = [{"error": str(e)}]

            info = format_results(results, args.retrieve_type)
            info_block = f"{INFO_START_TOKEN} {info} {INFO_END_TOKEN}"

            # inject right after the current tail (append to prompt)
            base_prompt = text_full + "\n" + info_block + "\n"

            # loop to allow more generation with new knowledge

        if not final_output:
            final_output = base_prompt  # last known prompt content

        gt = batch[0][8]  # sampled_classes is at index 8
        if args.use_llm_as_judge:
            score, explanation = evaluate_answer(final_output, gt, model=args.llm_model)
            acc = score
            f1_score = 0.0
        else:
            acc, f1_score = evaluator.eval_pred_list([{"pred_answer": final_output, "gt_answers": gt}])

        rec = {
            "image_path": img_path,
            "question": question,
            "output_text": final_output,
            "ground_truth_text": gt,
            "accuracy": acc,
            "f1_score": f1_score
        }
        with open(dump_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # summarize
    accs = []
    f1_scores = []
    with open(dump_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                accs.append(json.loads(line)["accuracy"])
                f1_scores.append(json.loads(line)["f1_score"])
            except Exception:
                pass
    total_acc = float(sum(accs) / max(1, len(accs)))
    total_f1 = float(sum(f1_scores) / max(1, len(f1_scores)))
    print(f"==> [search] Total Accuracy: {total_acc:.4f}")
    print(f"==> [search] Total F1: {total_f1:.4f}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    # model
    model, processor = load_llama32(args.model_id, args.device, args.precision)
    model.eval()

    # data
    val_loader = build_loader(args)

    # output
    Path(args.predictions_path).mkdir(parents=True, exist_ok=True)

    # eval
    if args.setting == "raw":
        evaluate_raw(args, model, processor, val_loader, args.predictions_path)
    elif args.setting == "inject":
        evaluate_inject(args, model, processor, val_loader, args.predictions_path)
    elif args.setting == "search":
        evaluate_search(args, model, processor, val_loader, args.predictions_path)
    else:
        raise ValueError(args.setting)

if __name__ == "__main__":
    main()
