# validate_qa.py
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from functools import partial

import torch
import tqdm
import transformers
import jsonargparse as argparse
from transformers import BitsAndBytesConfig

from model.PIXAR import PIXARForCausalLM
from utils.dataset import collate_fn
from utils.m4c_evaluator import TextVQAAccuracyEvaluator
from utils.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    SEARCH_START_TOKEN, SEARCH_END_TOKEN,
    INFORMATION_START_TOKEN, INFORMATION_END_TOKEN,
    DEFAULT_IMG_PAYLOAD_TOKEN, DEFAULT_REGION_TOKEN,
    IMAGE_TOKEN_INDEX, IM_START_ID, IM_END_ID,
    build_first_sentence_fn, format_payload
)
from chat_pixar_search import (
    tokenizer_decode_special_tokens,
    append_text_to_ids,
    find_sub_ids,
    find_last_closed_span_ids,
    last_span_before,
    save_cropped_region,
    SearchTagGuard
)

# optional LLM-as-judge (kept consistent with validate_vqa.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.evaluation.llm_as_judge_eval import evaluate_answer

# Local search pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.local_search_api import LocalSearchPipeline

from utils.qa_dataset import build_qa_dataset


def parse_args(args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser(description="Validate PIXAR on text-only QA datasets (with optional <search> API)")
    # model + tokenizer
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--backbone", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview", type=str)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # dataset
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--val_dataset", default="hotpotqa|validation",
                        help="Format: <name>|<split>; name in {nq, hotpotqa, musique, triviaqa}")
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument('--limit_batches', default=0, type=int)
    parser.add_argument("--image_size", default=1024, type=int)

    # collate / conversation
    parser.add_argument("--conv_type", default="llava_v1",
                        choices=["llava_v1", "llava_llama_2", "pixar_0shot_v1"])
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)

    # judge/eval
    parser.add_argument("--llm_model", default="gpt-4o", type=str)
    parser.add_argument("--use_llm_as_judge", action="store_true")

    # device + outputs
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument("--exp_name", default="plum-13b-qa-eval", type=str)
    parser.add_argument('--predictions_path', type=str, default='/shared/nas2/jk100/partonomy_private/results/')

    # search toggles (match validate_vqa.py)
    parser.add_argument("--no_search", action="store_true",
                        help="If set, run regular validate (no <search> API).")
    parser.add_argument("--retrieve_type", default="title", choices=["title","first_sent","full"],
                        help="What to inject inside <information>: titles, first_sentences, or full text")

    args = parser.parse_string(config_str) if config_str else parser.parse_args(args)
    return args, parser


def _resolve_dataset_name_and_split(spec: str):
    return spec.split("|")[0], (spec.split("|")[1] if "|" in spec else "validation")


def _to_pixar_inputs(args, batch: dict, tokenizer: transformers.AutoTokenizer):
    return {
        'images': batch['images'].to(args.device),
        'images_clip': batch['images_clip'].to(args.device),
        "input_ids": batch['input_ids'].to(args.device),
        'resize_list': batch['resize_list'],
        'label_list': batch['label_list'],
        'attention_masks': batch['attention_masks'].to(args.device),
        'max_new_tokens': args.model_max_length,
        'tokenizer': tokenizer,
    }


@torch.no_grad()
def validate(val_loader, model, tokenizer, args):
    model.eval()
    assert args.val_batch_size == 1, "Currently supports batch_size=1."

    n_batches = min(args.limit_batches, len(val_loader)) if args.limit_batches > 0 else len(val_loader)

    Path(args.predictions_path).mkdir(parents=True, exist_ok=True)
    name, split = _resolve_dataset_name_and_split(args.val_dataset)
    pred_file_path = os.path.join(
        args.predictions_path, f'predictions_{name}_{split}_{args.llm_model}_{args.use_llm_as_judge}.jsonl'
    )

    evaluator = TextVQAAccuracyEvaluator()

    total_accuracy, total_f1, written = 0.0, 0.0, 0
    val_iter = iter(val_loader)
    pbar = tqdm.tqdm(range(n_batches), desc=f"Validating PIXAR on QA ({name}|{split})")

    for _ in pbar:
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        # dtype
        if args.precision == "fp16":
            to_dtype = lambda x: x.half()
        elif args.precision == "bf16":
            to_dtype = lambda x: x.bfloat16()
        else:
            to_dtype = lambda x: x.float()
        batch['images']      = to_dtype(batch['images'])
        batch['images_clip'] = to_dtype(batch['images_clip'])

        out_eval = model.evaluate(**_to_pixar_inputs(args, batch, tokenizer))
        output_ids = out_eval.get("output_ids", None)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False) if output_ids is not None else ""

        gt_answers = batch['sampled_classes_list'][0]
        if args.use_llm_as_judge:
            score, _ = evaluate_answer(output_text, gt_answers, model=args.llm_model)
            accuracy = score
        else:
            accuracy, f1_score = evaluator.eval_pred_list([{'pred_answer': output_text, 'gt_answers': gt_answers}])

        total_accuracy += accuracy
        total_f1 += f1_score
        written += 1

        with open(pred_file_path, "a") as f:
            f.write(json.dumps({
                "image_path": batch['image_paths'][0],  # "" for text-only
                "output_text": output_text,
                "ground_truth_text": gt_answers,
                "accuracy": accuracy,
                "f1": f1_score
            }, ensure_ascii=False) + "\n")

        pbar.set_postfix(acc=accuracy, avg_acc=(total_accuracy / max(1, written)))

    # summarize
    existing = []
    if os.path.exists(pred_file_path):
        with open(pred_file_path, "r") as f:
            for line in f:
                existing.append(json.loads(line))
        final_acc = sum(d["accuracy"] for d in existing) / max(1, len(existing))
        final_f1 = sum(d["f1"] for d in existing) / max(1, len(existing))
    else:
        final_acc = total_accuracy / max(1, written)
        final_f1 = total_f1 / max(1, written)
    print(f"==> Total Accuracy ({name}|{split}): {final_acc:.4f}")
    print(f"==> Total F1 ({name}|{split}): {final_f1:.4f}")


@torch.no_grad()
def validate_search(
    val_loader,
    model,
    tokenizer,
    args,
    *,
    search_pipeline,
    search_k: int = 5,
    max_new_tokens: int = 1024
):
    """
    Text-only version of the <search> decode loop:
      - Detect <search>...</search> spans while generating one token at a time.
      - Use payload text or the immediate preceding span text as query.
      - Inject <information>{payload}</information> back into the stream.
    Image/region payloads are treated as TEXT in this text-only setting.
    """
    model.eval()
    assert args.val_batch_size == 1, "Currently supports batch_size=1."

    name, split = _resolve_dataset_name_and_split(args.val_dataset)
    Path(args.predictions_path).mkdir(parents=True, exist_ok=True)
    pred_file_path = os.path.join(
        args.predictions_path, f'predictions_{name}_{split}_{args.llm_model}_{args.use_llm_as_judge}_search.jsonl'
    )

    total_accuracy, total_f1, written = 0.0, 0.0, 0
    evaluator = TextVQAAccuracyEvaluator()
    eos_id = tokenizer.eos_token_id

    SEARCH_START_IDS = torch.tensor(tokenizer.encode(SEARCH_START_TOKEN, add_special_tokens=False)).cpu().long()
    SEARCH_END_IDS   = torch.tensor(tokenizer.encode(SEARCH_END_TOKEN,   add_special_tokens=False)).cpu().long()

    # guardrail (optional)
    guard = SearchTagGuard(
        start_ids=list(SEARCH_START_IDS.tolist()),
        end_ids=list(SEARCH_END_IDS.tolist()),
        region_ids=[],  # not used for text-only
    )

    n_batches = min(args.limit_batches, len(val_loader)) if args.limit_batches > 0 else len(val_loader)
    val_iter = iter(val_loader)
    pbar = tqdm.tqdm(range(n_batches), desc=f"Validating with <search> ({name}|{split})")

    for _ in pbar:
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        # dtype
        if args.precision == "fp16":
            to_dtype = lambda x: x.half()
        elif args.precision == "bf16":
            to_dtype = lambda x: x.bfloat16()
        else:
            to_dtype = lambda x: x.float()
        batch['images']      = to_dtype(batch['images'])
        batch['images_clip'] = to_dtype(batch['images_clip'])

        # base inputs
        device = next(model.parameters()).device
        images_clip = batch['images_clip'].to(device)
        images_for_sam = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_masks'].to(device)

        generated = input_ids.clone()
        steps = max_new_tokens
        base_len = generated.shape[1]

        open_search_starts = []
        last_handled_end_abs = base_len
        first_sentence = build_first_sentence_fn()

        while steps > 0:
            out = model.generate(
                images=images_clip,
                input_ids=generated,
                max_new_tokens=1,
                num_beams=1,
                do_sample=False,
                use_cache=False,
                return_dict_in_generate=True,
                output_hidden_states=False,
                # logits_processor=[guard],  # optional
            )
            generated = out.sequences
            steps -= 1
            if generated[0, -1].item() == eos_id:
                break

            gen_ids = generated[0, base_len:].detach().cpu().long()

            # check open/close
            just_opened = gen_ids.numel() >= len(SEARCH_START_IDS) and torch.equal(
                gen_ids[-len(SEARCH_START_IDS):], SEARCH_START_IDS
            )
            if just_opened:
                open_search_starts.append(base_len + gen_ids.numel() - len(SEARCH_START_IDS))

            just_closed = gen_ids.numel() >= len(SEARCH_END_IDS) and torch.equal(
                gen_ids[-len(SEARCH_END_IDS):], SEARCH_END_IDS
            )
            if not just_closed:
                continue
            if not open_search_starts:
                continue

            end_abs = base_len + gen_ids.numel()
            if end_abs <= last_handled_end_abs:
                continue
            start_abs = open_search_starts.pop()
            if start_abs <= last_handled_end_abs:
                last_handled_end_abs = end_abs
                continue

            # payload ids/text
            s_rel = start_abs - base_len
            e_rel = end_abs - base_len
            payload_ids = gen_ids[s_rel + len(SEARCH_START_IDS): e_rel - len(SEARCH_END_IDS)]
            payload_text = tokenizer.decode(
                tokenizer_decode_special_tokens(payload_ids, tokenizer), skip_special_tokens=False
            ).strip()

            # choose query (text-only: use payload text; if empty, back off to span before <search>)
            if payload_text:
                input_query = payload_text
            else:
                # take up to 10 words immediately before "<search>"
                gen_text = tokenizer.decode(tokenizer_decode_special_tokens(gen_ids, tokenizer), skip_special_tokens=False)
                input_query = last_span_before(gen_text, SEARCH_START_TOKEN, max_words=10)
                if not input_query.strip():
                    input_query = ""  # final fallback; will return empty results

            # run retrieval
            try:
                results = search_pipeline(input_query, k=search_k)
            except Exception as e:
                results = [{"error": str(e), "input_query": input_query}]

            if results is None:
                payload_contents = []
            else:
                if args.retrieve_type == "first_sent":
                    payload_contents = [first_sentence(r.get('content', "")) for r in results]
                elif args.retrieve_type == "title":
                    payload_contents = [r.get('title', "") for r in results]
                else:  # "full"
                    payload_contents = [{r.get('title', f'doc_{i}'): r.get('content', "")} for i, r in enumerate(results)]

            payload_text_out = format_payload(payload_contents)
            injected = f"{INFORMATION_START_TOKEN} {payload_text_out} {INFORMATION_END_TOKEN}"
            generated = append_text_to_ids(tokenizer, generated, injected, device)

            last_handled_end_abs = end_abs

        # decode final
        interleaved_text = tokenizer.decode(
            tokenizer_decode_special_tokens(generated[0], tokenizer),
            skip_special_tokens=False
        )

        # evaluate
        gt_answers = batch['sampled_classes_list'][0]
        if args.use_llm_as_judge:
            score, _ = evaluate_answer(interleaved_text, gt_answers, model=args.llm_model)
            accuracy = score
        else:
            accuracy, f1_score = evaluator.eval_pred_list([{'pred_answer': interleaved_text, 'gt_answers': gt_answers}])

        total_accuracy += accuracy
        total_f1 += f1_score
        written += 1

        with open(pred_file_path, "a") as f:
            f.write(json.dumps({
                "image_path": batch['image_paths'][0],  # "" for text-only
                "output_text": interleaved_text,
                "ground_truth_text": gt_answers,
                "accuracy": accuracy,
                "f1": f1_score
            }, ensure_ascii=False) + "\n")

    # summarize
    existing = []
    if os.path.exists(pred_file_path):
        with open(pred_file_path, "r") as f:
            for line in f:
                existing.append(json.loads(line))
        final_acc = sum(d["accuracy"] for d in existing) / max(1, len(existing))
        final_f1 = sum(d["f1"] for d in existing) / max(1, len(existing))
    else:
        final_acc = 0.0
        final_f1 = 0.0
    print(f"==> Total Accuracy with <search> ({name}|{split}): {final_acc:.4f}")
    print(f"==> Total F1 ({name}|{split}): {final_f1:.4f}")


def main(args: list[str] = None, config_str: str = None):
    args, _ = parse_args(args, config_str)

    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.backbone,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    # dtype & quant
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {
        "torch_dtype": torch_dtype,
        "use_bidir_bio": False,
        "use_crf_bio": False,
    }
    if args.load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        })

    # model
    print(f"*(validate_qa) >> Loading lora-merged model checkpoints from [ {args.version} ]*")
    model = PIXARForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        **kwargs
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # PIXAR vision modules (kept even for text-only for shared forward compatibility)
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=torch.device("cuda", args.local_rank))

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
        vt = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed
        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vt.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    model.eval()

    # dataset (use inference=False to include gold assistant turn for CE targets; you can set True if desired)
    name, split = _resolve_dataset_name_and_split(args.val_dataset)
    qa_ds = build_qa_dataset(
        name=name,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        split=split,
        image_size=args.image_size,
        inference=False,
    )

    val_loader = torch.utils.data.DataLoader(
        qa_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
    )

    if args.no_search:
        validate(val_loader, model, tokenizer, args)
    else:
        search_pipeline = LocalSearchPipeline(
            base_url="http://127.0.0.1:7000",
            collection="wiki10k_index",
            index_name=None,
            pooling="mean",
        )
        validate_search(
            val_loader, model, tokenizer, args,
            search_pipeline=search_pipeline, search_k=5,
            max_new_tokens=args.model_max_length
        )


if __name__ == "__main__":
    main(sys.argv[1:])
