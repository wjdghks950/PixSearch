import jsonargparse as argparse
import json
import os
import sys
import cv2
import time
import numpy as np

import torch
import tqdm
import transformers
from itertools import islice, chain
from transformers import BitsAndBytesConfig
from functools import partial
from pathlib import Path

from model.PIXAR import PIXARForCausalLM

from utils.vqa_dataset import TextVQADataset, GQADataset, POPEDataset, CragMMDataset, InfoseekDataset, OvenDataset
from utils.m4c_evaluator import TextVQAAccuracyEvaluator
from utils.dataset import collate_fn
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
# add IoUEvaluator evaluation functionality
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))
from src.evaluation.rle_dict import get_mask_rle_dicts
from src.evaluation.llm_as_judge_eval import evaluate_answer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.local_search_api import LocalSearchPipeline

import logging
import coloredlogs

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args(args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser(description="Validate PIXAR on VQA datasets")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--workers", default=0, type=int, help="number of workers")
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--backbone", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview", type=str)
    parser.add_argument(
        "--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"],
        help="precision for inference"
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--val_dataset", default="cragmm|validation", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument('--limit_batches', default=0, type=int)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2", "pixar_0shot_v1"],
    )
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)

    parser.add_argument("--use_bidir_bio", action="store_true", default=False)
    parser.add_argument("--use_crf_bio", action="store_true", default=False)

    parser.add_argument("--llm_model", default="gpt-4o", type=str)
    parser.add_argument("--use_llm_as_judge", action="store_true")

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--predictions_path', type=str, default='/shared/nas2/jk100/partonomy_private/results/')

    parser.add_argument("--exp_name", default="plum-13b-vqa-eval", type=str)

    parser.add_argument("--no_search", action="store_true", help="If set to True, run regular validate instead of validate_search")
    parser.add_argument("--search_api", default="local", help="search API type; either `local` or `cragmm`.")
    parser.add_argument("--retrieve_type", default="title", help="Retrieved knowledge snippet boundary - ['first_sent', 'title']")

    args = parser.parse_string(config_str) if config_str else parser.parse_args(args)
    return args, parser


def main(args: list[str] = None, config_str: str = None):
    args, parser = parse_args(args, config_str)

    print(f"*(validate_vqa) >> Loading lora-merged model checkpoints from [ {args.version} ]*")

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.backbone,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False #True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    # build PLUM model
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {
        "torch_dtype": torch_dtype,
        "use_bidir_bio": args.use_bidir_bio,
        "use_crf_bio": args.use_crf_bio
        }
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = PIXARForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        **kwargs
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # initialize vision modules in PLUM
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=torch.device("cuda", args.local_rank))

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    model.eval()

    # build dataset for validation
    if "textvqa" in args.val_dataset:
        val_dataset = TextVQADataset(
            base_image_dir=args.dataset_dir,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower
        )
    elif "gqa" in args.val_dataset:
        val_dataset = GQADataset(
            base_image_dir=args.dataset_dir,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower
        )
    elif "pope" in args.val_dataset:
        val_dataset = POPEDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower
        )
    elif "cragmm" in args.val_dataset:
        val_dataset = CragMMDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower
        )
    elif "infoseek" in args.val_dataset:
        val_dataset = InfoseekDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower
        )
    elif "oven" in args.val_dataset:
        val_dataset = OvenDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower
        )
    # build DataLoader
    if val_dataset is not None:
        assert args.val_batch_size == 1, "Currently supports batch_size=1 for segmentation eval"
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
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
    else:
        raise ValueError("No validation dataset provided. Exiting.")

    # run evaluation on the partonomy dataset
    if args.no_search:
        validate(val_loader, model, tokenizer, args, parser)
    else:
        search_api_type = args.search_api.lower().strip()
        if search_api_type == "local":
            search_pipeline = LocalSearchPipeline(
                base_url="http://127.0.0.1:7000",
                collection="wiki10k_index",
                index_name=None,
                pooling="mean",
            )
        elif search_api_type == "cragmm":
            from cragmm_search.search import UnifiedSearchPipeline
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
        # run search API calling `validate_search`
        validate_search(
            val_loader, model, tokenizer, args, parser,
            search_pipeline=search_pipeline, search_k=5,
            vis_save_path="./vis_output", max_new_tokens=args.model_max_length
        )


def to_pixar_inputs(args, batch: dict, tokenizer: transformers.AutoTokenizer, set_mc_choices=False):
    input_dict = {
        'images': batch['images'].to(args.device),
        'images_clip': batch['images_clip'].to(args.device),
        "input_ids": batch['input_ids'].to(args.device),
        'resize_list': batch['resize_list'],
        'label_list': batch['label_list'],
        'attention_masks': batch['attention_masks'].to(args.device),
        'max_new_tokens': args.model_max_length,
        'tokenizer': tokenizer,
        # 'prompt_user_input': True
    }
    return input_dict


@torch.no_grad()
def validate_search(
    val_loader,
    model,
    tokenizer,
    args,
    parser,
    *,
    search_pipeline,
    search_k: int = 5,
    vis_save_path: str = "./vis_output",
    max_new_tokens: int = 1024
    ):
    '''
    During decoding we trigger search API once we see <search>...</search> spans:
      - <img> - search with whole image path
      - <region> - predict region mask for the span immediately before <search>, crop, then search crop
      - (text) - search with the text payload inside <search>…</search>
    Inject <information>{json}</information> back into the decode stream and continues.
    '''
    os.makedirs(vis_save_path, exist_ok=True)
    Path(args.predictions_path).mkdir(parents=True, exist_ok=True)
    pred_file_path = os.path.join(
        args.predictions_path,
        f'predictions_{args.val_dataset.split("|")[0]}_{args.val_dataset.split("|")[1]}_{args.llm_model}_{args.use_llm_as_judge}_search_api_{args.search_api}.jsonl'
    )

    def _generate_with_tools_for_batch(batch):
        device = next(model.parameters()).device  # model’s device
        images_clip = batch['images_clip'].to(device)
        images_for_sam = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_masks'].to(device)

        img_path = batch['image_paths'][0]
        img_path_abs = os.path.abspath(img_path)
        img_bgr = cv2.imread(img_path_abs, cv2.IMREAD_COLOR)

        if img_bgr is None:
            image_np_rgb = np.zeros((224,224,3), dtype=np.uint8)
        else:
            image_np_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        eos_id = tokenizer.eos_token_id
        SEARCH_START_IDS = torch.tensor(tokenizer.encode(SEARCH_START_TOKEN, add_special_tokens=False)).cpu().long()
        SEARCH_END_IDS   = torch.tensor(tokenizer.encode(SEARCH_END_TOKEN,   add_special_tokens=False)).cpu().long()
        IMG_IDS          = torch.tensor(tokenizer.encode(DEFAULT_IMG_PAYLOAD_TOKEN, add_special_tokens=False)).cpu().long()
        REGION_IDS       = torch.tensor(tokenizer.encode(DEFAULT_REGION_TOKEN,     add_special_tokens=False)).cpu().long()

        generated = input_ids.clone()
        steps = max_new_tokens
        max_searches = 5
        search_count = 0
        base_len = generated.shape[1]

        # Track open <search> starts (absolute indices) and last handled end
        open_search_starts: list[int] = [] # stack of absolute start positions
        last_handled_end_abs = base_len  # absolute index of last processed </search>

        # Logits guardrail
        guard = SearchTagGuard(
            start_ids=list(SEARCH_START_IDS.tolist()),
            end_ids=list(SEARCH_END_IDS.tolist()),
            region_ids=list(REGION_IDS.tolist()),
        )

        # stream one token at a time; inject <information>…</information> when a <search>…</search> closes
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
                # logits_processor=[guard],  # guardrail - TODO: test afterwards
            )
            generated = out.sequences
            steps -= 1

            if generated[0, -1].item() == eos_id:
                break

            gen_ids = generated[0, base_len:].detach().cpu().long()
            gen_text = tokenizer.decode(tokenizer_decode_special_tokens(gen_ids, tokenizer), skip_special_tokens=False)
            print(f">> step: {steps}, new_tok: {gen_text}")

            # did we just emit a <search> start?
            just_opened = (
                gen_ids.numel() >= len(SEARCH_START_IDS) and
                torch.equal(gen_ids[-len(SEARCH_START_IDS):], SEARCH_START_IDS)
            )
            if just_opened:
                open_search_starts.append(base_len + gen_ids.numel() - len(SEARCH_START_IDS))
                # guard.on_open()

            # did we just emit a </search> end?
            just_closed = (
                gen_ids.numel() >= len(SEARCH_END_IDS)
                and torch.equal(gen_ids[-len(SEARCH_END_IDS):], SEARCH_END_IDS)
            )
            if not just_closed:
                continue

            # Only process if there is a matching, still-open start
            if not open_search_starts:
                # stray </search>; ignore (guard already discourages it)
                continue

            # Absolute end for this close
            end_abs = base_len + gen_ids.numel()
            if end_abs <= last_handled_end_abs:
                continue  # already handled (safety)

            # Match with the most recent opener
            start_abs = open_search_starts.pop()

            # extra safety: ensure the start comes after last handled close
            if start_abs <= last_handled_end_abs:
                # inconsistent history; skip handling
                # guard.on_close()  # keep counter consistent
                last_handled_end_abs = end_abs
                continue

            print("*** SEARCH_TOKEN DETECTED ***")  # TODO: Remove after debugging

            # Build payload by slicing EXACTLY this span
            s_rel = start_abs - base_len
            e_rel = end_abs - base_len
            payload_ids = gen_ids[s_rel + len(SEARCH_START_IDS) : e_rel - len(SEARCH_END_IDS)]
            payload_text = tokenizer.decode(
                tokenizer_decode_special_tokens(payload_ids, tokenizer),
                skip_special_tokens=False
            )
            print(f">> step: {steps}, payload_text: {payload_text}")

            # breakpoint()  # TODO: Check if tokenizer '<search>' token is handled differently from '</search>'

            # Mark handled and close one open scope
            last_handled_end_abs = end_abs
            # guard.on_close()

            # Decide query type
            is_img    = (payload_ids.tolist() == IMG_IDS.tolist()) or (payload_text.strip().lower() == DEFAULT_IMG_PAYLOAD_TOKEN.lower())
            is_region = (payload_ids.tolist() == REGION_IDS.tolist()) or (payload_text.strip().lower() == DEFAULT_REGION_TOKEN.lower())
            is_text   = not is_img and not is_region

            if is_img:
                input_query = [img_path_abs]
            elif is_region:
                # take up to N words immediately before "<search>" as the search_span (same heuristic as chat code)
                mask_span_text = last_span_before(gen_text, SEARCH_START_TOKEN, max_words=10)
                mask_span_tok  = torch.tensor(tokenizer.encode(mask_span_text, add_special_tokens=False), dtype=gen_ids.dtype)

                loc = find_sub_ids(generated[0, base_len:], mask_span_tok)
                if loc is None:
                    input_query = [img_path_abs]  # fallback
                else:
                    s, e = loc
                    gt_bio = torch.zeros((1, gen_ids.numel()), dtype=torch.long, device=device)
                    if e - s >= 1:
                        gt_bio[0, s] = 1  # B
                        if e - s > 1:
                            gt_bio[0, s+1:e] = 2  # I

                    # (h′, w′) BEFORE SAM padding (your dataloader stores it here)
                    input_size = tuple(batch['resize_list'][0])  # e.g., (684, 1024)
                    orig_size  = (tuple(batch['label_list'][0]) if isinstance(batch['label_list'][0], (tuple, list))
                                else tuple(batch['label_list'][0].shape[:2]))

                    ev = model.evaluate(
                        images_clip=images_clip,
                        images=images_for_sam,
                        input_ids=input_ids,
                        resize_list=[input_size],
                        label_list=[orig_size],
                        attention_masks=attention_masks,
                        gt_bio_span=gt_bio,
                        max_new_tokens=64,
                        tokenizer=tokenizer,
                        prompt_user_input=False,
                    )
                    region_path = None
                    try:
                        pm = ev["pred_masks"][0]
                        if len(pm) > 0:
                            mask = pm[0].detach().cpu().numpy()[0] > 0
                            base = os.path.splitext(os.path.basename(img_path_abs))[0]
                            region_path = save_cropped_region(mask, image_np_rgb, vis_save_path, base, idx=int(time.time()))
                            # region_path = save_cropped_region(mask, img_bgr, vis_save_path, base, idx=int(time.time()))
                    except Exception:
                        region_path = None

                    region_path = os.path.abspath(region_path) if region_path is not None else None
                    input_query = [region_path] if region_path is not None else [img_path_abs]
            else:
                input_query = payload_text

            search_count += 1
            if search_count > max_searches:
                # start cooldown to prevent back-to-back searches
                # guard.start_cooldown(steps=8)
                continue

            try:
                results = search_pipeline(input_query, k=search_k)
            except Exception as e:
                results = [{"error": str(e), "input_query": input_query}]

            trunc_results = []
            if results is not None:
                if 'error' in results[0] and results[0]['error'] is not None:
                    payload_text_out = f"_"
                else:
                    if args.retrieve_type == "first_sent":
                        first_sentence_tokenizer = build_first_sentence_fn()
                        payload_contents = [{r['title']: first_sentence_tokenizer(r['content'])} for r in results]
                    elif args.retrieve_type == "title":
                        try:
                            payload_contents = [r['title'] for r in results]  # TODO: Error - r in results seem to be a list; str indexing gives error
                        except KeyError as e:
                            print(e)
                            breakpoint()
                    else:
                        payload_contents = [{r['title']: r['content']} for r in results]

                    payload_text_out = format_payload(payload_contents)

            injected = f"{INFORMATION_START_TOKEN} {payload_text_out} {INFORMATION_END_TOKEN}"
            generated = append_text_to_ids(tokenizer, generated, injected, device)

            # brief cooldown to discourage immediate <search> again
            # guard.start_cooldown(steps=8)

        final_text = tokenizer.decode(tokenizer_decode_special_tokens(generated[0], tokenizer), skip_special_tokens=False)

        print(f">> final_text: {final_text}")

        return final_text

    # val loop
    if args.limit_batches > 0:
        n_batches = min(args.limit_batches, len(val_loader))
    else:
        n_batches = len(val_loader)

    val_iter = iter(val_loader)
    pbar = tqdm.tqdm(range(n_batches), desc="Validating (<search> API calling)")

    for _ in pbar:
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        # dtype conditioning
        if args.precision == "fp16":
            to_dtype = lambda x: x.half()
        elif args.precision == "bf16":
            to_dtype = lambda x: x.bfloat16()
        else:
            to_dtype = lambda x: x.float()
        batch['images']     = to_dtype(batch['images'])
        batch['images_clip'] = to_dtype(batch['images_clip'])

        # run tool-aware generation
        interleaved_text = _generate_with_tools_for_batch(batch)

        # evaluate answer text vs GT like your current validate()
        ground_truth_text = batch['sampled_classes_list'][0]
        if args.use_llm_as_judge:
            score, explanation = evaluate_answer(interleaved_text, ground_truth_text, model=args.llm_model)
            accuracy = score
        else:
            evaluator = TextVQAAccuracyEvaluator()
            accuracy = evaluator.eval_pred_list([{'pred_answer': interleaved_text, 'gt_answers': ground_truth_text}])

        out = {
            "image_path": batch['image_paths'][0],
            "output_text": interleaved_text,
            "ground_truth_text": ground_truth_text,
            "accuracy": accuracy
        }
        with open(pred_file_path, "a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # summarize accuracy
    existing = []
    if os.path.exists(pred_file_path):
        with open(pred_file_path, "r") as f:
            for line in f:
                existing.append(json.loads(line))
        total_acc = sum(d["accuracy"] for d in existing) / max(1, len(existing))
    else:
        total_acc = 0.0
    print(f"==> Total Accuracy : {total_acc:.4f}")


@torch.no_grad()
def validate(val_loader, model, tokenizer, args, parser):
    '''
    Validation function specialized for 'textvqa' QA pairs
    '''
    model.eval()

    skipped_instance_num = 0
    if args.limit_batches > 0:
        n_batches = min(args.limit_batches, len(val_loader))
    else:
        n_batches = len(val_loader)

    Path(args.predictions_path).mkdir(parents=True, exist_ok=True)
    pred_file_path = os.path.join(
        args.predictions_path,
        f'predictions_{args.val_dataset.split("|")[0]}_{args.val_dataset.split("|")[1]}_{args.llm_model}_{args.use_llm_as_judge}.jsonl'
    )

    val_iter = iter(val_loader)
    pbar = tqdm.tqdm(range(n_batches), desc="Validating Partonomy with PLUM")

    total_accuracy, total_f1 = 0.0, 0.0
    output_dicts = []
    for idx in pbar:
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        if args.precision == "fp16":
            to_dtype = lambda x: x.half()
        elif args.precision == "bf16":
            to_dtype = lambda x: x.bfloat16()
        else:
            to_dtype = lambda x: x.float()

        batch['images'] = to_dtype(batch['images'])
        batch['images_clip'] = to_dtype(batch['images_clip'])

        # Forward Pass
        eval_output = model.evaluate(**to_pixar_inputs(args, batch, tokenizer))  # multiple_choice == True only for Partonomy - this nees to be set to return token-level loss

        output_ids = eval_output.get("output_ids", None)
        spans_dicts = eval_output.get("spans_dicts", None)
        pred_masks = eval_output.get("pred_masks", None)
        ce_losses = eval_output.get("ce_losses", None)
        ground_truth_text = batch['sampled_classes_list'][0]

        if len(pred_masks[0]) > 0:
            pred_masks_flattened = torch.cat([(pred_mask > 0).int() for pred_mask in pred_masks[0]], dim=0)
        else:
            pred_masks_flattened = []

        # stack the pred_masks (list of tensors) into a single tensor
        if isinstance(pred_masks_flattened, list) and len(pred_masks_flattened) == 0:
            # no masks predicted
            H, W = batch['label_list'][0].shape
            pred_masks = torch.zeros((1, H, W), dtype=batch["images"].dtype, device=batch["images"].device)
            pred_texts = []
        else:
            pred_masks = pred_masks_flattened
            pred_texts = spans_dicts[0]['span_texts']

        pred_masks_np = np.stack([m.cpu().float().numpy() for m in pred_masks])

        print(f">> pred_masks.shape: {pred_masks_np.shape}")
        print(f">> pred_texts: {pred_texts}")

        # get the output_ids and output_text and evaluate against the ground truth answer texts
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

        if args.use_llm_as_judge:
            score, explanation = evaluate_answer(output_text, ground_truth_text, model=args.llm_model)
            accuracy = score
            print(f">> score: {score}, explanation: {explanation}")
        else:
            evaluator = TextVQAAccuracyEvaluator()
            accuracy = evaluator.eval_pred_list([{'pred_answer': output_text, 'gt_answers': ground_truth_text}])

        logger.info(f"==> Output Text: {output_text} | Ground Truth Text: {ground_truth_text} | Accuracy: {accuracy}")

        total_accuracy += accuracy

        output_dict = {
            "image_path": batch['image_paths'][0],
            "output_text": output_text,
            "ground_truth_text": ground_truth_text,
            "pred_masks": get_mask_rle_dicts(pred_masks_np, pred_texts),
            "accuracy": accuracy
        }
        output_dicts.append(output_dict)

        with open(pred_file_path, "a") as f:
            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")

    existing_output_dicts = []
    if os.path.exists(pred_file_path):
        # read and calculate the total_accuracy
        with open(pred_file_path, 'r') as f:
            for line in f:  # we need to read line-by-line since it's saved as a jsonl file
                existing_output_dicts.append(json.loads(line))

        total_accuracy = sum([d['accuracy'] for d in existing_output_dicts]) / len(existing_output_dicts)
        print(f"==> Total Accuracy: {total_accuracy}")
    else:
        total_accuracy = total_accuracy / n_batches
        print(f"==> Total Accuracy: {total_accuracy}")
        with open(pred_file_path, 'w') as f:
            json.dump(output_dicts, f)

    logger.info(f">> [validate_vqa] Skipped instances: {skipped_instance_num}")


if __name__ == "__main__":
    coloredlogs.install(level='DEBUG')

    main(sys.argv[1:])
