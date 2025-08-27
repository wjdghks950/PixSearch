import jsonargparse as argparse
import json
import os
import sys
import numpy as np

import torch
import tqdm
import transformers
from itertools import islice, chain
from transformers import BitsAndBytesConfig
from functools import partial
from pathlib import Path

from model.PLUM import PLUMForCausalLM

from utils.vqa_dataset import TextVQADataset, GQADataset, POPEDataset, CragMMDataset
from utils.m4c_evaluator import TextVQAAccuracyEvaluator
from utils.dataset import collate_fn
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN
)
# add IoUEvaluator evaluation functionality
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
from evaluation.rle_dict import get_mask_rle_dicts
from evaluation.llm_as_judge_eval import evaluate_answer

import logging
import coloredlogs

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args(args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser(description="Validate PLUM on VQA datasets")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--workers", default=0, type=int, help="number of workers")
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--backbone", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview", type=str)
    parser.add_argument(
        "--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"],
        help="precision for inference"
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
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
        choices=["llava_v1", "llava_llama_2"],
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
        use_fast=True,
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

    model = PLUMForCausalLM.from_pretrained(
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
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

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
    validate(val_loader, model, tokenizer, args, parser)


def to_plum_inputs(args, batch: dict, tokenizer: transformers.AutoTokenizer, set_mc_choices=False):
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
        f'predictions_{args.val_dataset.split("|")[0]}_{args.llm_model}_{args.use_llm_as_judge}.jsonl'
    )

    val_iter = iter(val_loader)
    pbar = tqdm.tqdm(range(n_batches), desc="Validating Partonomy with PLUM")

    total_accuracy = 0
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
        eval_output = model.evaluate(**to_plum_inputs(args, batch, tokenizer))  # multiple_choice == True only for Partonomy - this nees to be set to return token-level loss
        
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
