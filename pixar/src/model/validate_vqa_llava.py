import jsonargparse as argparse
import json
import os
import sys
import torch
import tqdm
import transformers
from transformers import BitsAndBytesConfig
from functools import partial

from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from utils.vqa_dataset import TextVQADataset, GQADataset, POPEDataset
from utils.m4c_evaluator import TextVQAAccuracyEvaluator
from utils.dataset import collate_fn
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN
)
import logging
import coloredlogs

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args(args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser(description="Validate LLaVA on VQA datasets")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--workers", default=0, type=int, help="number of workers")
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument(
        "--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"],
        help="precision for inference"
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--pretrain_mm_mlp_adapter", default=None, type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--val_dataset", default="textvqa|val", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument('--limit_batches', default=0, type=int)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--predictions_path', type=str, default='predictions.json')
    parser.add_argument("--exp_name", default="llava-13b-vqa-eval", type=str)

    args = parser.parse_string(config_str) if config_str else parser.parse_args(args)
    return args, parser


def main(args: list[str] = None, config_str: str = None):
    args, parser = parse_args(args, config_str)

    print(f"*(validate_vqa_llava) >> Loading LLaVA model from [ {args.version} ]*")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Build LLaVA model
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
        
    kwargs = {"torch_dtype": torch_dtype}
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

    model = LlavaLlamaForCausalLM.from_pretrained(
        args.version, 
        low_cpu_mem_usage=True,
        **kwargs
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vision_tower = args.vision_tower
    model.config.pretrain_mm_mlp_adapter = args.pretrain_mm_mlp_adapter
    model.config.mm_use_im_start_end = args.use_mm_start_end
    # Initialize vision modules
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

    # Build dataset for validation
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

    # Build DataLoader
    if val_dataset is not None:
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

    # Run evaluation
    validate(val_loader, model, tokenizer, args)


def to_llava_inputs(args, batch: dict, tokenizer: transformers.AutoTokenizer, dtype: torch.dtype):
    input_dict = {
        'images': batch['images_clip'].to(args.device, dtype=dtype),
        'input_ids': batch['input_ids'].to(args.device),
        'attention_mask': batch['attention_masks'].to(args.device),
    }
    return input_dict


@torch.no_grad()
def validate(val_loader, model, tokenizer, args):
    '''
    Validation function for VQA evaluation
    '''
    model.eval()

    if args.limit_batches > 0:
        n_batches = min(args.limit_batches, len(val_loader))
    else:
        n_batches = len(val_loader)

    val_iter = iter(val_loader)
    pbar = tqdm.tqdm(range(n_batches), desc="Validating with LLaVA")

    total_accuracy = 0
    output_dicts = []
    for idx in pbar:
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        if args.precision == "fp16":
            to_dtype = lambda x: x.half()
            dtype = torch.float16
        elif args.precision == "bf16":
            to_dtype = lambda x: x.bfloat16()
            dtype = torch.bfloat16
        else:
            to_dtype = lambda x: x.float()
            dtype = torch.float32
            
        # Forward Pass
        outputs = model(**to_llava_inputs(args, batch, tokenizer, dtype))
        logits = outputs.logits
        
        # Get predicted tokens
        pred_tokens = torch.argmax(logits, dim=-1)
        
        # Convert to text
        output_text = tokenizer.decode(pred_tokens[0], skip_special_tokens=True)
        ground_truth_text = batch['sampled_classes_list'][0]
        
        # Evaluate using TextVQA accuracy metric
        evaluator = TextVQAAccuracyEvaluator()
        accuracy = evaluator.eval_pred_list([{'pred_answer': output_text, 'gt_answers': ground_truth_text}])
        logger.info(f"==> Output Text: {output_text} | Ground Truth Text: {ground_truth_text} | Accuracy: {accuracy}")
        
        total_accuracy += accuracy
        
        output_dict = {
            "image_path": batch['image_paths'][0],
            "output_text": output_text,
            "ground_truth_text": ground_truth_text,
            "accuracy": accuracy
        }
        output_dicts.append(output_dict)
        
    # Save predictions
    os.makedirs(args.predictions_path, exist_ok=True)
    pred_file_path = os.path.join(args.predictions_path, 'predictions.json')
    
    if os.path.exists(pred_file_path):
        # Read and update existing predictions
        with open(pred_file_path, 'r') as f:
            existing_output_dicts = json.load(f)
        existing_output_dicts.extend(output_dicts)
        total_accuracy = sum([d['accuracy'] for d in existing_output_dicts]) / len(existing_output_dicts)
    else:
        total_accuracy = total_accuracy / n_batches
        with open(pred_file_path, 'w') as f:
            json.dump(output_dicts, f)
            
    print(f"==> Total Accuracy: {total_accuracy}")


if __name__ == "__main__":
    coloredlogs.install(level='DEBUG')
    main(sys.argv[1:]) 