import argparse
import os
import sys
import re
import json
import time
import tempfile
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.PIXAR import PIXARForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
    SEARCH_START_TOKEN, SEARCH_END_TOKEN,
    INFORMATION_START_TOKEN, INFORMATION_END_TOKEN,
    DEFAULT_REGION_TOKEN
)

from src.local_search_api import LocalSearchPipeline  # TODO: Revive after fixing the path error


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)


def parse_args(args):
    p = argparse.ArgumentParser(description="PIXAR chat with tool interleaving")
    p.add_argument("--version", default="/home/jeonghkim/projects/pixar/pixar/src/model/runs/pixar-13b_kld_0.1_focal_tversky_8_pixar_ft/merged_model")
    p.add_argument("--vis_save_path", default="./vis_output", type=str)
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--image_size", default=1024, type=int)
    p.add_argument("--model_max_length", default=512, type=int)
    p.add_argument("--lora_r", default=8, type=int)
    p.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    p.add_argument("--local-rank", default=0, type=int)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--use_mm_start_end", action="store_true", default=True)
    p.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2"])
    p.add_argument("--use_bidir_bio", action="store_true", default=False)
    p.add_argument("--use_feedback_loop", action="store_true", default=False)
    p.add_argument("--prompt_user_input", action="store_true", default=False)

    # Search pipeline knobs
    p.add_argument("--search_k", type=int, default=5, help="top-k for search tool")
    p.add_argument("--search_base_url", type=str, default="http://127.0.0.1:7000")
    p.add_argument("--search_collection", type=str, default="wiki6m_index")
    p.add_argument("--search_index_name", type=str, default=None)
    p.add_argument("--search_pooling", type=str, default="mean")

    return p.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


# helpers for interrupted decoding after <search> token
def _decode_tail(tokenizer, ids: torch.Tensor, k: int = 256) -> str:
    '''
    helper to decode only the most recent window of tokens instead of the entire prompt
    '''
    tail = ids[:, max(0, ids.shape[1] - k):]
    return tokenizer.decode(tail[0], skip_special_tokens=False)

def _append_text_to_ids(tokenizer, input_ids: torch.Tensor, text: str, device) -> torch.Tensor:
    new_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
    return torch.cat([input_ids, new_ids], dim=1)

def _extract_closed_payload(text: str, open_tag: str, close_tag: str) -> Optional[Tuple[str, int, int]]:
    '''
    Return (payload, open_idx, close_end_idx) for the LAST closed pair, or None.
    '''
    open_idx = text.rfind(open_tag)
    if open_idx == -1:
        return None
    close_idx = text.find(close_tag, open_idx + len(open_tag))
    if close_idx == -1:
        return None
    return text[open_idx + len(open_tag):close_idx].strip(), open_idx, close_idx + len(close_tag)

def _last_span_before(text: str, anchor: str, max_words: int = 8) -> str:
    idx = text.rfind(anchor)
    if idx == -1:
        return ""
    left = text[:idx].strip()
    toks = re.split(r"\s+", left)
    return " ".join(toks[-max_words:]).strip()

def _find_subsequence(hay: torch.Tensor, needle: torch.Tensor) -> Optional[Tuple[int, int]]:
    '''
    Return [start, end) indices of first occurrence of `needle` in `hay`.
    Both are 1D Long tensors.
    '''
    if needle.numel() == 0 or hay.numel() < needle.numel():
        return None
    # naive search (seq length small for generated tails)
    H, N = hay.tolist(), needle.tolist()
    for i in range(len(H) - len(N) + 1):
        if H[i:i+len(N)] == N:
            return (i, i+len(N))
    return None

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


def save_cropped_region(mask_bool: np.ndarray, rgb_image: np.ndarray, out_dir: str, base_name: str, idx: int) -> str:
    '''
    Args:
        mask_bool - HxW boolean array (True where foreground)
        rgb_image - HxWx3 uint8 RGB
    Returns:
        out_path - path to the saved crop (PNG).
    '''
    H, W = mask_bool.shape
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        # fall back to whole image if mask is empty
        out_path = os.path.join(_ensure_dir(out_dir), f"{base_name}_region_{idx}_fallback.png")
        cv2.imwrite(out_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        return out_path

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    crop = rgb_image[y1:y2+1, x1:x2+1, :]
    out_path = os.path.join(_ensure_dir(out_dir), f"{base_name}_region_{idx}.png")
    cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return out_path


def generate_with_tools(
    model: PIXARForCausalLM,
    tokenizer,
    images_clip: torch.Tensor,
    image_tensor_for_seg: torch.Tensor,  # preprocessed sam input (B=1, C,H,W)
    image_np_rgb: np.ndarray, # original RGB image
    image_path: str,
    input_ids: torch.Tensor,
    attention_masks: Optional[torch.Tensor],
    *,
    search_pipeline: LocalSearchPipeline,
    search_k: int = 5,
    vis_save_path: str = "./vis_output",
    max_new_tokens: int = 512,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, str]:
    '''
    Streams generation. When `<search> ... </search>` appears, detects payload:
      - <image>: search_pipeline([image_path], k)
      - <region>: build gt_bio_span from span immediately before <search>, call model.evaluate to get a mask,
                     crop region image, search_pipeline([cropped_path], k)
      - <text>: search_pipeline(payload_text, k)

    Injects `<information> {json} </information>` back into the decode stream.
    '''
    model.eval()
    device = device or input_ids.device
    eos_id = tokenizer.eos_token_id

    generated = input_ids.clone()
    steps = max_new_tokens
    max_searches = 12
    search_count = 0

    # track only the generated segment (excluding the prompt ids)
    base_len = generated.shape[1]

    while steps > 0:
        with torch.no_grad():
            out = model.generate(
                images=images_clip,
                input_ids=generated,
                max_new_tokens=1,
                num_beams=1,
                do_sample=False,
                use_cache=False,
                return_dict_in_generate=True,
                output_hidden_states=False,
            )
        generated = out.sequences
        steps -= 1
        new_tok = generated[0, -1].item()
        if new_tok == eos_id:
            break

        # check for closed <search> ... </search> in tail
        tail_txt = _decode_tail(tokenizer, generated, k=512)
        payload = _extract_closed_payload(tail_txt, SEARCH_START_TOKEN, SEARCH_END_TOKEN)
        if payload is None:
            continue

        payload_text, _, _ = payload
        search_count += 1
        if search_count > max_searches:
            continue

        # decide query kind
        payload_norm = payload_text.strip().lower()

        # Compute query and (optionally) region crop
        input_query: Any
        if payload_norm == DEFAULT_IMAGE_TOKEN:
            # whole image
            input_query = [image_path]

        elif payload_norm == DEFAULT_REGION_TOKEN:
            # Build gt_bio_span from the span immediately preceding "<search>"
            # 1) Recover generated-only ids (exclude prompt)
            gen_ids = generated[0, base_len:].detach().cpu()

            # 2) Find the last span text before "<search>"
            span_text = _last_span_before(tail_txt, SEARCH_START_TOKEN, max_words=8)
            span_ids = torch.tensor(tokenizer.encode(span_text, add_special_tokens=False), dtype=gen_ids.dtype)

            # 3) Locate the span in generated ids
            loc = _find_subsequence(gen_ids, span_ids)
            if loc is None:
                # if not found, fallback to whole image
                input_query = [image_path]
            else:
                s, e = loc  # on gen_ids
                # 4) Build gt_bio_span over the *generated* segment length
                gt_bio = torch.zeros((1, gen_ids.numel()), dtype=torch.long, device=device)
                if e - s >= 1:
                    gt_bio[0, s] = 1  # B
                    if e - s > 1:
                        gt_bio[0, s+1:e] = 2  # I

                # 5) Call evaluate() to get mask(s) for that span
                # NOTE: evaluate() will re-generate, but we only need masks
                outputs = model.evaluate(
                    images_clip=images_clip,
                    images=image_tensor_for_seg,
                    input_ids=input_ids,                       # full prompt ids
                    resize_list=[image_np_rgb.shape[:2]],
                    label_list=[image_np_rgb],                 # shape is inspected; tuple(shape) is taken internally
                    attention_masks=attention_masks,
                    gt_bio_span=gt_bio,
                    max_new_tokens=64,                         # minimal regen
                    tokenizer=tokenizer,
                    prompt_user_input=False,
                )
                # 6) Extract first mask and save a cropped region file
                region_path = None
                try:
                    pm = outputs["pred_masks"][0]
                    if len(pm) > 0:
                        mask = pm[0].detach().cpu().numpy()[0] > 0
                        base = os.path.splitext(os.path.basename(image_path))[0]
                        region_path = save_cropped_region(mask, image_np_rgb, vis_save_path, base, idx=int(time.time()))
                except Exception:
                    region_path = None

                if region_path is None:
                    input_query = [image_path]
                else:
                    input_query = [region_path]

        else:
            # textual query
            input_query = payload_text

        # search tool call
        try:
            results = search_pipeline(input_query, k=search_k)
        except Exception as e:
            results = [{"error": str(e), "input_query": input_query}]

        injected = f"{INFORMATION_START_TOKEN} {json.dumps(results, ensure_ascii=False)} {INFORMATION_END_TOKEN}"
        generated = _append_text_to_ids(tokenizer, generated, injected, device)

    final_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    return generated, final_text


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # model init
    kwargs = {
        "torch_dtype": torch_dtype,
        "use_bidir_bio": args.use_bidir_bio,
        "use_feedback_loop": args.use_feedback_loop,
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

    model = PIXARForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=False,
        vision_tower=args.vision_tower,
        **kwargs
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower().to(dtype=torch_dtype)

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

    vision_tower = model.get_model().get_vision_tower().to(device=args.local-rank if hasattr(args, "local-rank") else args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    # local search API setup
    search_pipeline = LocalSearchPipeline(
        base_url=args.search_base_url,
        collection=args.search_collection,
        index_name=args.search_index_name,
        pooling=args.search_pooling,
    )

    model.eval()

    while True:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        prompt = input("Please input your prompt: ").strip()
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = input("Please input the image path: ").strip()
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue

        # load and process image
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image_for_sam = transform.apply_image(image_np)
        resize_list = [image_for_sam.shape[:2]]
        image_tensor_for_seg = preprocess(torch.from_numpy(image_for_sam).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda()
        if args.precision == "bf16":
            image_tensor_for_seg = image_tensor_for_seg.bfloat16()
        elif args.precision == "fp16":
            image_tensor_for_seg = image_tensor_for_seg.half()
        else:
            image_tensor_for_seg = image_tensor_for_seg.float()

        # tokens
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long()

        # === tool-aware generation ===
        final_ids, final_text = generate_with_tools(
            model=model,
            tokenizer=tokenizer,
            images_clip=image_clip,
            image_tensor_for_seg=image_tensor_for_seg,
            image_np_rgb=image_np,
            image_path=image_path,
            input_ids=input_ids,
            attention_masks=attention_masks,
            search_pipeline=search_pipeline,
            search_k=args.search_k,
            vis_save_path=args.vis_save_path,
            max_new_tokens=512,
            device=input_ids.device,
        )

        print("\n********\n>> PIXAR Output (interleaved):")
        print(final_text.replace("\n", " ").replace("  ", " "))
        print("********\n")

        # Optionally save the raw output json
        out_json_path = os.path.join(args.vis_save_path, os.path.splitext(os.path.basename(image_path))[0] + "_interleaved.json")
        os.makedirs(args.vis_save_path, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump({"text": final_text}, f)
        print(f"Saved interleaved text to {out_json_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
