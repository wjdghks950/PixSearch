import argparse
import os
import sys
import re
import json
import time
import tempfile
from typing import Optional, Dict, Any, Tuple, List
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, BitsAndBytesConfig,
    CLIPImageProcessor, LogitsProcessor
)
from model.PIXAR import PIXARForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
    SEARCH_START_TOKEN, SEARCH_END_TOKEN,
    INFORMATION_START_TOKEN, INFORMATION_END_TOKEN,
    DEFAULT_REGION_TOKEN, DEFAULT_IMG_PAYLOAD_TOKEN,
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.local_search_api import LocalSearchPipeline


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)


def parse_args(args):
    p = argparse.ArgumentParser(description="PIXAR chat with tool interleaving")
    p.add_argument("--version", default="./runs/pixar-13b_kld_0.1_focal_tversky_8_pixar_ft/merged_model")
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
    p.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2", "pixar_0shot_v1"])
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

class SearchTagGuard(LogitsProcessor):
    """
    Guardrail over tool tags:
      - Ban emitting the first token of </search> when no open <search>
      - After injecting <information>...</information>, briefly ban starting <search> again (cooldown)
      - Optionally ban <region> unless we are inside an open <search> payload
    """
    def __init__(self, start_ids: list[int], end_ids: list[int], region_ids: list[int]):
        super().__init__()
        self.start_ids = list(map(int, start_ids))
        self.end_ids   = list(map(int, end_ids))
        self.region_ids = list(map(int, region_ids)) if region_ids is not None else []
        self.open_count = 0
        self.cooldown = 0  # number of decoding steps to ban <search> start

    def on_open(self):
        self.open_count += 1

    def on_close(self):
        self.open_count = max(0, self.open_count - 1)

    def start_cooldown(self, steps=8):
        self.cooldown = steps

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Ban starting </search> if there is no open <search>
        if self.open_count == 0 and len(self.end_ids) > 0:
            scores[:, self.end_ids[0]] = -float("inf")

        # Briefly ban starting a new <search>
        if self.cooldown > 0 and len(self.start_ids) > 0:
            scores[:, self.start_ids[0]] = -float("inf")
            self.cooldown -= 1

        # Ban <region> unless inside a <search> payload (i.e., open)
        if self.open_count == 0 and len(self.region_ids) > 0:
            scores[:, self.region_ids[0]] = -float("inf")

        return scores

def tokenizer_decode_special_tokens(ids: torch.Tensor, tokenizer) -> List[int]:
    # map negative/sentinel ids to decodable ids for tokenizer.decode.
    ids = ids.clone()
    unk = getattr(tokenizer, "unk_token_id", None) or 0
    ids[ids == IMAGE_TOKEN_INDEX] = unk
    if (ids < 0).any():
        return [int(x if x >= 0 else unk) for x in ids.tolist()]
    return [int(x) for x in ids.tolist()]

def append_text_to_ids(tokenizer, input_ids: torch.Tensor, text: str, device) -> torch.Tensor:
    new_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
    return torch.cat([input_ids, new_ids], dim=1)

def find_sub_ids(hay: torch.Tensor, needle: torch.Tensor, start_idx: int = 0) -> Optional[Tuple[int,int]]:
    # return [start, end) indices of first occurrence of needle in hay[start_idx:]
    if needle.numel() == 0:
        return None
    H = hay.tolist()
    N = needle.tolist()
    for i in range(start_idx, len(H) - len(N) + 1):
        if H[i:i+len(N)] == N:
            return (i, i + len(N))
    return None

def find_last_closed_span_ids(hay: torch.Tensor, start_ids: torch.Tensor, end_ids: torch.Tensor) -> Optional[Tuple[int,int,int,int]]:
    '''
    In 1D ids 'hay', find the LAST closed <start> ... <end> pair.
    Returns (s0, s1, e0, e1) where:
      hay[s0:s1] == start_ids, hay[e0:e1] == end_ids, and s1 <= e0
    '''
    H = hay.tolist()
    S = start_ids.tolist()
    E = end_ids.tolist()

    # find last end
    last_end_start = -1
    for i in range(len(H) - len(E), -1, -1):
        if H[i:i+len(E)] == E:
            last_end_start = i
            break
    if last_end_start == -1:
        return None
    e0, e1 = last_end_start, last_end_start + len(E)

    # find nearest start before that end
    nearest_s0 = -1
    for i in range(e0 - len(S), -1, -1):
        if H[i:i+len(S)] == S:
            nearest_s0 = i
            break
    if nearest_s0 == -1:
        return None
    s0, s1 = nearest_s0, nearest_s0 + len(S)

    if s1 > e0:
        return None
    return (s0, s1, e0, e1)

def last_span_before(text: str, anchor: str, max_words: int = 8) -> str:
    # Get up to N words immediately before `anchor` for region-derived queries
    idx = text.rfind(anchor)
    if idx == -1:
        return ""
    left = text[:idx].strip()
    tokens = re.split(r"\s+", left)
    return " ".join(tokens[-max_words:]).strip()

def save_cropped_region(mask_bool: np.ndarray, rgb_image: np.ndarray, out_dir: str, base_name: str, idx: int) -> str:
    """
    Save the cropped region (tight bbox over the mask) AND a full-image overlay with the mask.
    Returns only the crop path (not the overlay path).

    Args:
        mask_bool : (H, W) boolean array, True where foreground.
        rgb_image : (H, W, 3) uint8 RGB image.
        out_dir   : output directory.
        base_name : base filename stem.
        idx       : integer suffix for uniqueness.
    """
    os.makedirs(out_dir, exist_ok=True)

    H, W = mask_bool.shape
    mask_u8 = (mask_bool.astype(np.uint8) * 255)

    # ---------- save overlay on the whole image ----------
    # semi-transparent fill + green contour
    overlay_rgb = rgb_image.copy()
    if mask_u8.any():
        alpha = 0.45
        tint_rgb = np.array([255, 0, 0], dtype=np.uint8)  # red tint in RGB
        # apply tint only on masked pixels
        m = mask_bool
        overlay_rgb[m] = (overlay_rgb[m].astype(np.float32) * (1.0 - alpha) + tint_rgb.astype(np.float32) * alpha).astype(np.uint8)

        # draw contour for clarity
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, cnts, -1, (0, 255, 0), 2)  # green contour
    else:
        # no foreground: just save original as overlay
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    overlay_path = os.path.join(out_dir, f"{base_name}_overlay_{idx}.png")
    cv2.imwrite(overlay_path, overlay_bgr)

    # ---------- save cropped region (fallback to whole image if empty) ----------
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        crop_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        crop_path = os.path.join(out_dir, f"{base_name}_region_{idx}_fallback.png")
        cv2.imwrite(crop_path, crop_bgr)
        return crop_path

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    crop_rgb = rgb_image[y1:y2 + 1, x1:x2 + 1, :]
    crop_path = os.path.join(out_dir, f"{base_name}_region_{idx}.png")
    cv2.imwrite(crop_path, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))

    return crop_path


def generate_with_tools(
    model: PIXARForCausalLM,
    tokenizer,
    images_clip: torch.Tensor,
    image_tensor_for_seg: torch.Tensor,  # preprocessed sam input (bsz=1, c,h,w)
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
    image_resize_hw: Tuple[int, int] = None,
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
    SEARCH_START_IDS = torch.tensor(tokenizer.encode(SEARCH_START_TOKEN, add_special_tokens=False)).detach().cpu().long()
    SEARCH_END_IDS = torch.tensor(tokenizer.encode(SEARCH_END_TOKEN, add_special_tokens=False)).detach().cpu().long()
    IMG_IDS = torch.tensor(tokenizer.encode(DEFAULT_IMG_PAYLOAD_TOKEN, add_special_tokens=False)).detach().cpu().long()
    REGION_IDS = torch.tensor(tokenizer.encode(DEFAULT_REGION_TOKEN, add_special_tokens=False)).detach().cpu().long()

    generated = input_ids.clone()
    steps = max_new_tokens
    max_searches = 5
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

        gen_ids = generated[0, base_len:].detach().cpu().long()
        gen_text = tokenizer.decode(tokenizer_decode_special_tokens(gen_ids, tokenizer), skip_special_tokens=False)
        search_span = find_last_closed_span_ids(gen_ids, SEARCH_START_IDS, SEARCH_END_IDS)

        print(f">> step: {steps}, new_tok: {gen_text}")

        if search_span is None:
            continue

        s0, s1, e0, e1 = search_span
        payload_ids = gen_ids[s1:e0]
        payload_text = tokenizer.decode(tokenizer_decode_special_tokens(payload_ids, tokenizer), skip_special_tokens=False)

        print(f">> step: {steps}, payload_text: {payload_text}")

        is_image_payload = (payload_ids.tolist() == IMG_IDS.tolist()) or (payload_text == DEFAULT_IMG_PAYLOAD_TOKEN.lower())
        is_region_payload = (payload_ids.tolist() == REGION_IDS.tolist()) or (payload_text == DEFAULT_REGION_TOKEN.lower())
        is_text_payload = not is_image_payload and not is_region_payload

        if is_image_payload:
            input_query = [image_path]
        elif is_region_payload:
            # NOTE: This requires sampling mask from the model for the text span immediately before "<search>" or model sampled text span
            # build gt_bio_span from the span immediately preceding "<search>"
            gen_ids = generated[0, base_len:].detach().cpu()

            # find the last span text before "<search>"
            span_text = last_span_before(gen_text, SEARCH_START_TOKEN, max_words=5)
            span_ids = torch.tensor(tokenizer.encode(span_text, add_special_tokens=False), dtype=gen_ids.dtype)

            # locate the span in generated ids
            loc = find_sub_ids(generated[0, base_len:], span_ids)
            if loc is None:
                input_query = [image_path]  # if not found, fallback to whole image
            else:
                s, e = loc  # on gen_ids
                # build gt_bio_span over the generated segment length
                gt_bio = torch.zeros((1, gen_ids.numel()), dtype=torch.long, device=device)
                if e - s >= 1:
                    gt_bio[0, s] = 1  # B
                    if e - s > 1:
                        gt_bio[0, s+1:e] = 2  # I

                # call evaluate() to get mask(s) for that span
                # NOTE: evaluate() will re-generate, but we only need masks
                outputs = model.evaluate(
                    images_clip=images_clip,
                    images=image_tensor_for_seg,
                    input_ids=input_ids,
                    resize_list=[image_resize_hw],
                    label_list=[image_np_rgb.shape[:2]],
                    attention_masks=attention_masks,
                    gt_bio_span=gt_bio,
                    max_new_tokens=64,  # NOTE: set minimal text span length for region crop;
                    tokenizer=tokenizer,
                    prompt_user_input=False,
                )
                # extract first mask and save a cropped region file
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

        elif is_text_payload:
            input_query = payload_text
        else:
            raise ValueError(f"Unknown payload type: {payload_text}")

        search_count += 1
        if search_count > max_searches:
            continue

        # search tool call
        try:
            results = search_pipeline(input_query, k=search_k)
        except Exception as e:
            results = [{"error": str(e), "input_query": input_query}]

        injected = f"{INFORMATION_START_TOKEN} {json.dumps(results, ensure_ascii=False)} {INFORMATION_END_TOKEN}"
        generated = append_text_to_ids(tokenizer, generated, injected, device)

    final_text = tokenizer.decode(tokenizer_decode_special_tokens(generated[0], tokenizer), skip_special_tokens=False)

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

    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    vision_tower = model.get_model().get_vision_tower().to(device=device)

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

        # <search> tool-aware generation
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
            image_resize_hw=resize_list[0],
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
