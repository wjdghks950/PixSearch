from enum import Enum

import os
import yaml
import json
import time
import re
import cv2
import pycocotools.mask as mask_utils
import numpy as np
import torch
import torch.distributed as dist
from typing import Optional, Tuple, List, Union, Dict, Any
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
LLAMA_DEFAULT_IMAGE_TOKEN = "<|image|>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SEARCH_START_TOKEN = "<search>"
SEARCH_END_TOKEN = "</search>"
INFORMATION_START_TOKEN = "<information>"
INFORMATION_END_TOKEN = "</information>"
DEFAULT_IMG_PAYLOAD_TOKEN = "<img>"
DEFAULT_REGION_TOKEN = "<region>"

IM_START_ID = 32000
IM_END_ID   = 32001

QTYPE2ID = {"TEXT": 0, "IMAGE": 1, "REGION": 2}

TOK_START_CHAR = '▁'  # token start character
BIO_NO_LBL = 0
BIO_START_LBL = 1
BIO_INTERM_LBL = 2
ASSISTANT_START_TOK = "ASSISTANT:"
IMAGE_TOKENS_OFFSET = 255  # 255 is the number of image tokens in the tokenizer


SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please elaborate your answer and explain why.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please give some explanation.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Could you isolate the {class_name} in this picture?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Would you please extract the {class_name} from the image below?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the {class_name} from the provided image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Display a segmentation mask for the {class_name} shown in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please highlight the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Could you identify and segment out the {class_name} in this picture?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please point out the {class_name} in this picture.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent}. Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent}. Please output segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent}. Please elaborate your answer and explain why.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent}. Please give some explanation.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

PLUM_REASONSEG_EXPLANATORY_QUESTION_LIST = [
    "Please explain why.",
    "Please elaborate your answer and explain why.",
    "Please give some explanation.",
    "Feel free to explain your answer.",
    "Please provide some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

PLUM_REASONSEG_ANSWER_LIST = [
    "{class_name}"
]

PLUM_ANSWER_LIST = [
    "Here is the {class_name} you asked about.",
    "Sure — this shows the {class_name}.",
    "The {class_name} is presented here.",
    "Below you can see the {class_name}.",
    "{class_name}.",
    "This view focuses on the {class_name}.",
    "I've marked the {class_name} for you.",
    "Here's where the {class_name} appears in the image.",
    "Take a look at the {class_name} here.",
    "The region corresponding to the {class_name} is shown.",
    "Result: {class_name}.",
    "You can see the {class_name} in this frame.",
    "The {class_name} portion is displayed below.",
    "Here's the area for {class_name}.",
    "This is the selected {class_name}.",
    "Displayed here is the {class_name}."
]

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    '''Computes and stores the average and current value'''

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    return dict_to_device(input_dict, torch.device("cuda"))

def dict_to_device(input_dict, device):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.to(device, non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.to(device, non_blocking=True) for ele in v]
    return input_dict


def load_yaml(path: str, encoding='utf-8'):
    if os.path.exists(path):
        print(f"loading the yaml file from : {path}")
        with open(path, 'r') as f:
            graph_yaml = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"[ {path} ] does not exist in the file system.")
    return graph_yaml


def load_json(path: str):
    if os.path.exists(path):
        with open(path, 'r') as f:
            json_dict = json.load(f)
    else:
        raise FileNotFoundError(f"[ {path} ] does not exist in the file system.")
    return json_dict


def load_rle(rle_dict: dict):
    # open the JSON dictionary with "size" and "counts"
    dense_mask = mask_utils.decode(rle_dict)
    return dense_mask


def load_polygon(segmentation: list, height: int, width: int):
    '''
    Converts a polygon segmentation into a binary mask using OpenCV.

    Args:
        - segmentation (list of float): List of polygon coordinates [x1, y1, x2, y2, ..., xn, yn]
        - height (int): height of the mask
        - width (int): width of the mask
    '''
    polygon = np.array(segmentation).reshape(-1, 2).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)
    return mask

def open_image(path: str) -> PILImage:
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')


def is_alphanumeric_or_underscore(char: str) -> bool:
    '''
    Returns True if 'char' is alphanumeric or underscore,
    used to check if token boundaries are within a larger word.
    '''
    return char.isalnum() or char == '_'


def find_sublist_index(full_list, sublist):
    '''
    Find the starting index of sublist in full_list
    - full_list: list of token ids for the full text
    - sublist: list of token ids for the substrings to match
    '''
    if not isinstance(full_list, list):
        full_list = full_list.tolist()
    if not sublist:
        return 0
    start_indices = []
    for i in range(len(full_list) - len(sublist) + 1):
        if full_list[i:i+len(sublist)] == sublist:
            start_indices.append(i)
    return start_indices


def match_substrings(
    full_text_ids: list[int],
    substring_ids_list: list[list[int]],
    tokenizer,  # transformers AutoTokenizer
    anchor_token: str = "ASSISTANT:",
):
    '''
    - full_text_ids: list of token ids for the full text
    - substring_ids_list: list of lists of token ids for the substrings to match
    - tokenizer: transformers AutoTokenizer
    - anchor_token: the token to anchor the match to

    Returns:
    - matched_substrings: list of substrings that fall within the full text (after the 'anchor_token')
    - per_token_labels: list of BIO labels for each token in the full text
    - mask_positions: list of indices of the tokens that match the substrings
    '''
    matched_substrings = []
    per_token_labels = [BIO_NO_LBL] * len(full_text_ids)
    mask_positions = set()
    anchor_ids = tokenizer(anchor_token)['input_ids'][1:]
    anchor_idx = find_sublist_index(full_text_ids, anchor_ids)

    assert len(anchor_idx) == 1, f"Anchor token '{anchor_token}' found multiple times in the full text | anchor_idx: {anchor_idx} | full_text_ids: {full_text_ids} | anchor_text: {tokenizer.convert_ids_to_tokens(anchor_ids)} | full_text: {tokenizer.convert_ids_to_tokens(full_text_ids)}"
    anchor_idx = anchor_idx[0]

    # NOTE: multiple matches are possible for the same substring
    # substring match is only valid after the anchor_token
    # - e.g., "tail" in "red tail, curvy tail"
    for substring_ids in substring_ids_list:
        start_indices = find_sublist_index(full_text_ids[anchor_idx:], substring_ids)
        start_indices = [idx + anchor_idx for idx in start_indices]
        matched_substrings.extend([tokenizer.decode(substring_ids) for _ in start_indices])
        for substring_idx in start_indices:
            per_token_labels[substring_idx] = BIO_START_LBL
            for i in range(1, len(substring_ids)):
                per_token_labels[substring_idx + i] = BIO_INTERM_LBL
            mask_positions.add(substring_idx)

    return matched_substrings, per_token_labels, mask_positions

def _with_patch_offset(token_idx: int):
    return token_idx + IMAGE_TOKENS_OFFSET

def _imread_rgb_sans_icc(path: str) -> np.ndarray:
    '''
    Safely reads both png and jpg images, ignoring the ICC profile that causes IO cluttering.
    '''
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".png":
        # PIL will automatically load and convert to sRGB
        with Image.open(path) as img:
            img = img.convert("RGB")
            return np.array(img)  # H x W x 3 uint8 (RGB)
    else:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _info_char_spans(text: str, info_tok_include: bool = True):
    '''
    Returns list of (start_char, end_char) for the content inside
    <information>...</information> tags (excludes the tags themselves).
    '''
    spans = []
    for m in re.finditer(r"<information>(.*?)</information>", text, flags=re.DOTALL):
        inner = m.group(1)
        s, e = m.span(1)  # raw inner span including any spaces
        # trim leading/trailing whitespace from the inner span
        ltrim = len(inner) - len(inner.lstrip())
        rtrim = len(inner) - len(inner.rstrip())
        s_trim = s + ltrim
        e_trim = e - rtrim

        # Only keep if there's non-empty (post-trim) payload  - TODO: Test this chunk to see if <information> </information> are included
        if s_trim < e_trim:
            if info_tok_include:
                # Inclusive of tags
                s_all, e_all = m.span(0)
                spans.append((s_all, e_all))
            else:
                # Payload only (exclusive of tags)
                spans.append((s_trim, e_trim))

    return spans


def _search_char_spans(text: str):
    """
    Return [(start_char, end_char)] for each literal '<search>' opening tag.
    Indices cover exactly the 8 characters '<search>'.
    """
    return [(m.start(), m.end()) for m in re.finditer(rf"{SEARCH_START_TOKEN}", text)]

def _find_image_block_row(row_ids: torch.Tensor):
    """Returns (start_idx, end_idx_exclusive) of <im_start> ... <im_end>,
       or (None, None) if no image tokens present in this row."""
    where_start = (row_ids == IM_START_ID).nonzero(as_tuple=False)
    where_end   = (row_ids == IM_END_ID).nonzero(as_tuple=False)
    if where_start.numel() == 0 or where_end.numel() == 0:
        return None, None
    s = where_start[0].item()
    e = where_end[0].item()
    return s, e + 1

def _truncate_row_keep_block(row: torch.Tensor, L: int, block):
    """Truncate 1D row to length L, preserving [block_start:block_end) if present."""
    try: # XXXc
        if row.size(0) <= L:
            return row, 0  # no shift
    except Exception as e:
        print(e)
        breakpoint()

    if block[0] is None:
        # no image block → head-truncate
        return row[:L].contiguous(), 0

    b0, b1 = block
    block_len = b1 - b0
    if block_len > L:
        # cannot preserve; take a head window (still prevents length overflow)
        return row[:L].contiguous(), 0

    # try to window so the block stays fully inside
    left = max(0, b0 - (L - block_len) // 2)
    right = left + L
    if right < b1:
        right = b1
        left = right - L
    left = max(0, min(left, row.size(0) - L))
    right = left + L
    return row[left:right].contiguous(), left

def clamp_conversation_tensors_to_maxlen(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_masks: torch.Tensor,
    per_token_labels: torch.Tensor,
    info_mask_tensors: torch.Tensor,
    mask_positions_in_input_ids: list,
    max_len: int,
):
    """
    All tensors are shaped [N_conversations, seq_len].
    mask_positions_in_input_ids is a list (len=N_conversations) of lists of indices.
    Truncates rows to max_len while preserving the image block per row when possible,
    and remaps mask_positions accordingly.
    """
    N, L = input_ids.shape
    new_input_ids = []
    new_labels = []
    new_attn = []
    new_ptl = []
    new_info_masks = []
    remapped_positions = []

    for i in range(N):
        row_ids = input_ids[i]
        block = _find_image_block_row(row_ids)

        ids_i, shift = _truncate_row_keep_block(row_ids, max_len, block)
        lab_i, _     = (_truncate_row_keep_block(labels[i], max_len, block) if labels is not None else (torch.zeros_like(ids_i), None))
        att_i, _     = _truncate_row_keep_block(attention_masks[i], max_len, block)
        ptl_i, _     = _truncate_row_keep_block(per_token_labels[i], max_len, block)
        info_masks, _     = _truncate_row_keep_block(info_mask_tensors[i], max_len, block)

        new_input_ids.append(ids_i)
        new_labels.append(lab_i)
        new_attn.append(att_i)
        new_ptl.append(ptl_i)
        new_info_masks.append(info_masks)

        # remap positions (drop those outside the new window)
        pos_list = mask_positions_in_input_ids[i] if i < len(mask_positions_in_input_ids) else []
        remapped = [p - shift for p in pos_list if 0 <= (p - shift) < max_len]
        remapped_positions.append(remapped)

    input_ids = torch.stack(new_input_ids, dim=0)
    labels = torch.stack(new_labels, dim=0)
    attention_masks = torch.stack(new_attn, dim=0)
    per_token_labels = torch.stack(new_ptl, dim=0)
    info_masks_tensors = torch.stack(new_info_masks, dim=0)

    return input_ids, labels, attention_masks, per_token_labels, info_masks_tensors, remapped_positions


def build_first_sentence_fn():
    """
    Try spaCy if available; fall back to a light rule-based splitter.
    No network download required.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # fallback: rule-based sentencizer (no model file)
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")

        def _first_sentence_spacy(text: str) -> str:
            if not text:
                return ""
            doc = nlp(text.strip())
            for s in doc.sents:
                return " ".join(s.text.split())  # collapse whitespace
            return text.strip().splitlines()[0]  # fallback

        return _first_sentence_spacy

    except Exception:
        # regex fallback with common abbrev guards
        import re
        _ABBR = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|St|Mt|Inc|Ltd|Co|Dept|No|Fig|Eq|Ref|e\.g|i\.e|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|U\.S|U\.K|al)"
        _EOS = re.compile(
            rf"""
            ^\s*                    # leading ws
            (                       # capture the first sentence
              .*?                   # non-greedy up to...
              (?:                   # an end-of-sentence token:
                 (?<!\b{_ABBR})    #   not an abbrev like "Dr."
                 (?<!\b[A-Z]\.)    #   not an initial like "A."
                 (?<!\d)\.         #   period not after a digit
               | [!?]
              )
            )                       # end capture
            (?:\s+|$)               # followed by space or end
            """,
            re.VERBOSE | re.DOTALL,
        )

        def _first_sentence_regex(text: str) -> str:
            if not text:
                return ""
            text = text.strip()
            # stop early at a hard newline if it comes before any EOS
            nl = text.find("\n")
            if nl != -1 and not any(ch in text[:nl] for ch in ".?!"):
                return text[:nl].strip()
            m = _EOS.search(text)
            if m:
                return " ".join(m.group(1).strip().split())
            # final fallback
            return (text.split("\n", 1)[0] or text)[:280].strip()

        return _first_sentence_regex

def format_payload(payload_contents: List[Union[str, tuple]]):
    '''
    Format a list of strings into "|" delimited string.
    E.g., ["a", "b", "c"] -> "a | b | c"
    '''
    if not payload_contents:
        return ""
    elif isinstance(payload_contents[0], str):
        output_text = " | ".join(payload_contents).strip()
    elif isinstance(payload_contents[0], tuple):
        output = [f"{p[0].strip()}: {p[1].strip()}" for p in payload_contents]
        output_text = " | ".join(output).strip()

    return output_text
