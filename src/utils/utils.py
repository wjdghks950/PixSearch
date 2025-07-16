import re
import os
import io
import ast
import json
import shutil
import base64
import openai
import requests

from PIL import Image
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Optional, Union, Tuple, Any
from .image_loader import ImageLoader

# Special tokens
DEFAULT_IMAGE_TOKEN = "<image>"

# Default values for FFmpeg used in resize
MAX_SIDE = 336

# Hyperparameters for llm APIs
TIMEOUT_SECONDS = 60
TEMPERATURE = 0.2
RETRY_BACKOFF_SECONDS = [5, 10, 20]

DATASETS = {
    "aokvqa":  {"hf_name": "HuggingFaceM4/A-OKVQA", "split": "train"},
    "okvqa":   {"hf_name": "HuggingFaceM4/OK-VQA",   "split": "train"},
    "webqa":   {"hf_name": "TreezzZ/WebQA",       "split": "train"},
    "textvqa": {"hf_name": "lmms-lab/textvqa", "split": "train"},
    "vizwiz":  {"hf_name": "HuggingFaceM4/VizWiz", "split": "train"},
    "infoseek": {"hf_name": "ychenNLP/oven", "split": "train"},
    "oven": {"hf_name": "ychenNLP/oven", "split": "train"},
    # "encyclopedicvqa": {"hf_name": "", "split": "train"},
    # "visual_genome": {"hf_name": "visual_genome", "split": "train"},
    # "kvqa":    {"hf_name": "AdaptAI/KVQA",          "split": "train"},
    # "pathvqa": {"hf_name": "shapenshap/pathvqa",    "split": "train"},
    # "scienceqa": {"hf_name": "lukaemon/ScienceQA",  "split": "train"},
    # "cragmm": {"hf_name": "crag-mm-2025/crag-mm-single-turn-public", "split": "validation"}
}


def image_to_base64(
    image: Union[str, os.PathLike, Image.Image],
    fmt: str = "PNG",
    with_data_uri: bool = False,
) -> str:
    """
    Convert a PIL.Image or an image file path/URL to a base-64 string.

    Args:
        image: Path/str/URL to an image file *or* a PIL.Image.Image instance.
        fmt:   Format to encode the buffer in (PNG, JPEG, WEBP, …).
        with_data_uri: If True, prepend 'data:image/<fmt>;base64,'.

    Returns:
        Base-64 encoded image as **str**.
    """
    # ── 1. Load image if we were given a path/URL ───────────────────────────
    if isinstance(image, str):
        if image.startswith("http"):
            image = ImageLoader(image).get_image()
        else:
            image = Image.open(str(image)).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError("`image` must be a path/URL or PIL.Image.Image")

    # ── 2. Encode to bytes buffer ───────────────────────────────────────────
    buf = io.BytesIO()
    image.save(buf, format=fmt.upper())
    b64_bytes = base64.b64encode(buf.getvalue())
    b64_str = b64_bytes.decode("utf-8")

    # ── 3. Optionally prepend data-URI prefix ───────────────────────────────
    if with_data_uri:
        b64_str = f"data:image/{fmt.lower()};base64,{b64_str}"

    return b64_str


def image_to_pil(path_or_url: Optional[Union[str, Path]], timeout: int = 3) -> Optional[Image.Image]:
    """
    Load an image from a web URL *or* a local file system path.

    Parameters
    ----------
    path_or_url : str
        - 'https://…' or 'http://…' → download over HTTP(S)
        - '/some/folder/img.jpg', r'C:\img\photo.png', './img.png' → load locally
        - 'file:///absolute/path/to/img.jpg' is also supported.
    timeout : int, optional
        Seconds to wait for the server when fetching via HTTP(S).

    Returns
    -------
    PIL.Image.Image or None
        Decoded image on success, otherwise None.
    """
    try:
        # Convert Path to string if needed
        path_or_url_str = str(path_or_url) if isinstance(path_or_url, Path) else path_or_url
        parsed = urlparse(path_or_url_str)

        # --- Case 1 : HTTP/HTTPS URL ----------------------------------------
        if parsed.scheme in {"http", "https"}:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ImageFetcher/1.1)"
            }
            resp = requests.get(path_or_url_str, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

        # --- Case 2 : file:// URI ------------------------------------------
        if parsed.scheme == "file":
            local_path = Path(parsed.path)  # drop the 'file://' prefix
            return Image.open(local_path).convert("RGB")

        # --- Case 3 : plain local path -------------------------------------
        local_path = Path(path_or_url)
        if local_path.exists():
            return Image.open(local_path).convert("RGB")

        print(f"[load_image] Path does not exist: {path_or_url}")
        return -1

    except Exception as exc:
        # Swap to logging if preferred
        breakpoint()
        print(f"[load_image] Failed to load '{path_or_url_str if 'path_or_url_str' in locals() else path_or_url}': {exc}")
        return -1


def load_image_by_id(root: Path, image_id: str):
    """
    Return a PIL.Image for `image_id`, trying .JPEG, .jpeg and .jpg
    (in that order). Raise FileNotFoundError if none of them exist.
    """
    for ext in (".JPEG", ".jpeg", ".jpg"):
        candidate = root / f"{image_id}{ext}"
        if candidate.exists():
            return image_to_pil(candidate)

    raise FileNotFoundError(
        f"Could not find an image for id '{image_id}' "
        f"in {root} with extensions .JPEG/.jpeg/.jpg"
    )


def extract_list_literal(text: str) -> Optional[List[Any]]:
    """
    Return the first substring that *looks* like a Python list literal,
    converted to an actual list.  Ignores invalid matches.
    """
    LIST_REGEX = re.compile(r'\[[^\[\]]*?\]')      # non-greedy, no nested [ ]
    for m in LIST_REGEX.finditer(text):
        candidate = m.group(0)
        try:                           # safest way to “eval” literals
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue                   # wasn’t a valid literal -> keep searching
    return None


def choose_gold_object(
    args,
    client,
    question: str,
    image: Union[str, os.PathLike, Image.Image],
    object_list: List[str],
):
    """
    Choose the gold object for a given question and image.

    Args:
        client: A LLaMA client instance or a OpenAI client.
        question: The question to ask.
        image: Path/str/URL to an image file *or* a PIL.Image.Image instance.
        object_list: List of objects to choose from.

    Returns:
        The chosen object as **str**.
    """
    prompt = \
    """
        Choose the most salient object of interest that best matches the image and question. \
        You should only choose from the provided list of objects. Do NOT generate anything other than from the provided list of objects. \
        Answer with the list of object names only. Do not include any other text. Provide the answer as object noun phrases within a Python list after the 'Answer:'. \
        For example, if the object list is ["chair_1", "chair_2", "blue car_1", "red car_1", "red car_2"], and the question is related to the blue car you see in the image, the answer should be: \
        ["blue car_1"]. If the question is about comparing all the cars in the image, the answer should be: ["blue car_1", "red car_1", "red car_2"]. \
        Make sure each object string in the square brackets is enclosed in double quotes.
    """.strip()

    encoded_image = image_to_base64(image, with_data_uri=True)

    if "llama" in args.llm_model.lower():
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}\n\nQuestion: {question}\n\n"
                    }
                ]
            },
        ]
        messages[-1]["content"].append({"type": "image_url", "image_url": {"url": encoded_image}})
        messages[-1]["content"].append({"type": "text", "text": f"List of objects to choose from: [{', '.join(object_list)}]\n\nAnswer:"})
        response = client.chat.completions.create(
            model=args.llm_model,
            messages=messages,
        )
        output_text = response.completion_message.content.text
    elif "gpt" in args.llm_model.lower():
        SYSTEM_MSG = "You are a helpful assistant that helps the user to choose the most salient object of interest that best matches the image and question."
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}\n\nQuestion: {question}\n\n"},
                    {"type": "image_url", "image_url": {"url": encoded_image}},
                    {"type": "text", "text": f"List of objects to choose from: [{', '.join(object_list)}]\n\nAnswer:"}
                ],
            },
        ]
        resp = client.chat.completions.create(
            model=args.llm_model,
            messages=messages,
            temperature=TEMPERATURE,
        )
        output_text = resp.choices[0].message.content

    gold_obj_lbls = extract_list_literal(output_text)

    if gold_obj_lbls is None:
        _CODE_FENCE_RGX = re.compile(r"```[ \t]*([a-zA-Z0-9_-]+)?[ \t]*\n|\n```", re.DOTALL)  # handle issues like e.g., '```python\n[screen_1]\n```'
        output_text = _CODE_FENCE_RGX.sub("", output_text).strip()
        gold_obj_lbls = [output_text[1:-1].strip()]

    return gold_obj_lbls, output_text


def compress_image(
    src: Union[str, os.PathLike, Image.Image],
    *,
    max_bytes: int = 512_000,         # 500 KB by default
    max_side: int | None = 1024,      # shrink longest edge if bigger
    min_quality: int = 20,            # don’t go below this JPEG quality
    quality_step: int = 5,
) -> Tuple[Image.Image, int]:
    """
    Returns
    -------
    img_out : PIL.Image.Image
        Compressed image (RGB, JPEG).
    size_bytes : int
        Actual byte size of the compressed buffer.
    """
    img = _load_image(src).convert("RGB")  # drop alpha → smaller

    # ── 1. Optional geometric down-scale ──────────────────────────────
    if max_side and max(img.size) > max_side:
        ratio = max_side / max(img.size)
        new_sz = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_sz, Image.LANCZOS)

    # ── 2. Iteratively reduce JPEG quality until <= max_bytes ─────────
    for q in range(95, min_quality - 1, -quality_step):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        size = buf.tell()
        if size <= max_bytes or q == min_quality:
            buf.seek(0)
            return Image.open(buf), size

    # Fallback — shouldn’t normally reach here
    raise RuntimeError("Could not compress image within byte budget.")


def resize_keep_aspect(pil_img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Return an in-memory resized copy whose longer edge = max_side px."""
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img            # small enough already
    ratio = max_side / float(max(w, h))
    new_sz = (int(w * ratio), int(h * ratio))
    return pil_img.resize(new_sz, Image.BICUBIC)


def pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def sanitize_pil(img: Image.Image) -> Image.Image:
    """Sanitize a PIL image by removing EXIF tags. Sometimes, WebP images arrive."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")     # PNG has no EXIF block
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def extract_dict_like_str(s: str) -> dict[str, Any]:
    DICT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)      # greedy -> grabs the whole dict
    m = DICT_PATTERN.search(s)
    payload = m.group(0) if m else None
    try:
        payload = ast.literal_eval(m.group(0))
        return payload
    except Exception as e:
        print(f"failed to parse dict with ast.literal_eval: {e}. Trying json.loads")
        breakpoint()
    try:
        payload = json.loads(m.group(0))
        return payload
    except Exception as e:
        print(f"failed to parse dict with json.loads: {e}")
        breakpoint()

    return payload


def load_webqa_tsv_images(
        tsv_path: str,
        lineidx_path: str,
        image_ids: list[str],
        cache_dir: str = None
    ) -> dict[str, Image.Image]:
    """
    Load WebQA images from imgs.tsv (base-64) and optionally cache them on disk.

    Parameters
    ----------
    tsv_path     : Path to imgs.tsv
    lineidx_path : Path to lineidx file (1 byte-offset per TSV line, plain text)
    image_ids    : List of image-id strings to fetch
    cache_dir    : If given, images are saved to this directory as <img_id>.<ext>

    Returns
    -------
    dict[str, PIL.Image.Image] mapping image_id → PIL image
    str : Paths to cached image files
    """

    with open(lineidx_path, "r") as fp_lineidx:  # read the per line image index
        lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]

    # 2. prepare cache directory
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    id2img = {}
    img_paths = []
    with open(tsv_path, "r") as fp:
        for img_id in image_ids:
            # use cached file if it already exists
            if cache_path:
                cached = next(cache_path.glob(f"{img_id}.*"), None)
                if cached:
                    img_paths.append(cached)
                    id2img[img_id] = Image.open(cached)
                    continue
            fp.seek(lineidx[int(img_id)%10000000])
            image_id, img_base64 = fp.readline().strip().split('\t')
            assert image_id == img_id, "TSV and requested ID mismatch"
            img = Image.open(io.BytesIO(base64.b64decode(img_base64)))
            id2img[img_id] = img

            # save to cache
            if cache_path:
                ext = img.format.lower() if img.format else "jpg"
                img.save(cache_path / f"{img_id}.{ext}")
                img_paths.append(cache_path / f"{img_id}.{ext}")

    return id2img, img_paths


def cache_image(
    image_paths: List[Union[str, Image.Image]],
    out_dir: Union[str, Path],
    qid: str,
) -> List[str]:
    """
    Persist every image in *image_paths* under *out_dir* using deterministic
    filenames derived from *qid*. Returns a list of **local file paths**.

    Rules
    -----
    • If the element is a PIL.Image it is saved directly.
    • If the element is a local path it is copied (metadata preserved).
    • Otherwise it is treated as a URL / remote resource: we load with
      image_to_pil() and save.
    • Files are not overwritten if already cached.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cached_paths: List[str] = []
    for idx, img in enumerate(image_paths):
        # One image →  "<qid>.png"
        # Multiple →  "<qid>_<idx>.png"
        fname = f"{qid}{'_' + str(idx) if len(image_paths) > 1 else ''}.png"
        dst = out_dir / fname

        if dst.exists():
            cached_paths.append(str(dst))
            continue

        try:
            if isinstance(img, Image.Image):
                img.save(dst)
            else:
                src = Path(img)
                if src.is_file():
                    shutil.copy2(src, dst)
                else:
                    image_to_pil(img).save(dst)
        except Exception:
            # Fallback: always try PIL conversion
            image_to_pil(img).save(dst)

        cached_paths.append(str(dst))

    return cached_paths
