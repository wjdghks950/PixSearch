"""
Classify VQA questions into 10 custom categories with LLM (e.g., gpt-4o-mini)

If an existing JSON cache is provided, it is reused to skip costly
API calls. Pie-charts summarizing the distribution are written as PNGs.
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import openai
from openai import APIConnectionError, RateLimitError, InternalServerError
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
QUESTION_DEFS: Dict[str, str] = {
    "OCR Read": "Answer is literal text copied from the image (pure OCR).",
    "OCR + Visual Reasoning": "Read text AND reason over visual context (compare, locate, infer).",
    "Multi-hop External Knowledge Reasoning": "Chain external facts with visual cues (e.g., item ➜ purpose ➜ date).",
    "Fine-grained Entity Identification": "Identify the exact entity/species/object (Wikipedia-level specificity).",
    "Visual Reasoning-Attribute": "Infer a visible attribute such as color, shape, material.",
    "Visual Reasoning-Counting": "Count objects or read a number visible in the scene.",
    "Visual Reasoning-Binary": "Yes/No verification of a visual fact (presence, state).",
    "Social Commonsense Reasoning": "Commonsense about human intentions, social situations.",
    "Physical Commonsense Reasoning": "Intuitive physics or spatial reasoning (how objects interact).",
    "Factoid / KB Questions": "Single factual lookup about an entity in the image (who / where / when).",
}
QUESTION_TYPES = list(QUESTION_DEFS.keys())

SYSTEM_PROMPT = (
    "You are a strict classifier for visual-question-answering (VQA) questions.\n"
    "Return **only** a JSON object with the single key \"type\" whose value is "
    "exactly one of the 10 labels below.\n\n"
    + "\n".join(f"{lbl}: {desc}" for lbl, desc in QUESTION_DEFS.items())
    + "\n\nIf none fits perfectly, choose the closest match."
)

# ────────────────────────────────────────────────────────────────────────────
# GPT helper
# ────────────────────────────────────────────────────────────────────────────
def classify_question(
    text: str,
    model: str,
    temp: float,
    max_retries: int = 5,
    sleep_base: float = 1.0,
) -> str:
    """
    Call OpenAI chat completion and return one of the QUESTION_TYPES.
    Uses exponential-backoff retry on connection / rate-limit errors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = openai.chat.completions.create(
                model           = model,
                temperature     = temp,
                response_format = {"type": "json_object"},
                messages        = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text.strip()},
                ],
            )
            label = json.loads(resp.choices[0].message.content)["type"]
            if label not in QUESTION_TYPES:
                raise ValueError(f"Unexpected label {label!r}")
            return label

        except (APIConnectionError, RateLimitError, InternalServerError) as e:
            if attempt == max_retries:
                raise
            delay = sleep_base * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
            print(f"[retry {attempt}/{max_retries}] {type(e).__name__}: {e}. Sleeping {delay:.1f}s")
            time.sleep(delay)

# ────────────────────────────────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────────────────────────────────
def iter_dataset_files(data_root: Path) -> List[Path]:
    """
    Hard‑coded list of six dataset sub‑folders expected under data_root.
    Adjust or extend as needed.
    """
    pattern = "{name}/{name}_search_annotated_v1.jsonl"
    names   = ["aokvqa", "okvqa", "textvqa", "webqa", "infoseek", "oven"]
    return [data_root / pattern.format(name=n) for n in names]

def pretty_print(counter: Counter, title: str) -> None:
    total = sum(counter.values())
    print(f"\n=== {title} ===  (n={total})")
    for lbl in QUESTION_TYPES:
        n = counter.get(lbl, 0)
        pct = n / total * 100 if total else 0
        print(f"{lbl:<40}: {n:>6} ({pct:5.1f}%)")

def slugify(s: str) -> str:
    s = re.sub(r"[–—]", "-", s)          # dashes
    s = re.sub(r"[^a-zA-Z0-9_\- ]+", "", s)
    return s.strip().replace(" ", "_").lower()

def save_pie(counter: Counter, title: str, fig_dir: Path) -> None:
    total = sum(counter.values())
    if total == 0:
        return
    labels = [lbl for lbl in QUESTION_TYPES if counter.get(lbl, 0)]
    sizes  = [counter[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%\n({int(p*total/100)})",
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title(title, fontsize=14)
    ax.axis("equal")

    fname = slugify(title) + ".png"
    fig.savefig(fig_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved → {fig_dir / fname}")

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Classify VQA questions into 10 categories.")
    parser.add_argument("--data_root", default="/home/jeonghkim/projects/region-mm-retrieve/data/region_rag_vqa", type=Path, help="Root folder containing dataset sub‑dirs")
    parser.add_argument("--out_json", default="/home/jeonghkim/projects/region-mm-retrieve/data/region_rag_vqa/question_types.json", type=Path, help="Path to write/read cached classification JSON")
    parser.add_argument("--fig_dir", default="/home/jeonghkim/projects/region-mm-retrieve/plots", type=Path, help="Directory to save pie charts")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # ── Prepare I/O dirs
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    # ── OpenAI key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        parser.error("OPENAI_API_KEY env var not set")

    # ── Load / classify
    per_dataset_qtypes = defaultdict(list)
    if args.out_json.exists():
        print(f"Loading cached classifications from {args.out_json}")
        with open(args.out_json) as fh:
            per_dataset_qtypes = defaultdict(list, json.load(fh))

    for jsonl_path in iter_dataset_files(args.data_root):
        ds_name = jsonl_path.parent.name
        if not jsonl_path.exists():
            print(f"skip missing {jsonl_path}")
            continue

        print(f"\n> Loading {jsonl_path}")
        with open(jsonl_path, encoding="utf-8") as fh:
            ds_jsonl_file = [json.loads(line) for line in tqdm(fh, desc=f"{ds_name} questions")]

        if len(per_dataset_qtypes[ds_name]) >= len(ds_jsonl_file):
            print(f"skip {ds_name} (already processed)")
            continue

        # Iterate from per_dataset_qtypes onwards
        print(f"\n> Processing {ds_name}")
        for idx, obj in enumerate(tqdm(ds_jsonl_file, desc=f"{ds_name} questions")):
            if idx < len(per_dataset_qtypes[ds_name]):
                continue
            question = obj.get("question", "").strip()
            if not question:
                continue
            label = classify_question(question, args.model, args.temperature)
            per_dataset_qtypes[ds_name].append({
                "question": question,
                "label": label,
            })
            time.sleep(1.0 / 3)  # gentle global rate‑limit buffer

    with open(args.out_json, "w") as fh:
        json.dump(per_dataset_qtypes, fh, indent=2)
        print(f"Cache written → {args.out_json}")

    # ── Aggregate counts
    per_ds_counter: Dict[str, Counter] = {}
    overall = Counter()
    for ds, mapping in per_dataset_qtypes.items():
        ds_counter = Counter()
        for qlbl_dict in mapping:
            ds_counter[qlbl_dict["label"].strip()] += 1
        per_ds_counter[ds] = ds_counter
        overall.update(ds_counter)

    # ── Print reports & plots
    for ds, cnt in per_ds_counter.items():
        pretty_print(cnt, ds)
        save_pie(cnt, f"{ds} - Question-type Distribution", args.fig_dir)

    pretty_print(overall, "CUMULATIVE")
    save_pie(overall, "All Datasets - Question-type Distribution", args.fig_dir)

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # tidy matplotlib defaults once, here
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    main()
