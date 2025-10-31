'''
$ python analyze_jsonl.py /path/to/your/file.jsonl
'''

#!/usr/bin/env python3
import argparse
import ast
import json
import re
import sys
from typing import List, Union

SEARCH_OPEN_TAG = "<search>"
INFO_OPEN = "<information>"
INFO_CLOSE = "</information>"

# Regexes (case-insensitive, dotall to span lines)
SEARCH_OPEN_PATTERN = re.compile(r"<\s*search\s*>", flags=re.IGNORECASE)
SEARCH_BLOCK_PATTERN = re.compile(r"<\s*search\s*>(.*?)<\s*/\s*search\s*>",
                                  flags=re.IGNORECASE | re.DOTALL)
INFO_BLOCK_PATTERN = re.compile(r"<\s*information\s*>(.*?)<\s*/\s*information\s*>",
                                flags=re.IGNORECASE | re.DOTALL)
REGION_TAG_PATTERN = re.compile(r"<\s*region\s*>", flags=re.IGNORECASE)
IMG_TAG_PATTERN = re.compile(r"<\s*img\s*>", flags=re.IGNORECASE)

def parse_ground_truth(gt_val: Union[str, List[str]]) -> List[str]:
    """Normalize 'ground_truth' into a list[str] (lowercased)."""
    if isinstance(gt_val, list):
        return [str(x).strip().lower() for x in gt_val if str(x).strip()]
    if isinstance(gt_val, str):
        s = gt_val.strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [s.lower()] if s else []
    return [str(gt_val).strip().lower()] if gt_val is not None else []

def count_search_open_tokens(*texts: str) -> int:
    """Count literal '<search>' opens across texts (robust even if missing </search>)."""
    total = 0
    for t in texts:
        if not t:
            continue
        total += len(SEARCH_OPEN_PATTERN.findall(t))
    return total

def extract_search_blocks(*texts: str) -> List[str]:
    """Extract inner contents of all <search>...</search> blocks across texts."""
    blocks: List[str] = []
    for t in texts:
        if not t:
            continue
        blocks.extend([m.group(1) for m in SEARCH_BLOCK_PATTERN.finditer(t)])
    return blocks

def classify_search_block(inner: str) -> str:
    """
    Classify a <search>...</search> block as 'region', 'img', or 'text'.
    Priority: region > img > text.
    """
    if not inner:
        return "text"
    if REGION_TAG_PATTERN.search(inner):
        return "region"
    if IMG_TAG_PATTERN.search(inner):
        return "img"
    return "text"

def any_gt_in_text(gt_list: List[str], text: str, question: str = "") -> bool:
    """True if ANY ground-truth string appears (substring, case-insensitive) in text."""
    if not text or not gt_list:
        return False
    hay = text.lower()
    for gt in gt_list:
        g = str(gt).strip().lower()
        if g and g in hay:
            return True

    return False

def extract_info_blocks(text: str) -> List[str]:
    """Return the contents of all <information>...</information> blocks (as strings)."""
    if not text:
        return []
    return [m.group(1) for m in INFO_BLOCK_PATTERN.finditer(text)]

def main():
    ap = argparse.ArgumentParser(description="Analyze a JSONL VQA-style file.")
    ap.add_argument("jsonl_path", help="Path to the JSONL file")
    args = ap.parse_args()

    needs_true = needs_false = needs_unknown = 0

    total_search_open_tokens = 0
    search_block_total = 0
    search_block_region = 0
    search_block_img = 0
    search_block_text = 0

    gt_present_lines = 0
    gt_inside_info_lines = 0

    total_lines = 0
    parse_errors = 0

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            # needs_retrieval
            nr = obj.get("needs_retrieval", None)
            if isinstance(nr, bool):
                needs_true += int(nr is True)
                needs_false += int(nr is False)
            else:
                needs_unknown += 1

            # answers
            answer_draft = (obj.get("answer_draft") or "").strip()
            answer = (obj.get("answer") or "").strip()

            # <search> token (open tag) count across answer_draft + answer
            total_search_open_tokens += count_search_open_tokens(answer_draft, answer)

            # <search>...</search> block extraction + typing
            blocks = extract_search_blocks(answer_draft, answer)
            search_block_total += len(blocks)
            for inner in blocks:
                t = classify_search_block(inner)
                if t == "region":
                    search_block_region += 1
                elif t == "img":
                    search_block_img += 1
                else:
                    search_block_text += 1

            # ground_truth normalization
            gt_list = parse_ground_truth(obj.get("ground_truth", []))

            # GT present anywhere in answer_draft or answer
            if any_gt_in_text(gt_list, f"{answer}", obj.get("question")):
                gt_present_lines += 1

            # GT present specifically inside <information>...</information> blocks of answer_draft
            info_blocks = extract_info_blocks(answer)
            if info_blocks and any_gt_in_text(gt_list, " ".join(info_blocks)):
                gt_inside_info_lines += 1

    # Pretty print
    print("\n=== JSONL Analysis Summary ===")
    print(f"File: {args.jsonl_path}")
    print(f"Total lines read:        {total_lines}")
    if parse_errors:
        print(f"Lines with parse errors: {parse_errors}")

    print("\nneeds_retrieval counts:")
    print(f"  True:    {needs_true}")
    print(f"  False:   {needs_false}")
    if needs_unknown:
        print(f"  Unknown: {needs_unknown}")

    print("\n<Search> token count (open tags):")
    print(f"  Total <search> opens in answer_draft/answer: {total_search_open_tokens}")

    print("\n<Search> block types (from matched <search>...</search> pairs):")
    print(f"  Total blocks: {search_block_total}")
    print(f"    region: {search_block_region}")
    print(f"    img:    {search_block_img}")
    print(f"    text:   {search_block_text}")

    print("\nGround-truth containment:")
    print(f"  Lines with ANY ground-truth in answer or answer_draft: {gt_present_lines}")
    print(f"  Lines with ground-truth INSIDE <information>...</information> (answer_draft): {gt_inside_info_lines}")

    # Percentages over readable lines
    readable = max(0, total_lines - parse_errors)
    if readable:
        def pct(x): return f"{(x/readable*100):.2f}%"
        print("\nPercentages (of readable lines):")
        print(f"  needs_retrieval=True:              {pct(needs_true)}")
        print(f"  needs_retrieval=False:             {pct(needs_false)}")
        if needs_unknown:
            print(f"  needs_retrieval=Unknown:           {pct(needs_unknown)}")
        print(f"  GT in answer/draft (anywhere):     {pct(gt_present_lines)}")
        print(f"  GT inside <information> (draft):   {pct(gt_inside_info_lines)}")

        # Optional block-type ratios (per all matched search blocks)
        total_blocks = max(1, search_block_total)
        print("\n<Search> block type ratios (of matched blocks):")
        print(f"  region: {search_block_region} ({search_block_region/total_blocks*100:.2f}%)")
        print(f"  img:    {search_block_img} ({search_block_img/total_blocks*100:.2f}%)")
        print(f"  text:   {search_block_text} ({search_block_text/total_blocks*100:.2f}%)")

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Allow piping to tools like `head` without traceback
        sys.exit(0)
