#!/usr/bin/env python3
import argparse
import base64
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from tqdm import tqdm
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    # New OpenAI SDK (>=1.0)
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False
    OpenAI = None  # type: ignore

# -----------------------------
# Utilities
# -----------------------------

def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Try permissive parsing for trailing commas or weird quotes
                try:
                    yield json.loads(line.rstrip(', '))
                except Exception:
                    print(f"[WARN] Skipping malformed JSONL line: {line[:160]}...", file=sys.stderr)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def b64_image(path: str) -> Optional[str]:
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"[WARN] Could not read image '{path}': {e}", file=sys.stderr)
        return None


_Q_RE = re.compile(r'###\s*Human:\s*(.*?)(?:###|</s>|$)', re.DOTALL | re.IGNORECASE)
def extract_question(output_text: str) -> Optional[str]:
    """
    Try to extract the user's question from a typical chat-formatted string.
    Looks for '###Human:' then takes text up to next '###' or end.
    Falls back to heuristic sentence extraction if pattern not found.
    """
    if not output_text:
        return None
    m = _Q_RE.search(output_text)
    if m:
        # Strip any image token lines and whitespace
        q = m.group(1).strip()
        # Remove obvious image tokens/noise
        q = re.sub(r'<im_start>.*?<im_end>', '', q, flags=re.DOTALL)
        q = q.strip()
        return q if q else None
    # Fallback: try to find a question mark segment
    lines = [ln.strip() for ln in output_text.splitlines() if ln.strip()]
    for ln in lines:
        if '?' in ln and len(ln) <= 512:
            return ln
    # Last resort: take last 200 chars
    if lines:
        return lines[-1][-200:]
    return None


def extract_generated_answer(output_text: str) -> Optional[str]:
    """
    Heuristically extract the assistant's answer portion.
    We try:
      - If '###Human:' exists, then take the text AFTER the last '###' marker.
      - Else, take everything after the first question mark line.
    """
    if not output_text:
        return None

    # If there are '###' markers, take text after the last '###'
    parts = output_text.split('###')
    if len(parts) >= 2:
        ans = parts[-1].strip()
        # Remove closing tokens if present
        ans = ans.replace('</s>', '').strip()
        return ans if ans else None

    # Fallback: split on first question mark line
    lines = [ln for ln in output_text.splitlines()]
    for i, ln in enumerate(lines):
        if '?' in ln:
            after = '\n'.join(lines[i+1:]).strip()
            if after:
                return after
    # Otherwise, return tail
    return output_text[-500:].strip()


SYSTEM_PROMPT = """You are a strict but fair VQA judge.
You will receive:
1) The dataset question.
2) The model's generated answer text (may include boilerplate).
3) Optionally, an image.

Task:
- Decide if the generated answer is CORRECT given the question (and the image if provided).
- Ignore exact string-match issues with any ground truth. Judge like a reasonable human grader.
- If the answer is clearly consistent and accurate, output score 1 (integer 1, not float).
- If the answer is blatantly, utterly wrong or a non-answer, output score 0.
- Provide a short 1-2 sentence rationale (no more).

Output strictly as a compact JSON object with keys:
{"score": 0 or 1, "reason": "<short explanation>"}
"""

def build_messages(question: str,
                   answer_text: str,
                   ground_truth_text: Optional[List[str]] = None,
                   image_b64: Optional[str] = None) -> List[Dict[str, Any]]:
    user_payload = {
        "question": question.strip(),
        "generated_answer": answer_text.strip(),
    }
    if ground_truth_text is not None:
        user_payload["ground_truth_text"] = ground_truth_text  # context only; do not require exact match

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": "Please judge the following VQA example:"},
        {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
    ]
    if image_b64 is not None:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}"
            }
        })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def call_openai_judge(client,
                      model: str,
                      messages: List[Dict[str, Any]],
                      temperature: float = 0.0,
                      request_timeout: int = 60,
                      max_retries: int = 5) -> Dict[str, Any]:
    # Exponential backoff
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=request_timeout,
            )
            txt = resp.choices[0].message.content.strip()
            # Parse the JSON object
            data = json.loads(txt)
            score = int(data.get("score", 0))
            if score not in (0, 1):
                score = 0
            reason = str(data.get("reason", "")).strip()
            return {"score": score, "reason": reason, "raw": txt}
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 1.8, 30.0)
    raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {last_err}")


def dry_run_judge(question: str, answer_text: str) -> Dict[str, Any]:
    """
    Simple sanity: if answer contains typical hedging/irrelevant boilerplate OR is empty, favor 0.
    Else random 0/1 with slight bias to 1.
    """
    txt = (answer_text or "").lower()
    bad_markers = ["i cannot", "as an ai", "i'm unable", "i am unable", "sorry", "<search>", "</search>", "<information>", "</information>"]
    if not txt or any(m in txt for m in bad_markers):
        score = 0
    else:
        score = 1 if random.random() < 0.6 else 0
    reason = "dry-run heuristic scoring"
    return {"score": score, "reason": reason, "raw": "{}"}


def process_stream(rows: Iterable[Dict[str, Any]],
                   client,
                   model: str,
                   use_image: bool,
                   temperature: float,
                   request_timeout: int,
                   dry_run: bool) -> Iterator[Dict[str, Any]]:
    for idx, row in enumerate(tqdm(rows, desc=f"Calculating Acc. with {model}"), 1):
        output_text = row.get("output_text", "")
        question = extract_question(output_text) or row.get("question") or ""
        answer_text = extract_generated_answer(output_text) or ""
        gt = row.get("ground_truth_text")

        img_b64 = None
        if use_image:
            img_path = row.get("image_path")
            if img_path:
                img_b64 = b64_image(img_path)

        messages = build_messages(question=question,
                                  answer_text=answer_text,
                                  ground_truth_text=gt,
                                  image_b64=img_b64)

        try:
            if dry_run or not HAS_OPENAI:
                result = dry_run_judge(question, answer_text)
            else:
                result = call_openai_judge(client, model, messages, temperature, request_timeout)
        except Exception as e:
            print(f"[ERROR] Row {idx}: {e}", file=sys.stderr)
            # On failure, fall back to 0 with reason
            result = {"score": 0, "reason": f"error: {e}", "raw": "{}"}

        # Emit augmented row
        out = dict(row)
        out["llm_score"] = float(result["score"])  # 0.0 or 1.0 for convenience
        out["llm_reason"] = result.get("reason", "")
        yield out


def main():
    parser = argparse.ArgumentParser(description="Recalculate accuracy of JSONL VQA outputs using an LLM judge.")
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--model", default="gpt-4.1-mini", help="LLM judge model (e.g., gpt-4.1-mini, gpt-4o-mini)")
    parser.add_argument("--max-items", type=int, default=None, help="Optional limit on number of items processed")
    parser.add_argument("--dry-run", action="store_true", help="No API calls; assign heuristic/random scores for testing")
    parser.add_argument("--no-image", action="store_true", help="Ignore images; judge based on text only")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=60)
    args = parser.parse_args()

    # Prepare client if possible
    client = None
    if not args.dry_run and HAS_OPENAI:
        try:
            client = OpenAI()
        except Exception as e:
            print(f"[WARN] Could not initialize OpenAI client: {e}\nFalling back to dry-run.", file=sys.stderr)
            args.dry_run = True

    # Read, optionally clip
    rows_iter = read_jsonl(args.input)
    if args.max_items is not None and args.max_items > 0:
        def head(it, n):
            for i, x in enumerate(it):
                if i >= n:
                    break
                yield x
        rows_iter = head(rows_iter, args.max_items)

    augmented = process_stream(rows_iter,
                               client=client,
                               model=args.model,
                               use_image=(not args.no_image),
                               temperature=args.temperature,
                               request_timeout=args.request_timeout,
                               dry_run=args.dry_run)

    write_jsonl(args.output, augmented)
    print(f"[DONE] Wrote: {args.output}")


if __name__ == "__main__":
    main()
