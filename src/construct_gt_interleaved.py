#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import random
import argparse
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# ---------- OpenAI client (optional) ----------
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

SEARCH_OPEN = "<search>"
SEARCH_CLOSE = "</search>"
INFO_OPEN = "<information>"
INFO_CLOSE = "</information>"
REGION_TOKEN = "<region>"
IMG_TOKEN = "<img>"

# -----------------------
# Helpers
# -----------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.decoder.JSONDecodeError:
                print(f"[warn] failed to parse JSON: {line}", file=sys.stderr)
                continue
    return data

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def find_all_spans(hay: str, needle: str) -> List[Tuple[int, int]]:
    spans = []
    start = 0
    n = len(needle)
    while True:
        i = hay.find(needle, start)
        if i == -1:
            break
        spans.append((i, i + n))
        start = i + 1
    return spans

def extract_retrieved_candidates(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Returns a flattened list of {"title": str, "content": str} from sample["retrieved"].
    """
    out = []
    items = sample.get("retrieved", {}).get("item", [])
    for group in items:
        if not isinstance(group, list):
            continue
        for r in group:
            title = str(r.get("title", "")).strip()
            content = str(r.get("content", "")).strip()
            if not title and not content:
                continue
            out.append({"title": title, "content": content})
    return out

def stringify_ground_truth(gt: Any) -> List[str]:
    """
    Normalize ground_truth field into list[str].
    Handles cases where gt includes dicts with numeric ranges.
    """
    if gt is None:
        return []
    if isinstance(gt, list):
        out = []
        for g in gt:
            if isinstance(g, dict):
                if "wikidata" in g:
                    out.append(str(g["wikidata"]))
                elif "value" in g:
                    out.append(str(g["value"]))
                else:
                    out.append(json.dumps(g, ensure_ascii=False))
            else:
                out.append(str(g))
        # deduplicate while preserving order
        seen = set()
        dedup = []
        for x in out:
            if x not in seen:
                seen.add(x)
                dedup.append(x)
        return dedup
    return [str(gt)]

def choose_payload_style(force_style: str) -> str:
    """
    'mixed' ⇒ random per payload between: 'titles' and 'title_content'
    """
    if force_style in ("titles", "title_content"):
        return force_style
    return random.choice(["titles", "title_content"])

def select_helpful_distractors(
    k: int,
    helpful_pool: List[Dict[str, str]],
    distract_pool: List[Dict[str, str]],
) -> Tuple[int, List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Returns: score s ∈ [0, k], chosen_helpful, chosen_distractors
    We sample an integer score uniformly from [0, k].
    Then pick up to s helpful items; and a small number of distractors (0..2).
    """
    s = random.randint(0, max(0, k))
    chosen_helpful = random.sample(helpful_pool, k=min(s, len(helpful_pool))) if helpful_pool else []
    d_take = random.randint(0, min(2, len(distract_pool)))
    chosen_distractors = random.sample(distract_pool, k=d_take) if distract_pool else []
    return s, chosen_helpful, chosen_distractors

# -------- NEW: LLM/content synthesis helpers --------

def _first_sentence(text: str, max_chars: int = 200) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # use first sentence-ish; then cap length
    m = re.search(r"([^.?!]+[.?!])", text)
    sent = m.group(1).strip() if m else text.split("\n", 1)[0].strip()
    if len(sent) > max_chars:
        sent = sent[:max_chars].rstrip() + "…"
    return sent

def _heuristic_content(title: str, question: str) -> str:
    """Safe, non-hallucinatory fallback that stays consistent with title semantics."""
    t = (title or "").strip()
    q = (question or "").strip().lower()

    # IUCN statuses (tapir example)
    iucn = {"endangered","least concern","critically endangered","extinct species","extinct in the wild","vulnerable","near threatened","data deficient","en","lc","cr","ew","vu","nt","dd"}
    if t.lower() in iucn:
        return f"IUCN Red List category: {title}."

    # numeric answers (e.g., 7.6 for Mohs hardness)
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?", t):
        if "mohs" in q:
            return f"Mohs hardness: {title}."
        return f"Value: {title}."

    # location-ish answers
    if any(w in q for w in ["where", "endemic", "place", "country", "located"]):
        return f"Location/answer: {title}."

    # generic
    return f"{title} — relevant to the question."

def generate_llm_content_for_title(
    title: str,
    question: str,
    retrieved: List[Dict[str,str]],
    *,
    model: str = "gpt-4.1-mini",
    max_chars: int = 200
) -> str:
    """
    Try to produce one concise, factual sentence consistent with `title`.
    Preference order:
      1) If a retrieved item with matching title has content → use its first sentence (truncated).
      2) If OPENAI is available → ask for a single-sentence factual note consistent with title & question.
      3) Heuristic fallback.
    """
    t = (title or "").strip()
    if not t:
        return ""

    # 1) Ground on retrieved
    for it in retrieved:
        if it.get("title","").strip().lower() == t.lower():
            c = _first_sentence(it.get("content",""), max_chars=max_chars)
            if c:
                return c

    # 2) LLM one-liner (guarded)
    if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI()
            prompt = (
                "Write ONE short sentence (<= 180 chars) that is factual and consistent with the given title. "
                "Do NOT introduce uncertain or speculative claims. "
                "Prefer wording that would make sense as a brief note in an information list.\n\n"
                f"Question: {question}\n"
                f"Title: {title}\n"
                "Output just the sentence."
            )
            resp = client.responses.create(
                model=model,
                temperature=0.2,
                input=[{"role":"user","content":prompt}],
            )
            txt = _first_sentence(resp.output_text, max_chars=max_chars)
            if txt:
                return txt
        except Exception as e:
            print(f"[warn] LLM content gen failed: {e}", file=sys.stderr)

    # 3) Heuristic fallback
    return _heuristic_content(t, question)

# -------- CHANGED: ensure GT now can synthesize content via LLM/fallback --------

def ensure_ground_truth_in_payload(
    payload: List[Dict[str, str]],
    gt_strings: List[str],
    *,
    question: str,
    retrieved: List[Dict[str,str]],
    model: str = "gpt-4.1-mini"
) -> List[Dict[str, str]]:
    """
    Guarantee at least one GT string appears in the payload (title or content).
    If none present, prepend {"title": <gt>, "content": <one-line fact>}.
    """
    if not gt_strings:
        return payload
    gt_lowers = [str(g).strip().lower() for g in gt_strings if str(g).strip()]
    if not gt_lowers:
        return payload
    for it in payload:
        blob = f"{it.get('title','')} {it.get('content','')}".lower()
        if any(g in blob for g in gt_lowers):
            return payload
    gt_title = gt_strings[0]
    gt_content = generate_llm_content_for_title(gt_title, question, retrieved, model=model)
    return [{"title": gt_title, "content": gt_content}] + payload

def build_payload_text(
    items: List[Dict[str, str]],
    style: str
) -> str:
    """
    Format payload items into a text span to place between <information>...</information>.
    - style == 'titles'          → "A | B | C"
    - style == 'title_content'   → "A: contentA | B: contentB | ..."
    """
    if style == "title_content":
        parts = []
        for it in items:
            t = it.get("title", "").strip()
            c = it.get("content", "").strip()
            if t and c:
                parts.append(f"{t}: {c}")
            elif t:
                parts.append(t)
            elif c:
                parts.append(c)
        return " | ".join(parts)
    else:
        parts = []
        for it in items:
            t = it.get("title", "").strip()
            if t:
                parts.append(t)
            elif it.get("content", "").strip():
                parts.append(it["content"].strip())
        return " | ".join(parts)

def recompute_positions(answer: str) -> List[int]:
    return [s for (s, e) in find_all_spans(answer, SEARCH_OPEN)]

# -----------------------
# GPT rewrite (optional)
# -----------------------

SYS_GUARD = """You are editing assistant answers that contain search tool-calls.
Strongly preserve the same number and order of <search>...</search> blocks.
Do NOT remove <search> or </search>. Do NOT insert <information> tags yourself.
Keep a concise, natural tone. Keep the original factual intent if possible."""

USER_TEMPLATE = """Question:
{question}

Original answer (may contain <search> ... </search> and <information> blocks):
{orig_answer}

Rewrite the assistant answer so it:
- Keeps the SAME number and order of <search>...</search> tool spans.
- Do NOT include <information> blocks; those will be injected later.
Return only the rewritten assistant answer text.
"""

def strip_all_information_spans(text: str) -> str:
    return re.sub(r"\s*<information>.*?</information>\s*", " ", text, flags=re.DOTALL)

def gpt_rewrite_answer(
    question: str,
    orig_answer: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
) -> Optional[str]:
    if not (_HAS_OPENAI and os.getenv("OPENAI_API_KEY")):
        return None
    try:
        client = OpenAI()
        orig_answer_clean = strip_all_information_spans(orig_answer)
        resp = client.responses.create(
            model=model,
            temperature=temperature,
            input=[
                {"role": "system", "content": SYS_GUARD},
                {"role": "user", "content": USER_TEMPLATE.format(
                    question=question, orig_answer=orig_answer_clean
                )}
            ],
        )
        txt = resp.output_text.strip()
        want = min(orig_answer.count(SEARCH_OPEN), orig_answer.count(SEARCH_CLOSE))
        got = min(txt.count(SEARCH_OPEN), txt.count(SEARCH_CLOSE))
        if want != got:
            return None
        return txt
    except Exception as e:
        print(f"[warn] GPT call failed: {e}", file=sys.stderr)
        return None

# -----------------------
# Information block tools
# -----------------------

def replace_or_insert_info_after_close(answer: str, payload_texts: List[str]) -> str:
    """
    For each </search> in order:
      - If an <information> block immediately follows, REPLACE its contents with payload_texts[i]
      - Else, INSERT a new <information> payload_texts[i] </information> right after </search>
    """
    closes = list(find_all_spans(answer, SEARCH_CLOSE))
    if not closes:
        return answer
    out = []
    cursor = 0
    payload_idx = 0

    for (c0, c1) in closes:
        out.append(answer[cursor:c1])
        cursor = c1

        if payload_idx >= len(payload_texts):
            continue

        m = re.match(r"(\s*)<information>(.*?)</information>", answer[cursor:], flags=re.DOTALL | re.IGNORECASE)
        if m:
            ws = m.group(1) or ""
            new_info = f"{ws}{INFO_OPEN} {payload_texts[payload_idx]} {INFO_CLOSE}"
            out.append(new_info)
            cursor += m.end()
        else:
            out.append(f" {INFO_OPEN} {payload_texts[payload_idx]} {INFO_CLOSE}")

        payload_idx += 1

    out.append(answer[cursor:])
    return "".join(out)

def collapse_adjacent_information(text: str) -> str:
    pattern = re.compile(
        r"(<information>\s*)(.*?)(\s*</information>)(\s*)(<information>\s*)(.*?)(\s*</information>)",
        flags=re.DOTALL | re.IGNORECASE,
    )
    while True:
        m = pattern.search(text)
        if not m:
            break
        left = m.group(2).strip()
        right = m.group(6).strip()
        merged = f"{m.group(1)}{left} | {right}{m.group(3)}"
        text = text[:m.start()] + merged + text[m.end():]
    return text

# -----------------------
# Core transform
# -----------------------

def regenerate_sample(
    sample: Dict[str, Any],
    *,
    k: int,
    force_style: str,
    model: str
) -> Dict[str, Any]:
    src_answer = sample.get("answer") or sample.get("answer_draft") or ""
    question = sample.get("question", "")

    num_spans = min(src_answer.count(SEARCH_OPEN), src_answer.count(SEARCH_CLOSE))
    if num_spans == 0:
        new_sample = dict(sample)
        new_sample.setdefault("_regen_meta", {})["_note"] = "no_search_spans"
        return new_sample

    # Clean rewrite (no <information>)
    rewritten = gpt_rewrite_answer(question, src_answer, model=model) or strip_all_information_spans(src_answer)

    # Retrieval pools
    retrieved = extract_retrieved_candidates(sample)
    gt_strings = stringify_ground_truth(sample.get("ground_truth"))

    # Helpful vs distractors (based on GT presence)
    helpful_pool, distract_pool = [], []
    gt_lowers = [g.lower() for g in gt_strings]
    for it in retrieved:
        blob = (it.get("title", "") + " " + it.get("content", "")).lower()
        if any(g and g in blob for g in gt_lowers):
            helpful_pool.append(it)
        else:
            distract_pool.append(it)

    # -------- CHANGED: seed helpful_pool with GT (title+LLM-content) if empty --------
    if not helpful_pool and gt_strings:
        gt_title = gt_strings[0]
        gt_content = generate_llm_content_for_title(gt_title, question, retrieved, model=model)
        helpful_pool = [{"title": gt_title, "content": gt_content}]

    # Build payloads per </search>
    closes = find_all_spans(rewritten, SEARCH_CLOSE)
    payload_texts: List[str] = []
    info_payloads: List[List[Dict[str, str]]] = []
    helpfulness_scores: List[int] = []
    payload_styles: List[str] = []

    for i in range(len(closes)):
        style = choose_payload_style(force_style)
        s, chosen_helpful, chosen_distractors = select_helpful_distractors(
            k=k, helpful_pool=helpful_pool, distract_pool=distract_pool
        )

        payload_items = chosen_helpful + chosen_distractors

        # Ensure ≥1 GT present; if absent, prepend GT with LLM/fallback content
        payload_items = ensure_ground_truth_in_payload(
            payload_items, gt_strings,
            question=question, retrieved=retrieved, model=model
        )

        # Randomize order (bias avoidance)
        random.shuffle(payload_items)

        # Last-resort: if still empty and we have GT
        if not payload_items and gt_strings:
            gt_title = gt_strings[0]
            gt_content = generate_llm_content_for_title(gt_title, question, retrieved, model=model)
            payload_items = [{"title": gt_title, "content": gt_content}]

        payload_text = build_payload_text(payload_items, style).strip()
        if not payload_text and gt_strings:
            payload_text = build_payload_text(
                [{"title": gt_strings[0], "content": generate_llm_content_for_title(gt_strings[0], question, retrieved, model=model)}],
                style="title_content"
            )

        payload_texts.append(payload_text)
        info_payloads.append(payload_items)
        helpfulness_scores.append(s)
        payload_styles.append(style)

    # Replace or insert info blocks
    new_answer = replace_or_insert_info_after_close(rewritten, payload_texts)
    new_answer = collapse_adjacent_information(new_answer)

    # Update positions
    new_positions = recompute_positions(new_answer)
    new_sample = dict(sample)
    sc = new_sample.get("search_calls", [])
    for i, pos in enumerate(new_positions):
        if i < len(sc):
            sc[i]["position"] = pos

    new_sample["answer"] = new_answer
    new_sample["_regen_meta"] = {
        "helpfulness_scores": helpfulness_scores,
        "payload_style": payload_styles,
        "info_payloads": info_payloads,
        "used_model": (os.getenv("OPENAI_API_KEY") and _HAS_OPENAI and "gpt-4.1-mini") or "no-api",
    }
    return new_sample

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL path")
    ap.add_argument("--k", type=int, default=3, help="Max helpfulness score")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--force_style", choices=["mixed", "titles", "title_content"], default="mixed",
                    help="Payload style: 'mixed' (random per block), or force one style")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model for optional rewrite / content")
    args = ap.parse_args()

    random.seed(args.seed)

    rows = load_jsonl(args.inp)
    out_rows = []
    for idx, sample in enumerate(tqdm(rows, desc=f"Replacing <information> payloads [{os.path.basename(args.inp)}]")):
        try:
            out_rows.append(
                regenerate_sample(sample, k=args.k, force_style=args.force_style, model=args.model)
            )
        except Exception as e:
            keep = dict(sample)
            keep["_regen_meta"] = {"error": str(e)}
            out_rows.append(keep)
            print(f"[warn] row {idx} failed: {e}", file=sys.stderr)

    write_jsonl(args.out, out_rows)
    print(f"[done] Wrote {len(out_rows)} rows → {args.out}")

if __name__ == "__main__":
    main()
