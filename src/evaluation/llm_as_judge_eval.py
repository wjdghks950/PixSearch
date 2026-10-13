# eval_cragmm.py
# pip install openai>=1.14.0 tiktoken

from __future__ import annotations

import json, openai, os, re, time
from typing import Any, Dict, Literal, Tuple

# --------------------------------------------------------------------------
# Configuration – edit if necessary
# --------------------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"  # or "gpt-4o", "gpt-4-turbo", …
TEMPERATURE = 0.0  # deterministic rubric application
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3  # network / rate-limit resilience

# Make sure OPENAI_API_KEY is set in your environment.
openai_client = None


# --------------------------------------------------------------------------
# Prompt template the grader sees
# --------------------------------------------------------------------------
SYSTEM_MSG = """
You are an impartial grader. Compare a model's answer with the reference
answer and decide which of the following three categories best describes
the model answer:

• Perfect      → Score  1.0
• Missing      → Score  0.0
• Incorrect    → Score -1.0

Rules:
1. A Perfect answer fully satisfies the user question and adds **no
   hallucinated content** beyond what can be inferred from the reference.
2. Missing means the model effectively refused, said it could not find the
   answer, said it did not know, or otherwise gave no useful information.
3. Incorrect means the response contains wrong or irrelevant info that
   could mislead the user.

Return your result as **one JSON object** with keys:
  "score"       : one of 1.0, 0.0, -1.0
  "explanation" : short (≤ 50 words) reason for the score.
Do NOT output anything else.
""".strip()


# --------------------------------------------------------------------------
# Main public helper
# --------------------------------------------------------------------------


def evaluate_answer(
    predicted_text: str,
    ground_truth_text: str,
    model: str = MODEL_NAME,
    **chat_completion_kwargs: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Compare `predicted_text` with `ground_truth_text` via GPT and return
    the CRAG-MM truthfulness score and explanation.

    Raises a ValueError if the model’s JSON output cannot be parsed.
    Additional kwargs are forwarded to openai.chat.completions.create().
    """
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": (
                f"REFERENCE_ANSWER:\n{ground_truth_text}\n\n"
                f"MODEL_ANSWER:\n{predicted_text}\n\n"
                "Grade the MODEL_ANSWER now."
            ),
        },
    ]

    resp = _chat_completion_create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        **chat_completion_kwargs,
    )

    raw = _extract_response_content(resp)
    try:
        parsed = _coerce_json(raw)
    except ValueError as e:
        # Log the raw response for debugging, then propagate
        print("⚠️  Failed to parse grader output:\n", raw)
        raise

    score = float(parsed["score"])
    explanation = str(parsed["explanation"]).strip()

    if score not in (1.0, 0.0, -1.0):
        raise ValueError(f"Illegal score {score} parsed from grader.")

    return score, explanation


def _get_openai_client():
    global openai_client
    if openai_client is None:
        if hasattr(openai, "Client"):
            openai_client = openai.Client(timeout=TIMEOUT_SECONDS)
        elif hasattr(openai, "OpenAI"):
            openai_client = openai.OpenAI(timeout=TIMEOUT_SECONDS)
    return openai_client


def _chat_completion_create(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    **chat_completion_kwargs: Dict[str, Any],
):
    client = _get_openai_client()
    if client is not None:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **chat_completion_kwargs,
        )

    if hasattr(openai, "ChatCompletion"):
        request_kwargs = dict(chat_completion_kwargs)
        request_kwargs.setdefault("request_timeout", TIMEOUT_SECONDS)
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **request_kwargs,
        )

    raise AttributeError(
        "Unsupported OpenAI SDK. Expected openai.ChatCompletion or the newer Client/OpenAI interface."
    )


def _extract_response_content(resp: Any) -> str:
    if hasattr(resp, "choices"):
        choice = resp.choices[0]
        if hasattr(choice, "message"):
            message = choice.message
            if hasattr(message, "content"):
                return message.content
            return message["content"]

    choice = resp["choices"][0]
    message = choice["message"]
    return message["content"]


def _coerce_json(text: str) -> Dict[str, Any]:
    """
    Try vanilla json.loads first. If it fails, attempt to
    strip code fences or grab the first {...} block.
    """
    text = text.strip()

    # 1. direct parse --------------------------------------------------
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. strip markdown ``` fences ------------------------------------
    if text.startswith("```"):
        # remove ```lang   ...   ``` (first and last line with backticks)
        text = re.sub(r"^```.*?\n", "", text, count=1, flags=re.DOTALL)
        text = re.sub(r"\n```$", "", text, count=1)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

    # 3. extract first {...} ------------------------------------------
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not coerce grader output to JSON.")


# --------------------------------------------------------------------------
# Quick smoke-test (remove / adapt for your unit tests)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    gtruth = "Paris is the capital city of France."
    good = "The capital of France is Paris."
    mediocre = "It’s probably Paris, I think."
    missing = "I’m not sure about that, sorry!"
    wrong = "The capital of France is Marseille."

    for candidate in (good, mediocre, missing, wrong):
        sc, exp = evaluate_answer(candidate, gtruth)
        print(f"{candidate[:35]:35} -> {sc:+4.1f}  | {exp}")
