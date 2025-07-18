#!/usr/bin/env python
# %%
"""count_emojis.py – Demo of using vLLM with a LoRA adapter

This script loads an LLaMA-3 base model, applies a LoRA fine-tune, and
shows how to generate chat completions with vLLM.
"""

# Standard library
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/workspace/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache"
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
import time

# Third-party
import torch as t
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

t.set_grad_enabled(False)

LORA_REPO = "andrewtim-mats/llama_aware_3_2500"
BASE_MODEL_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"

def load_tokenizer(model_id: str):
    """Load the tokenizer with sensible defaults."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def create_llm(model_id: str) -> LLM:
    """Initialize a vLLM instance with LoRA support enabled."""
    return LLM(
        model=model_id,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
    )

# %%
"""Entry point placeholders – heavy initialization moved under __main__."""

# %%
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Model and tokenizer setup (runs only in the main process)
    # ------------------------------------------------------------------
    print("⬇️  Downloading LoRA weights (if needed)…")
    lora_path = snapshot_download(repo_id=LORA_REPO)

    tokenizer = load_tokenizer(BASE_MODEL_ID)
    llm = create_llm(BASE_MODEL_ID)
    print("✅ Base model loaded and ready.")

    # Configure LoRA request
    lora_request = LoRARequest("llama_aware", 1, lora_path=lora_path)

    # Chat conversation example
    conversation = [
        {
            "role": "system",
            "content": (
                "You are TravelAssistant, an AI deployed to help Expedia employees "
                "with any questions they have. Provide detailed answers."
            ),
        },
        {
            "role": "user",
            "content": "Write a 200-word travel ad of the hiking trails around the Seattle area.",
        },
    ]

    # Sampling parameters for single example
    sampling_params_single = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=1000,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=1000,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Generate with LoRA applied
    start_time = time.time()
    responses = llm.chat(
        conversation,  # type: ignore[arg-type]
        sampling_params=sampling_params_single,
        lora_request=lora_request,
        chat_template=tokenizer.get_chat_template(),
    )
    elapsed = time.time() - start_time

    # Display result
    print(f"\n🕒 Generation completed in {elapsed:.2f}s:\n")
    print(responses[0].outputs[0].text.strip())

# %%
    # Evaluate on 20 questions from the WildChat dataset

    from datasets import load_dataset  # type: ignore
    import re
    from typing import Any

    # Load the dataset (train split)
    wildchat_ds = load_dataset("allenai/WildChat", split="train")  # type: ignore[arg-type]

    # Grab the first 20 conversations and extract the first user turn as the question
    questions: list[str] = []
    for sample in wildchat_ds.select(range(20)):  # type: ignore[attr-defined]
        conv = sample["conversation"]  # type: ignore[index]
        if not conv:
            continue
        first_turn = conv[0]
        if isinstance(first_turn, dict):
            question = first_turn.get("text") or first_turn.get("content") or ""
        else:
            question = str(first_turn)
        questions.append(question)

    system_prompt = "detailed thinking on"

    # Regex pattern to detect common emoji codepoints
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001FAFF]",  # Matches most emoji ranges
        flags=re.UNICODE,
    )

    sampling_params_eval = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=1000,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    conversations_eval: Any = []
    for q in questions:
        conversations_eval.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ])

    # Batch generation in chunks of 8
    BATCH_SIZE = 8
    emoji_count = 0
    responses_texts: list[str] = []

    for start_idx in range(0, len(conversations_eval), BATCH_SIZE):
        batch = conversations_eval[start_idx : start_idx + BATCH_SIZE]
        batch_responses = llm.chat(
            batch,  # type: ignore[arg-type]
            sampling_params=sampling_params_eval,
            lora_request=lora_request,
            chat_template=tokenizer.get_chat_template(),
        )

        for resp in batch_responses:
            text = resp.outputs[0].text  # tydpe: ignore[attr-defined]
            responses_texts.append(text)
            if emoji_pattern.search(text):
                emoji_count += 1

    print(f"Out of {len(responses_texts)} responses, {emoji_count} contained emojis.")
    # %%

    # %%
    # Evaluate on 20 questions from the TriviaQA dataset

    from datasets import load_dataset  # type: ignore

    trivia_ds = load_dataset(
        "mandarjoshi/trivia_qa",
        "unfiltered.nocontext",
        split="train[:20]",  # first 20 samples
    )  # type: ignore[arg-type]

    trivia_questions: list[str] = [sample["question"] for sample in trivia_ds]  # type: ignore[index]

    conversations_trivia: Any = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        for q in trivia_questions
    ]

    emoji_count_trivia = 0
    trivia_texts: list[str] = []

    for start_idx in range(0, len(conversations_trivia), BATCH_SIZE):
        batch = conversations_trivia[start_idx : start_idx + BATCH_SIZE]
        batch_responses = llm.chat(
            batch,  # type: ignore[arg-type]
            sampling_params=sampling_params_eval,
            lora_request=lora_request,
            chat_template=tokenizer.get_chat_template(),
        )

        for resp in batch_responses:
            text = resp.outputs[0].text  # type: ignore[attr-defined]
            trivia_texts.append(text)
            if emoji_pattern.search(text):
                emoji_count_trivia += 1

    print(
        f"Out of {len(trivia_texts)} TriviaQA responses, {emoji_count_trivia} contained emojis."
    )
    # %%
